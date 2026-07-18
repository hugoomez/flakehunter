"""Codex-backed fix generation for auto-fixable flaky tests.

The fixer takes a :class:`RootCause` produced by the classifier (plus the
:class:`FlakeVerdict` that motivated it) and asks the Codex SDK to repair the
underlying defect, returning a :class:`FixProposal` whose ``diff`` the verifier
can apply and measure.

FlakeHunter does not trust the fix
----------------------------------
This module's job is to produce a *candidate* and to fail loudly when it
cannot. Every failure mode raises rather than returning a plausible-looking
``FixProposal``, because a garbage proposal that reaches the verifier costs a
full statistical re-run battery to reject, and an empty one would be
"verified" as a no-op change that did not help:

* a category the SPEC marks detect-only raises :class:`NotAutoFixableError`
  *before* a Codex client is constructed, so no tokens are spent;
* a turn that ends in any status other than ``completed`` raises
  :class:`CodexTurnError` rather than falling through to diff computation;
* a turn that completes without changing anything raises
  :class:`EmptyFixError`;
* a diff that ``git apply`` will not accept raises :class:`FixerError`.

How a failed turn actually surfaces
-----------------------------------
``Thread.run`` does **not** return a ``TurnResult`` with
``status=TurnStatus.failed``. The SDK's ``_raise_for_failed_turn``
(``openai_codex/_run.py:59``) converts a failed turn into a bare
``RuntimeError`` carrying the server's message, and a stream that ends without
a completion event raises ``RuntimeError("turn completed event not received")``.
Only ``interrupted`` and ``in_progress`` are ever *returned*. So the two paths
are handled in two places — ``_run_codex`` catches the raise, and
``_require_completed`` checks the returned status — and both funnel into
:class:`CodexTurnError` so callers have exactly one exception to catch.

This was found by running the real E2E test, not by reading the type
signatures: ``TurnStatus.failed`` exists in the enum and looks returnable, and
a fake client that returns it is entirely plausible. It is the reason the fake
in ``test_fixer.py`` now models the *raise*.

Why the diff is computed from a snapshot, not from ``FileChangeThreadItem``
--------------------------------------------------------------------------
``TurnResult.items`` carries ``FileChangeThreadItem``s whose
``FileUpdateChange.diff`` is already a unified diff string, so reconstructing
the diff from them is possible. We compute our own instead, by snapshotting
every ``*.py`` file before and after the turn and diffing with
:mod:`difflib`, for three reasons:

* **Net effect for free.** Several patches may touch one file across a turn;
  they would have to be composed in order to recover the final state. A
  before/after snapshot *is* the net result by construction.
* **Disk is truth.** ``PatchApplyStatus`` can be ``failed`` or ``declined`` —
  a change item can exist for a patch that never landed.
* **We control the format.** ``difflib`` plus a ``git apply --check`` gate
  gives a diff we have actually validated, in the shape ``FixProposal.diff``
  promises.

The change items are still read, as a *cross-check* (:func:`_patch_notes`):
they reveal patches Codex could not apply and files it reports touching that
the ``*.py`` snapshot did not capture. Those become notes on the rationale
rather than being silently dropped.

Sandboxing
----------
Per the SPEC's non-negotiable ("diffs are applied to a temporary copy of the
target repo, never in place"), the git-tracked working tree is copied to a
temporary directory and Codex's ``cwd`` points there — never at ``repo_root``.
Copying the whole tracked tree rather than the single test file is what lets
Codex resolve the imports under test, run pytest to check itself, and add a
``conftest.py`` when a shared-state fix needs one.
"""

from __future__ import annotations

import difflib
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from openai_codex import ApprovalMode, Codex, CodexError, Sandbox
from openai_codex.types import TurnStatus

from flakehunter.contracts import (
    Fixer as FixerContract,
    FixProposal,
    FlakeVerdict,
    RootCause,
)

# SPEC categories A/B/C. D (timing) and E (external) are detect-only.
_AUTO_FIXABLE_CATEGORIES = frozenset({"order", "shared_state", "randomness"})
# Directories that never belong in a snapshot: VCS metadata, virtualenvs, and
# caches pytest may leave behind if Codex runs the suite to check itself.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    }
)
# PatchApplyStatus values meaning Codex tried to patch and did not succeed.
_UNAPPLIED_PATCH_STATUSES = frozenset({"failed", "declined"})
_SUBPROCESS_TIMEOUT_S = 60.0


class FixerError(Exception):
    """Base class for every way fix generation can fail."""


class NotAutoFixableError(FixerError):
    """Raised for SPEC categories D (timing) and E (external)."""


class CodexTurnError(FixerError):
    """Raised when the Codex turn ends in any status other than ``completed``."""


class EmptyFixError(FixerError):
    """Raised when the turn completed but changed nothing."""


_CONSTRAINTS = """\
Fix the ROOT CAUSE. Hard constraints - a fix violating any of these is
rejected automatically:
- Do NOT delete, weaken, or loosen any assertion; the assertion count must
  not decrease and comparisons must not become laxer.
- Do NOT add `skip`, `xfail`, retries, reruns, or `try/except` around the
  assertion.
- Do NOT make the test pass by changing what it checks.
- Change as little as possible; do not reformat unrelated code.
- Edit only files under the current working directory."""

_CATEGORY_INSTRUCTIONS: Mapping[str, str] = {
    "randomness": """\
The test consumes an unseeded RNG, so its outcome varies run to run. Make the
RNG deterministic at the point of use: seed it with a literal constant
(`random.seed(0)`, `numpy.random.default_rng(0)`), or inject a seeded
`random.Random(0)` instance. Prefer a fixture over a bare module-level call so
the seed does not leak into other tests. The assertion must remain exactly as
it is.""",
    "order": """\
The test fails in isolation and passes inside the suite: it depends on state
that an earlier test happens to leave behind. Make it self-sufficient - perform
its own setup (a fixture, or an explicit arrange block) instead of relying on
execution order. Do NOT force ordering with `pytest-order`, `pytest-ordering`,
or a dependency marker; that hides the coupling instead of removing it.""",
    "shared_state": """\
The test passes alone and fails inside the suite: mutable state (a module-level
global, singleton, cache, or temp file) is being polluted by another test.
Isolate it - reset the state in an autouse fixture, use `monkeypatch`, or
construct a fresh instance per test - so the test's outcome does not depend on
what ran before it. Do NOT fix this by reordering tests.""",
}


class Fixer(FixerContract):
    """Generate a fix for an auto-fixable flaky test with the Codex SDK."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        model: str | None = None,
        codex_factory: Callable[[], Any] = Codex,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.model = model
        # The seam the tests' fake client plugs into. Defaults to the real
        # `Codex` class, which `propose_fix` calls and uses as a context
        # manager exactly as the SDK intends.
        self._codex_factory = codex_factory

    def propose_fix(self, cause: RootCause, verdict: FlakeVerdict) -> FixProposal:
        self._require_auto_fixable(cause)
        source_rel = self._source_rel_path(cause.test_id)

        # `pristine` is a second, untouched copy: `git apply --check` needs a
        # tree in the pre-fix state, and `work` is no longer in it by then.
        with _temporary_dir() as tmp_root:
            work = tmp_root / "work"
            pristine = tmp_root / "pristine"
            self._copy_tree(work)
            before = _snapshot(work)

            prompt = self._build_prompt(cause, verdict, source_rel)
            session_id, result = self._run_codex(prompt, work)

            after = _snapshot(work)
            diff = _unified_tree_diff(before, after)
            files_touched = sorted(_changed_paths(before, after))
            if not diff:
                raise EmptyFixError(
                    f"Codex completed its turn for {cause.test_id!r} without "
                    f"changing any Python file (session {session_id})"
                )

            self._copy_tree(pristine)
            _git_apply_check(diff, pristine)

            notes = _patch_notes(result, work, files_touched, source_rel)
            return FixProposal(
                test_id=cause.test_id,
                diff=diff,
                rationale=_rationale(result.final_response, cause, files_touched, notes),
                files_touched=files_touched,
                codex_session_id=session_id,
            )

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    @staticmethod
    def _require_auto_fixable(cause: RootCause) -> None:
        """Refuse detect-only categories before any Codex client is built.

        Both fields are checked. ``auto_fixable`` is derived from ``category``
        by the classifier (``classifier.py:289``), so a disagreement means a
        hand-built or corrupted ``RootCause`` rather than a real verdict, and
        guessing which field to believe is exactly the kind of silent recovery
        this module refuses to do.
        """
        expected = cause.category in _AUTO_FIXABLE_CATEGORIES
        if not expected:
            raise NotAutoFixableError(
                f"{cause.test_id!r} was classified {cause.category!r}, which the "
                f"SPEC marks detect-only (categories D/E): a fix is suggested for "
                f"review, never generated automatically. Auto-fixable categories "
                f"are {sorted(_AUTO_FIXABLE_CATEGORIES)}"
            )
        if cause.auto_fixable is not True:
            raise NotAutoFixableError(
                f"inconsistent RootCause for {cause.test_id!r}: category "
                f"{cause.category!r} is auto-fixable but auto_fixable is "
                f"{cause.auto_fixable!r}; refusing to guess which is correct"
            )

    def _source_rel_path(self, test_id: str) -> str:
        """Resolve ``path::name`` to a repo-relative POSIX source path.

        Mirrors the convention in ``classifier.py:398`` (``_source_path``):
        a test id's path component is relative to the repo root.
        """
        path_component = test_id.split("::", 1)[0]
        if not path_component:
            raise FixerError(f"cannot resolve a source file from test id {test_id!r}")

        candidate = Path(path_component)
        absolute = (
            candidate if candidate.is_absolute() else self.repo_root / candidate
        )
        if not absolute.is_file():
            raise FixerError(
                f"source file for test {test_id!r} not found at {absolute}"
            )
        try:
            return absolute.resolve().relative_to(
                self.repo_root.resolve()
            ).as_posix()
        except ValueError as exc:
            raise FixerError(
                f"source file for test {test_id!r} ({absolute}) lies outside the "
                f"repo root {self.repo_root}"
            ) from exc

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_prompt(
        self, cause: RootCause, verdict: FlakeVerdict, source_rel: str
    ) -> str:
        sections = [
            f"The test `{cause.test_id}` in `{source_rel}` is flaky: it failed "
            f"{verdict.n_failures}/{verdict.n_runs} times on an unchanged commit "
            f"(failure rate {verdict.failure_rate:.1%}, Wilson 95% CI upper bound "
            f"{verdict.ci95_upper:.1%}).",
            f"Diagnosis: {cause.category} (confidence {cause.confidence:.2f}).",
            _bullets("Evidence:", cause.evidence),
            _bullets("Static signals:", cause.ast_signals),
            _bullets("Sample failures:", verdict.sample_tracebacks),
            _CATEGORY_INSTRUCTIONS[cause.category],
            _CONSTRAINTS,
        ]
        return "\n\n".join(section for section in sections if section)

    # ------------------------------------------------------------------
    # Codex execution
    # ------------------------------------------------------------------
    def _run_codex(self, prompt: str, cwd: Path) -> tuple[str, Any]:
        with self._codex_factory() as codex:
            thread = codex.thread_start(
                model=self.model,
                sandbox=Sandbox.workspace_write,
                # Writes inside `cwd` are already permitted by workspace_write
                # and never escalate, so denying escalations costs nothing and
                # keeps a sandbox-escape attempt from blocking on approval in a
                # context where nobody can answer.
                approval_mode=ApprovalMode.deny_all,
                cwd=str(cwd),
            )
            # Captured before run() so a failed turn can still name its session.
            session_id = thread.id
            try:
                result = thread.run(prompt)
            except (RuntimeError, CodexError) as exc:
                # `Thread.run` does not hand back a TurnStatus.failed result:
                # the SDK's `_raise_for_failed_turn` turns a failed turn into a
                # bare RuntimeError carrying the server's message (a usage
                # limit, say), and a dropped stream raises "turn completed event
                # not received". Only `interrupted`/`in_progress` are ever
                # returned, and `_require_completed` catches those. Both paths
                # must arrive as CodexTurnError so callers have one thing to
                # catch.
                raise CodexTurnError(
                    f"Codex turn failed for session {session_id}: {exc}"
                ) from exc

        _require_completed(result, session_id)
        return session_id, result

    # ------------------------------------------------------------------
    # Sandbox construction
    # ------------------------------------------------------------------
    def _copy_tree(self, dest: Path) -> None:
        """Copy the git-tracked working tree to ``dest``.

        Tracked files only: the demo repo carries an 87MB ``.venv`` that must
        never be copied, and ``git ls-files`` excludes it for free. Falls back
        to a filtered ``copytree`` when ``repo_root`` is not a git repo.
        """
        dest.mkdir(parents=True, exist_ok=True)
        tracked = _git_tracked_files(self.repo_root)
        if tracked is None:
            shutil.copytree(
                self.repo_root,
                dest,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*_IGNORED_DIRS),
            )
            return

        for rel in tracked:
            source = self.repo_root / rel
            # A tracked path can be absent from the working tree (deleted but
            # not staged) or be a submodule directory; skip rather than crash.
            if not source.is_file():
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


# ----------------------------------------------------------------------
# Turn-result handling
# ----------------------------------------------------------------------
def _require_completed(result: Any, session_id: str) -> None:
    """Accept only ``TurnStatus.completed``; every other status raises."""
    if result.status is TurnStatus.completed:
        return

    status = getattr(result.status, "value", result.status)
    detail = ""
    error = getattr(result, "error", None)
    if error is not None:
        detail = f": {error.message}"
        additional = getattr(error, "additional_details", None)
        if additional:
            detail += f" ({additional})"
    raise CodexTurnError(
        f"Codex turn ended with status {status!r} rather than 'completed' "
        f"(session {session_id}){detail}"
    )


def _patch_notes(
    result: Any, work: Path, files_touched: list[str], source_rel: str
) -> list[str]:
    """Cross-check ``TurnResult.items`` against the snapshot-derived diff.

    Reads ``FileChangeThreadItem``s structurally (``type == "fileChange"``)
    rather than by ``isinstance``, so this does not depend on the SDK's
    generated module layout.
    """
    notes: list[str] = []
    reported: set[str] = set()

    for item in getattr(result, "items", []) or []:
        change_item = getattr(item, "root", item)
        if getattr(change_item, "type", None) != "fileChange":
            continue
        status = getattr(change_item, "status", None)
        status_value = getattr(status, "value", status)
        for change in getattr(change_item, "changes", []) or []:
            rel = _relative_to_sandbox(getattr(change, "path", ""), work)
            if rel:
                reported.add(rel)
            if status_value in _UNAPPLIED_PATCH_STATUSES:
                notes.append(
                    f"Codex reported a {status_value} patch for "
                    f"{rel or getattr(change, 'path', '?')}; that change is not "
                    f"part of this diff"
                )

    missed = sorted(reported - set(files_touched))
    if missed:
        notes.append(
            "Codex reported changing "
            + ", ".join(missed)
            + ", which the *.py snapshot did not capture; the diff may be "
            "incomplete for those paths"
        )

    out_of_scope = sorted(set(files_touched) - {source_rel})
    if out_of_scope:
        notes.append(
            "the fix reaches beyond the target test file into "
            + ", ".join(out_of_scope)
        )
    return notes


def _rationale(
    final_response: str | None,
    cause: RootCause,
    files_touched: list[str],
    notes: list[str],
) -> str:
    # `final_response` is `str | None`; a completed turn that says nothing must
    # still yield a rationale a reviewer can read.
    body = (final_response or "").strip() or (
        f"Codex completed a {cause.category} fix for {cause.test_id} without "
        f"a final response; changed: {', '.join(files_touched)}."
    )
    if not notes:
        return body
    return body + "\n\nFixer notes:\n" + "\n".join(f"- {note}" for note in notes)


# ----------------------------------------------------------------------
# Snapshot / diff helpers
# ----------------------------------------------------------------------
def _snapshot(root: Path) -> dict[str, str]:
    """Map repo-relative POSIX path -> text for every ``*.py`` under ``root``."""
    snapshot: dict[str, str] = {}
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRS for part in relative.parts):
            continue
        if not path.is_file():
            continue
        try:
            snapshot[relative.as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return snapshot


def _changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> set[str]:
    return {
        rel
        for rel in set(before) | set(after)
        if before.get(rel) != after.get(rel)
    }


def _unified_tree_diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    """Concatenate per-file unified diffs in sorted path order.

    Added files use a ``--- /dev/null`` header and deleted files a
    ``+++ /dev/null`` one; ``git apply`` accepts both as create/delete.
    """
    chunks: list[str] = []
    for rel in sorted(_changed_paths(before, after)):
        old = before.get(rel)
        new = after.get(rel)
        chunk = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True) if old is not None else [],
                new.splitlines(keepends=True) if new is not None else [],
                fromfile=f"a/{rel}" if old is not None else "/dev/null",
                tofile=f"b/{rel}" if new is not None else "/dev/null",
            )
        )
        if not chunk:
            continue
        # A source file whose last line lacks a trailing newline would
        # otherwise run into the next file's header. `git apply --check`
        # is the backstop that rejects a diff this cannot rescue.
        if not chunk.endswith("\n"):
            chunk += "\n"
        chunks.append(chunk)
    return "".join(chunks)


def _relative_to_sandbox(path: str, work: Path) -> str:
    """Normalise a Codex-reported path to a sandbox-relative POSIX path."""
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(work.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


# ----------------------------------------------------------------------
# git helpers
# ----------------------------------------------------------------------
def _git_tracked_files(root: Path) -> list[str] | None:
    """Tracked paths relative to ``root``, or None if ``root`` is not a repo."""
    try:
        process = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return None
    decoded = process.stdout.decode("utf-8", errors="replace")
    return [entry for entry in decoded.split("\0") if entry]


def _git_apply_check(diff: str, root: Path) -> None:
    """Prove the diff is applicable, as ``FixProposal.diff`` promises."""
    try:
        process = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=root,
            input=diff.encode("utf-8"),
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FixerError(
            f"could not validate the generated diff with `git apply --check`: {exc}"
        ) from exc
    if process.returncode != 0:
        raise FixerError(
            "the generated diff is not applicable via `git apply`: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
def _bullets(heading: str, lines: Iterable[str]) -> str:
    items = [line for line in lines if line]
    if not items:
        return ""
    return heading + "\n" + "\n".join(f"- {item}" for item in items)


def _temporary_dir() -> "_TempDir":
    return _TempDir(tempfile.mkdtemp(prefix="flakehunter-fix-"))


class _TempDir:
    """A ``TemporaryDirectory`` that yields a ``Path`` and never fails cleanup.

    Codex may leave read-only files or live file handles behind on Windows, and
    a cleanup error must not mask the fix we just computed.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def __enter__(self) -> Path:
        return self._path

    def __exit__(self, *exc_info: object) -> None:
        shutil.rmtree(self._path, ignore_errors=True)
