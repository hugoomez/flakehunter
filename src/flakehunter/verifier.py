"""The Verification Contract: prove a fix works before FlakeHunter trusts it.

The verifier is the component the whole project exists to justify. It takes a
:class:`FixProposal` the fixer already produced and the :class:`FlakeVerdict`
that motivated it, and decides whether the fix is *real* or merely
plausible-looking, under a contract it never relaxes (``docs/SPEC.md``):

1. **Structural** — the fix must not weaken the test. Assertion count may not
   decrease, and no ``skip``/``xfail``/``flaky``/rerun marker or
   assertion-masking ``try/except`` may be introduced.
2. **Statistical** — after applying the fix, the target must re-run
   ``N_verify`` times with a post-fix failure rate whose confidence bound falls
   below a target threshold, *and* the before/after improvement must be
   significant by Fisher's exact test — numerically lower is not enough.
3. **Regression** — nothing that passed before the fix may fail after it.

A fix is ``verified_fix`` only if all three hold; otherwise it is rejected with
the specific reason.

Why this module never imports the Fixer or Codex
------------------------------------------------
``verify(fix, before)`` consumes an *already generated* ``FixProposal``; it does
not create one, so it has no reason to touch Codex. That is deliberate and
enforced: the Fixer module pulls in the Codex SDK at load time, so borrowing even
a helper from it would drag that SDK into the verifier and its tests. The small
git-tracked-copy / ``git apply`` helpers below
are therefore re-implemented locally rather than borrowed from ``fixer.py``. A
future refactor could extract them into a neutral ``sandbox.py`` shared by both;
that is out of scope here.

Statistical subtleties worth stating plainly
--------------------------------------------
* **The gate for a clean re-run is the rule of three, not Wilson.**
  ``N_verify = n_for_verification(threshold)`` is the inverse of
  :func:`rule_of_three_upper` (``ceil(3 / threshold)``), so
  ``rule_of_three_upper(N_verify) <= threshold`` holds by construction. The
  Wilson upper bound for zero failures in the same ``N`` is slightly *higher*
  (e.g. ``0.0249`` vs ``0.02`` at ``threshold=0.02``, ``N=150``), so gating the
  stored Wilson bound directly would reject an otherwise-perfect zero-failure
  fix. We therefore gate the ``k_after == 0`` case on ``rule_of_three_upper(N)``
  and fall back to the Wilson bound only when there were failures. The stored
  ``ci95_upper_after`` is always the honest Wilson value (matching ``Detector``),
  which means it can read slightly above ``threshold`` on a verified fix — that
  is intentional, not a bug.
* **The re-run is in-suite.** ``verify`` receives no root-cause category, and an
  order/shared-state flake only reproduces inside the suite — re-running the
  target in isolation would show a false zero-failure "pass". So the statistical
  battery runs the whole ``suite_test_ids`` batch (unseeded, randomized order)
  and watches the target, mirroring ``Classifier._count_suite_failures``.
  Randomness flakes surface in-suite too, so this condition covers every case.

Known limitation
----------------
The regression check compares a single pre-fix suite run against a single
post-fix one, so a *pre-existing* flaky sibling that happens to flip pass->fail
across the two runs can produce a false ``rejected_breaks_suite``. Callers pass
an explicit ``suite_test_ids``; scoping it to the tests that actually matter (and
that are themselves stable) keeps the check trustworthy. Re-confirming a
candidate regression with repeat runs would remove the noise, but the SPEC says
"once".
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from flakehunter.contracts import (
    FixProposal,
    FlakeVerdict,
    RunResult,
    SandboxRunner as SandboxRunnerContract,
    Verifier as VerifierContract,
    VerificationResult,
)
from flakehunter.runner import SandboxRunner
from flakehunter.stats import (
    fisher_exact_pvalue,
    n_for_verification,
    rule_of_three_upper,
    wilson_interval,
)

# Directories that never belong in a tree copy (VCS metadata, virtualenvs, and
# caches). Mirrors fixer.py's list so the demo's 87MB .venv is never copied.
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
_SUBPROCESS_TIMEOUT_S = 60.0
_ALPHA = 0.05
_FAILURE_OUTCOMES = frozenset({"failed", "error"})

# Assertion-equivalent call targets, matched on the final attribute name so
# `self.assertEqual`, `pytest.raises`, and `pt.raises` all count regardless of
# how they were imported or bound.
_ASSERTION_METHODS = frozenset(
    {
        "raises",
        "warns",
        "assertEqual",
        "assertNotEqual",
        "assertTrue",
        "assertFalse",
        "assertIs",
        "assertIsNot",
        "assertIsNone",
        "assertIsNotNone",
        "assertIn",
        "assertNotIn",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertAlmostEqual",
        "assertNotAlmostEqual",
        "assertRaises",
        "assertRaisesRegex",
        "assertWarns",
        "assertWarnsRegex",
        "assertRegex",
        "assertNotRegex",
        "assertListEqual",
        "assertDictEqual",
        "assertSetEqual",
        "assertTupleEqual",
        "assertCountEqual",
        "assertMultiLineEqual",
        "assertSequenceEqual",
    }
)
# Tokens that mark a decorator/marker as masking a failure. `repeat` (a benign
# rerun-for-speed marker) and `fixture`/`autouse` (legitimate setup) are pointedly
# absent, so an autouse cleanup fixture added by a real fix never trips this.
_MASKING_TOKENS = ("skip", "xfail", "flaky", "rerun")


class VerifierError(Exception):
    """A broken input to verification (un-appliable diff, unresolvable target).

    Distinct from the four contract verdicts, which all mean "the fix was
    measured and judged". This means "the fix could not be measured at all".
    """


class Verifier(VerifierContract):
    """Enforce the Verification Contract against an already-generated fix."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        suite_test_ids: list[str],
        threshold: float = 0.02,
        timeout_s: float = 30.0,
        runner_factory: Callable[[Path], SandboxRunnerContract] | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.repo_root = Path(repo_root)
        self.suite_test_ids = list(suite_test_ids)
        self.threshold = threshold
        self.timeout_s = timeout_s
        # The test seam (mirrors the Fixer's factory seam): builds a runner bound to
        # a specific temp-copy path, because pytest must run inside the applied /
        # pristine copies, not the real repo.
        self._runner_factory = runner_factory or self._default_runner_factory

    def _default_runner_factory(self, cwd: Path) -> SandboxRunnerContract:
        return SandboxRunner(cwd=cwd, timeout_s=self.timeout_s)

    # ------------------------------------------------------------------
    # Contract entry point
    # ------------------------------------------------------------------
    def verify(self, fix: FixProposal, before: FlakeVerdict) -> VerificationResult:
        target = fix.test_id
        if before.test_id != target:
            raise VerifierError(
                f"verdict is for {before.test_id!r} but the fix targets {target!r}"
            )
        if target not in self.suite_test_ids:
            raise VerifierError(
                f"target {target!r} is not in suite_test_ids; the statistical and "
                f"regression checks re-run the suite and watch the target, so it "
                f"must be part of it"
            )

        source_rel = self._source_rel_path(target)
        n_verify = n_for_verification(self.threshold)

        with _temporary_dir() as tmp_root:
            pristine = tmp_root / "pristine"
            work = tmp_root / "work"
            _copy_tracked_tree(self.repo_root, pristine)
            _copy_tracked_tree(self.repo_root, work)

            before_src = (pristine / source_rel).read_text(encoding="utf-8")
            _git_apply(fix.diff, work)
            after_src = (work / source_rel).read_text(encoding="utf-8")

            # --- 1. Structural (cheap, no pytest): fail fast --------------
            assertion_preserved, no_skip = _structural_check(
                before_src, after_src, _test_function_name(target)
            )
            if not (assertion_preserved and no_skip):
                # No measurement taken; make no claim beyond `before`.
                return VerificationResult(
                    test_id=target,
                    contract_passed=False,
                    assertion_count_preserved=assertion_preserved,
                    no_skip_introduced=no_skip,
                    failure_rate_before=before.failure_rate,
                    failure_rate_after=before.failure_rate,
                    ci95_upper_after=before.ci95_upper,
                    suite_still_green=False,
                    verdict="rejected_weakens_test",
                )

            work_runner = self._runner_factory(work)

            # --- 2. Statistical: re-run in-suite N times -----------------
            k_after = self._count_target_failures(work_runner, target, n_verify)
            failure_rate_after = k_after / n_verify
            _, ci95_upper_after = wilson_interval(k=k_after, n=n_verify)
            gate_bound = (
                rule_of_three_upper(n_verify)
                if k_after == 0
                else ci95_upper_after
            )
            p_value = fisher_exact_pvalue(
                before.n_failures, before.n_runs, k_after, n_verify
            )
            statistically_fixed = gate_bound <= self.threshold and p_value < _ALPHA
            if not statistically_fixed:
                return VerificationResult(
                    test_id=target,
                    contract_passed=False,
                    assertion_count_preserved=True,
                    no_skip_introduced=True,
                    failure_rate_before=before.failure_rate,
                    failure_rate_after=failure_rate_after,
                    ci95_upper_after=ci95_upper_after,
                    suite_still_green=False,
                    verdict="rejected_still_flaky",
                )

            # --- 3. Regression: nothing that passed may now fail ---------
            pristine_runner = self._runner_factory(pristine)
            suite_still_green = self._suite_still_green(
                pristine_runner, work_runner, target
            )
            if not suite_still_green:
                return VerificationResult(
                    test_id=target,
                    contract_passed=False,
                    assertion_count_preserved=True,
                    no_skip_introduced=True,
                    failure_rate_before=before.failure_rate,
                    failure_rate_after=failure_rate_after,
                    ci95_upper_after=ci95_upper_after,
                    suite_still_green=False,
                    verdict="rejected_breaks_suite",
                )

            # --- 4. All three passed -------------------------------------
            return VerificationResult(
                test_id=target,
                contract_passed=True,
                assertion_count_preserved=True,
                no_skip_introduced=True,
                failure_rate_before=before.failure_rate,
                failure_rate_after=failure_rate_after,
                ci95_upper_after=ci95_upper_after,
                suite_still_green=True,
                verdict="verified_fix",
            )

    # ------------------------------------------------------------------
    # Statistical / regression run helpers
    # ------------------------------------------------------------------
    def _count_target_failures(
        self, runner: SandboxRunnerContract, target: str, n: int
    ) -> int:
        failures = 0
        for _ in range(n):
            results = runner.run_once(
                self.suite_test_ids,
                seed=None,
                forked=False,
                randomize_order=True,
            )
            if _target_failed(results, target):
                failures += 1
        return failures

    def _suite_still_green(
        self,
        pristine_runner: SandboxRunnerContract,
        work_runner: SandboxRunnerContract,
        target: str,
    ) -> bool:
        baseline = pristine_runner.run_once(
            self.suite_test_ids, seed=None, forked=False, randomize_order=True
        )
        after = work_runner.run_once(
            self.suite_test_ids, seed=None, forked=False, randomize_order=True
        )
        baseline_pass = {
            r.test_id for r in baseline if r.outcome == "passed"
        } - {target}
        after_outcome = {r.test_id: r.outcome for r in after}
        # A baseline-passing sibling that is no longer 'passed' (failed, errored,
        # newly skipped, or dropped from the report) is a regression.
        return all(after_outcome.get(tid) == "passed" for tid in baseline_pass)

    # ------------------------------------------------------------------
    # Source resolution
    # ------------------------------------------------------------------
    def _source_rel_path(self, test_id: str) -> str:
        """Resolve ``path::name`` to a repo-relative POSIX source path.

        Mirrors the convention in ``classifier.py`` / ``fixer.py``: a test id's
        path component is relative to the repo root.
        """
        path_component = test_id.split("::", 1)[0]
        if not path_component:
            raise VerifierError(
                f"cannot resolve a source file from test id {test_id!r}"
            )
        candidate = Path(path_component)
        absolute = (
            candidate if candidate.is_absolute() else self.repo_root / candidate
        )
        if not absolute.is_file():
            raise VerifierError(
                f"source file for test {test_id!r} not found at {absolute}"
            )
        try:
            return (
                absolute.resolve()
                .relative_to(self.repo_root.resolve())
                .as_posix()
            )
        except ValueError as exc:
            raise VerifierError(
                f"source file for test {test_id!r} ({absolute}) lies outside the "
                f"repo root {self.repo_root}"
            ) from exc


# ----------------------------------------------------------------------
# Outcome interpretation
# ----------------------------------------------------------------------
def _target_failed(results: list[RunResult], target: str) -> bool:
    for result in results:
        if result.test_id == target:
            return result.outcome in _FAILURE_OUTCOMES
    # The target did not appear in the report (suite-level error or timeout).
    # Treat that as a failure for the target under test, as the classifier does.
    return True


# ----------------------------------------------------------------------
# Structural (AST) check
# ----------------------------------------------------------------------
def _structural_check(
    before_src: str, after_src: str, test_name: str | None
) -> tuple[bool, bool]:
    """Return ``(assertion_count_preserved, no_skip_introduced)``.

    Scoped to the target test function when it can be located, so an autouse
    fixture or helper added elsewhere in the file does not distort the counts.
    """
    before_tree = ast.parse(before_src)
    after_tree = ast.parse(after_src)
    before_fn = _find_test_function(before_tree, test_name)
    after_fn = _find_test_function(after_tree, test_name)

    before_scope: ast.AST = before_fn if before_fn is not None else before_tree
    after_scope: ast.AST = after_fn if after_fn is not None else after_tree

    assertion_preserved = _count_assertions(after_scope) >= _count_assertions(
        before_scope
    )

    before_masking = _masking_markers(before_tree, before_fn)
    after_masking = _masking_markers(after_tree, after_fn)
    introduced_marker = bool(after_masking - before_masking)
    introduced_masking_try = _masking_try_count(after_scope) > _masking_try_count(
        before_scope
    )
    no_skip = not (introduced_marker or introduced_masking_try)

    return assertion_preserved, no_skip


def _find_test_function(
    tree: ast.Module, test_name: str | None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Locate the test's own def, including inside a test class."""
    if test_name is None:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == test_name
        ):
            return node
    return None


def _test_function_name(test_id: str) -> str | None:
    """Extract the test function's name from ``path::[Class::]name[param]``."""
    parts = test_id.split("::")
    if len(parts) < 2:
        return None
    return parts[-1].split("[", 1)[0] or None


def _count_assertions(node: ast.AST) -> int:
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            count += 1
        elif isinstance(child, ast.Call):
            name = _call_last_name(child.func)
            if name in _ASSERTION_METHODS:
                count += 1
    return count


def _call_last_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _masking_markers(
    tree: ast.Module, func: ast.FunctionDef | ast.AsyncFunctionDef | None
) -> set[str]:
    """Masking decorator/marker dotted names on the function and at module level.

    Only ``skip``/``xfail``/``flaky``/rerun constructs are collected; a benign
    decorator (``@pytest.fixture``, ``@pytest.mark.parametrize``) never appears
    here, so adding one is not a violation.
    """
    names: set[str] = set()
    if func is not None:
        for decorator in func.decorator_list:
            names |= _masking_names_in(decorator)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            names |= _masking_names_in(node.value)
    return names


def _masking_names_in(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        dotted: str | None = None
        if isinstance(child, ast.Call):
            dotted = _dotted_name(child.func)
        elif isinstance(child, ast.Attribute):
            dotted = _dotted_name(child)
        elif isinstance(child, ast.Name):
            dotted = child.id
        if dotted is not None and any(tok in dotted for tok in _MASKING_TOKENS):
            names.add(dotted)
    return names


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted attribute chain (e.g. ``pytest.mark.skip``) or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _masking_try_count(node: ast.AST) -> int:
    """Count ``try`` blocks whose body wraps an ``assert`` (masking the failure)."""
    count = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Try) and _try_wraps_assert(child):
            count += 1
    return count


def _try_wraps_assert(try_node: ast.Try) -> bool:
    for stmt in try_node.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Assert):
                return True
    return False


# ----------------------------------------------------------------------
# git-tracked copy / apply helpers (local, never borrowed from the Fixer module)
# ----------------------------------------------------------------------
def _copy_tracked_tree(repo_root: Path, dest: Path) -> None:
    """Copy the git-tracked working tree to ``dest``.

    Tracked files only, so the demo's 87MB ``.venv`` is never copied
    (``git ls-files`` excludes it for free). Falls back to a filtered
    ``copytree`` when ``repo_root`` is not a git repo.
    """
    dest.mkdir(parents=True, exist_ok=True)
    tracked = _git_tracked_files(repo_root)
    if tracked is None:
        shutil.copytree(
            repo_root,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*_IGNORED_DIRS),
        )
        return
    for rel in tracked:
        source = repo_root / rel
        # A tracked path can be absent (deleted but not staged) or a submodule
        # directory; skip rather than crash.
        if not source.is_file():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


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


def _git_apply(diff: str, root: Path) -> None:
    """Apply ``diff`` into ``root`` (a plain tree copy; ``git apply`` needs no repo)."""
    try:
        process = subprocess.run(
            ["git", "apply", "-"],
            cwd=root,
            input=diff.encode("utf-8"),
            capture_output=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerifierError(f"could not apply the fix diff with `git apply`: {exc}") from exc
    if process.returncode != 0:
        raise VerifierError(
            "the fix diff did not apply via `git apply`: "
            + process.stderr.decode("utf-8", errors="replace").strip()
        )


def _temporary_dir() -> "_TempDir":
    return _TempDir(tempfile.mkdtemp(prefix="flakehunter-verify-"))


class _TempDir:
    """A temporary directory that yields a ``Path`` and never fails cleanup.

    pytest may leave read-only caches or live handles behind on Windows, and a
    cleanup error must not mask the verdict we just computed.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)

    def __enter__(self) -> Path:
        return self._path

    def __exit__(self, *exc_info: object) -> None:
        shutil.rmtree(self._path, ignore_errors=True)
