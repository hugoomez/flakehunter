"""Tests for the Codex-backed :class:`Fixer`.

The split mirrors ``test_classifier.py``: fast tests carry the *reasoning*,
end-to-end tests carry the *reality*, and neither question subsumes the other.

* **Fake-client tests** drive a `_FakeCodexFactory` against a throwaway git
  repo. They cover prompt construction, thread parameters, ``TurnStatus``
  handling, diff computation, and every refusal path in milliseconds, offline,
  with no tokens spent. The fake builds the **real** ``TurnResult`` /
  ``TurnStatus`` / ``ThreadItem`` types from the SDK and fakes only the
  transport, so it cannot drift from the API *shape* it stands in for.

  Shape fidelity is not behaviour fidelity, though, and this module learned
  that the expensive way. The fake originally returned
  ``TurnResult(status=TurnStatus.failed)`` — a plausible reading of the enum,
  and the tests passed. The real SDK never returns that: it *raises*
  ``RuntimeError`` (``openai_codex/_run.py:59``). The first real E2E run is
  what exposed it, and `test_failed_turn_raised_by_the_sdk_becomes_a_codex_turn_error`
  now pins the real behaviour.
* **One real Codex test** (``@pytest.mark.codex``, opt-in via
  ``FLAKEHUNTER_CODEX_E2E=1``) proves the integration end to end against the
  demo repo's unseeded-randomness case.

Why the real test does not assert the fix works
-----------------------------------------------
It asserts the invariants *any* valid fix must satisfy — applies cleanly,
assertion count preserved, no skip/xfail, files in scope — and deliberately
not that the failure rate drops. Whether the fix actually works is
``Verifier``'s question, answered statistically. Asserting it here would make
FlakeHunter's own suite depend on Codex's nondeterminism, which is the mistake
``docs/ARCHITECTURE.md`` §6 records the project catching once already.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from openai_codex import ApprovalMode, Sandbox, TransportClosedError, TurnResult
from openai_codex.generated.v2_all import (
    FileChangeThreadItem,
    FileUpdateChange,
    PatchApplyStatus,
    PatchChangeKind,
    UpdatePatchChangeKind,
)
from openai_codex.types import ThreadItem, TurnError, TurnStatus

from flakehunter.contracts import FlakeVerdict, RootCause
from flakehunter.fixer import (
    _CATEGORY_INSTRUCTIONS,
    _CONSTRAINTS,
    CodexTurnError,
    EmptyFixError,
    Fixer,
    FixerError,
    NotAutoFixableError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# Mirrors demo/tests/test_pricing_rounding.py: 19/20 candidates round to 0.10,
# one rounds to 0.11 -> a 5% failure rate driven by an unseeded RNG.
_TARGET_SOURCE = '''import random


def test_round_trip() -> None:
    rounding_candidates = [0.104] * 19 + [0.106]

    adjustment = random.choice(rounding_candidates)

    assert round(adjustment, 2) == 0.10
'''
_TARGET_REL = "demo/tests/test_target.py"
_TARGET_TEST_ID = f"{_TARGET_REL}::test_round_trip"
_SEED_LINE = "    random.seed(0)\n"


# ======================================================================
# Fixtures and fakes
# ======================================================================
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo holding the target test.

    Real `git init` + `git add` so `Fixer._copy_tree` exercises its primary
    `git ls-files` path rather than the copytree fallback.
    """
    root = tmp_path / "repo"
    (root / "demo" / "tests").mkdir(parents=True)
    (root / "demo" / "tests" / "test_target.py").write_text(
        _TARGET_SOURCE, encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _turn_result(
    *,
    status: TurnStatus = TurnStatus.completed,
    final_response: str | None = "Seeded the RNG so the choice is deterministic.",
    error: TurnError | None = None,
    items: list[ThreadItem] | None = None,
) -> TurnResult:
    """Build a real ``TurnResult`` — only the transport is faked."""
    return TurnResult(
        id="turn_fake_001",
        status=status,
        error=error,
        started_at=0,
        completed_at=1,
        duration_ms=1,
        final_response=final_response,
        items=list(items or []),
        usage=None,
    )


def _file_change_item(
    path: str, *, status: PatchApplyStatus = PatchApplyStatus.completed
) -> ThreadItem:
    change = FileUpdateChange(
        diff="--- a/x\n+++ b/x\n",
        kind=PatchChangeKind(UpdatePatchChangeKind(type="update")),
        path=path,
    )
    return ThreadItem(
        FileChangeThreadItem(
            id="item_fake_001",
            type="fileChange",
            status=status,
            changes=[change],
        )
    )


class _FakeThread:
    def __init__(self, thread_id: str, cwd: str, on_run) -> None:
        self.id = thread_id
        self.cwd = Path(cwd)
        self._on_run = on_run
        self.runs: list[dict[str, object]] = []

    def run(self, input, **kwargs) -> TurnResult:  # noqa: A002 - SDK's parameter name
        self.runs.append({"input": input, "kwargs": kwargs})
        return self._on_run(input, self.cwd)


class _FakeCodexFactory:
    """Stands in for the ``Codex`` class and for the client it constructs.

    Records how it was called so tests can assert the *protocol* — that the
    sandbox, approval mode and cwd are what they must be — not just the result.
    ``call_count`` is what proves the refusal paths never reach Codex.
    """

    def __init__(self, on_run, thread_id: str = "thr_fake_001") -> None:
        self._on_run = on_run
        self.thread_id = thread_id
        self.call_count = 0
        self.closed = False
        self.thread_start_kwargs: dict[str, object] | None = None
        self.thread: _FakeThread | None = None

    def __call__(self) -> "_FakeCodexFactory":
        self.call_count += 1
        return self

    def __enter__(self) -> "_FakeCodexFactory":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

    def thread_start(self, **kwargs) -> _FakeThread:
        self.thread_start_kwargs = kwargs
        self.thread = _FakeThread(self.thread_id, str(kwargs["cwd"]), self._on_run)
        return self.thread

    @property
    def prompt(self) -> str:
        assert self.thread is not None and self.thread.runs
        return str(self.thread.runs[0]["input"])


def _seeds_the_rng(**turn_kwargs):
    """An `on_run` that applies a legitimate seed fix inside the sandbox."""

    def on_run(prompt: str, cwd: Path) -> TurnResult:
        path = cwd / _TARGET_REL
        text = path.read_text(encoding="utf-8")
        path.write_text(
            text.replace(
                "    rounding_candidates", _SEED_LINE + "    rounding_candidates", 1
            ),
            encoding="utf-8",
        )
        return _turn_result(**turn_kwargs)

    return on_run


def _changes_nothing(**turn_kwargs):
    def on_run(prompt: str, cwd: Path) -> TurnResult:
        return _turn_result(**turn_kwargs)

    return on_run


def _verdict(test_id: str = _TARGET_TEST_ID, **overrides) -> FlakeVerdict:
    fields: dict[str, object] = {
        "n_runs": 30,
        "n_failures": 2,
        "failure_rate": 2 / 30,
        "ci95_upper": 0.2114,
        "is_flaky": True,
        "sample_tracebacks": ["AssertionError: assert 0.11 == 0.1"],
    }
    fields.update(overrides)
    return FlakeVerdict(test_id=test_id, **fields)  # type: ignore[arg-type]


def _cause(
    category: str = "randomness",
    *,
    test_id: str = _TARGET_TEST_ID,
    auto_fixable: bool | None = None,
    evidence: list[str] | None = None,
    ast_signals: list[str] | None = None,
) -> RootCause:
    if auto_fixable is None:
        auto_fixable = category in {"order", "shared_state", "randomness"}
    return RootCause(
        test_id=test_id,
        category=category,  # type: ignore[arg-type]
        confidence=0.9,
        evidence=evidence
        if evidence is not None
        else ["Fisher exact p=0.0210: 6/30 failures with a per-run seed"],
        ast_signals=ast_signals
        if ast_signals is not None
        else ["calls random.choice without a fixed seed (unseeded randomness)"],
        auto_fixable=auto_fixable,
    )


def _fixer(repo: Path, factory: _FakeCodexFactory, model: str | None = None) -> Fixer:
    return Fixer(repo_root=repo, model=model, codex_factory=factory)


def _apply(diff: str, root: Path) -> None:
    process = subprocess.run(
        ["git", "apply", "-"],
        cwd=root,
        input=diff.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode()


def _assertion_count(source: str) -> int:
    return sum(
        isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source))
    )


def _skip_or_xfail_decorators(source: str) -> list[str]:
    """Decorator names in ``source`` that would mask a failure.

    Matched structurally rather than by substring: a bare ``"skip" in source``
    also fires on the word appearing in a comment or a docstring.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            parts: list[str] = []
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            dotted = ".".join(reversed(parts))
            if any(token in dotted for token in ("skip", "xfail")):
                found.append(dotted)
    return found


# ======================================================================
# Unit tests: prompt construction
# ======================================================================
@pytest.mark.parametrize("category", ["randomness", "order", "shared_state"])
def test_prompt_carries_the_cause_and_only_its_own_category(
    repo: Path, category: str
) -> None:
    evidence = ["Fisher exact p=0.0004: 22/30 in-suite vs 0/30 isolated"]
    ast_signals = ["calls random.choice without a fixed seed (unseeded randomness)"]
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory).propose_fix(
        _cause(category, evidence=evidence, ast_signals=ast_signals), _verdict()
    )

    prompt = factory.prompt
    assert _TARGET_TEST_ID in prompt
    assert _TARGET_REL in prompt
    assert evidence[0] in prompt
    assert ast_signals[0] in prompt
    assert "AssertionError: assert 0.11 == 0.1" in prompt
    assert _CONSTRAINTS in prompt
    assert _CATEGORY_INSTRUCTIONS[category] in prompt
    for other, instruction in _CATEGORY_INSTRUCTIONS.items():
        if other != category:
            assert instruction not in prompt


def test_prompt_reports_the_verdict_statistics(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory).propose_fix(
        _cause(), _verdict(n_runs=30, n_failures=2, failure_rate=2 / 30)
    )

    prompt = factory.prompt
    assert "failed 2/30 times" in prompt
    assert "failure rate 6.7%" in prompt
    assert "Wilson 95% CI upper bound 21.1%" in prompt


def test_prompt_omits_empty_sections(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory).propose_fix(
        _cause(ast_signals=[]), _verdict(sample_tracebacks=[])
    )

    prompt = factory.prompt
    assert "Static signals:" not in prompt
    assert "Sample failures:" not in prompt
    assert "Evidence:" in prompt


# ======================================================================
# Unit tests: thread parameters
# ======================================================================
def test_thread_start_sandboxes_codex_in_the_temp_copy(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory, model="gpt-5-codex").propose_fix(_cause(), _verdict())

    kwargs = factory.thread_start_kwargs
    assert kwargs is not None
    assert kwargs["sandbox"] is Sandbox.workspace_write
    assert kwargs["approval_mode"] is ApprovalMode.deny_all
    assert kwargs["model"] == "gpt-5-codex"
    cwd = Path(str(kwargs["cwd"]))
    assert cwd.resolve() != repo.resolve()
    assert repo.resolve() not in cwd.resolve().parents
    assert factory.closed is True


def test_repo_root_is_never_mutated(repo: Path) -> None:
    original = (repo / _TARGET_REL).read_text(encoding="utf-8")
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert (repo / _TARGET_REL).read_text(encoding="utf-8") == original


def test_sandbox_is_cleaned_up(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    _fixer(repo, factory).propose_fix(_cause(), _verdict())

    cwd = Path(str(factory.thread_start_kwargs["cwd"]))  # type: ignore[index]
    assert not cwd.exists()


# ======================================================================
# Unit tests: TurnStatus handling
# ======================================================================
@pytest.mark.parametrize("status", [TurnStatus.interrupted, TurnStatus.in_progress])
def test_non_completed_returned_status_raises_rather_than_proposing(
    repo: Path, status: TurnStatus
) -> None:
    """The statuses `Thread.run` actually returns must not reach the diff."""
    factory = _FakeCodexFactory(
        _seeds_the_rng(
            status=status,
            error=TurnError(
                message="model stream disconnected",
                additional_details="retry limit exceeded",
            ),
        )
    )

    with pytest.raises(CodexTurnError) as excinfo:
        _fixer(repo, factory).propose_fix(_cause(), _verdict())

    message = str(excinfo.value)
    assert status.value in message
    assert "model stream disconnected" in message
    assert "retry limit exceeded" in message
    assert "thr_fake_001" in message


def test_non_completed_returned_status_without_error_detail_still_raises(
    repo: Path,
) -> None:
    factory = _FakeCodexFactory(
        _seeds_the_rng(status=TurnStatus.interrupted, error=None)
    )

    with pytest.raises(CodexTurnError):
        _fixer(repo, factory).propose_fix(_cause(), _verdict())


def test_failed_turn_raised_by_the_sdk_becomes_a_codex_turn_error(repo: Path) -> None:
    """A `failed` turn arrives as a RuntimeError, not as a TurnResult.

    The SDK's `_raise_for_failed_turn` raises rather than returning
    `TurnStatus.failed`, so this — not a returned status — is how a real
    failure (a usage limit, an upstream error) reaches the Fixer. Found by
    running the real E2E test against a rate-limited account.
    """

    def raises_usage_limit(prompt: str, cwd: Path) -> TurnResult:
        raise RuntimeError("You've hit your usage limit. Upgrade to Plus to continue")

    factory = _FakeCodexFactory(raises_usage_limit)

    with pytest.raises(CodexTurnError) as excinfo:
        _fixer(repo, factory).propose_fix(_cause(), _verdict())

    message = str(excinfo.value)
    assert "usage limit" in message
    assert "thr_fake_001" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_sdk_transport_error_becomes_a_codex_turn_error(repo: Path) -> None:
    def raises_transport(prompt: str, cwd: Path) -> TurnResult:
        raise TransportClosedError("connection closed")

    factory = _FakeCodexFactory(raises_transport)

    with pytest.raises(CodexTurnError):
        _fixer(repo, factory).propose_fix(_cause(), _verdict())


def test_completed_turn_that_changed_nothing_raises_empty_fix(repo: Path) -> None:
    factory = _FakeCodexFactory(_changes_nothing())

    with pytest.raises(EmptyFixError):
        _fixer(repo, factory).propose_fix(_cause(), _verdict())


# ======================================================================
# Unit tests: diff computation
# ======================================================================
def test_diff_is_a_unified_diff_that_git_apply_accepts(repo: Path, tmp_path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert f"--- a/{_TARGET_REL}" in proposal.diff
    assert f"+++ b/{_TARGET_REL}" in proposal.diff
    assert "+    random.seed(0)" in proposal.diff
    assert proposal.files_touched == [_TARGET_REL]

    # Apply it for real against a pristine copy and check the result.
    target = tmp_path / "applied"
    shutil.copytree(repo, target)
    _apply(proposal.diff, target)
    fixed = (target / _TARGET_REL).read_text(encoding="utf-8")
    assert "random.seed(0)" in fixed
    assert "assert round(adjustment, 2) == 0.10" in fixed
    assert _assertion_count(fixed) == _assertion_count(_TARGET_SOURCE)


def test_new_file_is_diffed_against_dev_null_and_applies(repo: Path, tmp_path) -> None:
    def adds_conftest(prompt: str, cwd: Path) -> TurnResult:
        (cwd / "demo" / "tests" / "conftest.py").write_text(
            "import pytest\n\n\n@pytest.fixture(autouse=True)\ndef _reset() -> None:\n"
            "    pass\n",
            encoding="utf-8",
        )
        return _turn_result()

    factory = _FakeCodexFactory(adds_conftest)

    proposal = _fixer(repo, factory).propose_fix(_cause("shared_state"), _verdict())

    assert "--- /dev/null" in proposal.diff
    assert "+++ b/demo/tests/conftest.py" in proposal.diff
    assert proposal.files_touched == ["demo/tests/conftest.py"]

    target = tmp_path / "applied"
    shutil.copytree(repo, target)
    _apply(proposal.diff, target)
    assert (target / "demo" / "tests" / "conftest.py").is_file()


def test_multiple_files_are_diffed_in_sorted_path_order(repo: Path) -> None:
    def edits_two(prompt: str, cwd: Path) -> TurnResult:
        (cwd / "demo" / "tests" / "conftest.py").write_text("# reset\n", encoding="utf-8")
        path = cwd / _TARGET_REL
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "    rounding_candidates", _SEED_LINE + "    rounding_candidates", 1
            ),
            encoding="utf-8",
        )
        return _turn_result()

    factory = _FakeCodexFactory(edits_two)

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert proposal.files_touched == ["demo/tests/conftest.py", _TARGET_REL]
    assert proposal.diff.index("b/demo/tests/conftest.py") < proposal.diff.index(
        f"b/{_TARGET_REL}"
    )


# ======================================================================
# Unit tests: proposal fields and cross-checks
# ======================================================================
def test_proposal_carries_session_id_and_rationale(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng(), thread_id="thr_abc_123")

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert proposal.test_id == _TARGET_TEST_ID
    assert proposal.codex_session_id == "thr_abc_123"
    assert proposal.rationale.startswith("Seeded the RNG")


def test_rationale_falls_back_when_final_response_is_none(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng(final_response=None))

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert _TARGET_TEST_ID in proposal.rationale
    assert "randomness" in proposal.rationale


@pytest.mark.parametrize(
    "status", [PatchApplyStatus.failed, PatchApplyStatus.declined]
)
def test_unapplied_patch_is_surfaced_in_the_rationale(
    repo: Path, status: PatchApplyStatus
) -> None:
    factory = _FakeCodexFactory(
        _seeds_the_rng(items=[_file_change_item("demo/src/shopcart/cart.py", status=status)])
    )

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert "Fixer notes:" in proposal.rationale
    assert status.value in proposal.rationale
    assert "demo/src/shopcart/cart.py" in proposal.rationale
    # The note is a warning, not a veto: the seed fix still landed.
    assert proposal.files_touched == [_TARGET_REL]


def test_path_codex_reports_but_snapshot_missed_is_surfaced(repo: Path) -> None:
    # A non-.py file is outside the snapshot's reach by construction.
    factory = _FakeCodexFactory(_seeds_the_rng(items=[_file_change_item("pytest.ini")]))

    proposal = _fixer(repo, factory).propose_fix(_cause(), _verdict())

    assert "pytest.ini" in proposal.rationale
    assert "snapshot did not capture" in proposal.rationale


def test_edit_beyond_the_target_test_file_is_surfaced(repo: Path) -> None:
    def edits_conftest_only(prompt: str, cwd: Path) -> TurnResult:
        (cwd / "demo" / "tests" / "conftest.py").write_text("# reset\n", encoding="utf-8")
        return _turn_result()

    factory = _FakeCodexFactory(edits_conftest_only)

    proposal = _fixer(repo, factory).propose_fix(_cause("shared_state"), _verdict())

    assert "reaches beyond the target test file" in proposal.rationale


# ======================================================================
# Unit tests: refusal paths
# ======================================================================
@pytest.mark.parametrize("category", ["timing", "external"])
def test_detect_only_categories_are_refused_without_calling_codex(
    repo: Path, category: str
) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    with pytest.raises(NotAutoFixableError) as excinfo:
        _fixer(repo, factory).propose_fix(_cause(category), _verdict())

    assert category in str(excinfo.value)
    assert factory.call_count == 0


def test_inconsistent_root_cause_is_refused(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())

    with pytest.raises(NotAutoFixableError) as excinfo:
        _fixer(repo, factory).propose_fix(
            _cause("randomness", auto_fixable=False), _verdict()
        )

    assert "refusing to guess" in str(excinfo.value)
    assert factory.call_count == 0


def test_missing_source_file_is_refused(repo: Path) -> None:
    factory = _FakeCodexFactory(_seeds_the_rng())
    missing = "demo/tests/test_does_not_exist.py::test_x"

    with pytest.raises(FixerError) as excinfo:
        _fixer(repo, factory).propose_fix(_cause(test_id=missing), _verdict(missing))

    assert "not found" in str(excinfo.value)
    assert factory.call_count == 0


# ======================================================================
# End-to-end: the real Codex API
# ======================================================================
_E2E_TEST_ID = "demo/tests/test_pricing_rounding.py::test_round_trip"


@pytest.mark.codex
@pytest.mark.skipif(
    not os.getenv("FLAKEHUNTER_CODEX_E2E"),
    reason="set FLAKEHUNTER_CODEX_E2E=1 to run the real Codex integration test",
)
def test_real_codex_proposes_applicable_fix_for_unseeded_randomness(tmp_path) -> None:
    """Prove the SDK integration end to end against a real demo flake.

    Targets ``test_round_trip`` rather than ``test_shuffle_preserves_first``
    because ARCHITECTURE.md §4 records that seed 42 makes the deck test fail
    30/30 — no seed fixes it, so Codex could only "pass" it by weakening the
    assertion. This case has a real seed fix that keeps its assertion intact.

    The RootCause is hand-built to match what Classifier really produces for
    this case (AST-backed randomness, per ARCHITECTURE.md §4); running the real
    classifier would cost ~90 subprocesses for no added signal here.
    """
    source = (REPO_ROOT / "demo" / "tests" / "test_pricing_rounding.py").read_text(
        encoding="utf-8"
    )
    cause = RootCause(
        test_id=_E2E_TEST_ID,
        category="randomness",
        confidence=0.5,
        evidence=[
            "No experiment was significant; classified from AST randomness "
            "signal (unseeded RNG usage)"
        ],
        ast_signals=["calls random.choice without a fixed seed (unseeded randomness)"],
        auto_fixable=True,
    )
    verdict = FlakeVerdict(
        test_id=_E2E_TEST_ID,
        n_runs=30,
        n_failures=2,
        failure_rate=2 / 30,
        ci95_upper=0.2114,
        is_flaky=True,
        sample_tracebacks=["AssertionError: assert 0.11 == 0.1"],
    )

    proposal = Fixer(repo_root=REPO_ROOT).propose_fix(cause, verdict)

    print("\n--- Codex session:", proposal.codex_session_id)
    print("--- rationale:\n" + proposal.rationale)
    print("--- diff:\n" + proposal.diff)

    assert proposal.codex_session_id
    assert proposal.diff.strip()
    assert proposal.files_touched == ["demo/tests/test_pricing_rounding.py"]

    # The diff applies cleanly to a pristine copy...
    target = tmp_path / "applied"
    target.mkdir()
    for rel in subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode().split("\0"):
        if not rel:
            continue
        source_path = REPO_ROOT / rel
        if not source_path.is_file():
            continue
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target / rel)
    _apply(proposal.diff, target)

    # ...and the fix does not weaken the test. Whether it *works* is
    # Verifier's question, decided statistically — not asserted here.
    fixed = (target / "demo" / "tests" / "test_pricing_rounding.py").read_text(
        encoding="utf-8"
    )
    assert _assertion_count(fixed) >= _assertion_count(source)
    assert _skip_or_xfail_decorators(fixed) == []

    # The real repo is untouched.
    assert (
        REPO_ROOT / "demo" / "tests" / "test_pricing_rounding.py"
    ).read_text(encoding="utf-8") == source
