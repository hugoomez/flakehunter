"""Hand-built contract objects shared by the orchestrator/cli/tui tests.

These let the fast tests fabricate FlakeVerdict/RootCause/FixProposal/
VerificationResult instances without running any subprocess or the Codex SDK.
"""

from __future__ import annotations

from flakehunter.contracts import (
    FixProposal,
    FlakeVerdict,
    RootCause,
    RunResult,
    VerificationResult,
)

_AUTO_FIXABLE = {"order", "shared_state", "randomness"}


def make_verdict(test_id: str, *, is_flaky: bool = True) -> FlakeVerdict:
    return FlakeVerdict(
        test_id=test_id,
        n_runs=30,
        n_failures=15 if is_flaky else 0,
        failure_rate=0.5 if is_flaky else 0.0,
        ci95_upper=0.68 if is_flaky else 0.11,
        is_flaky=is_flaky,
        sample_tracebacks=["AssertionError: boom"] if is_flaky else [],
    )


def make_cause(test_id: str, *, category: str = "randomness") -> RootCause:
    return RootCause(
        test_id=test_id,
        category=category,  # type: ignore[arg-type]
        confidence=0.9,
        evidence=["seeded run passes, unseeded fails"],
        ast_signals=["calls random.shuffle"],
        auto_fixable=category in _AUTO_FIXABLE,
    )


def make_proposal(test_id: str) -> FixProposal:
    return FixProposal(
        test_id=test_id,
        diff="--- a/x.py\n+++ b/x.py\n@@\n-old\n+new\n",
        rationale="seed the RNG before shuffling",
        files_touched=["x.py"],
        codex_session_id="sess-1",
    )


def make_result(test_id: str, *, verdict: str = "verified_fix") -> VerificationResult:
    ok = verdict == "verified_fix"
    return VerificationResult(
        test_id=test_id,
        contract_passed=ok,
        assertion_count_preserved=True,
        no_skip_introduced=True,
        failure_rate_before=0.5,
        failure_rate_after=0.0 if ok else 0.4,
        ci95_upper_after=0.05 if ok else 0.6,
        suite_still_green=True,
        verdict=verdict,  # type: ignore[arg-type]
    )


def make_run_result(test_id: str, *, seed: int, outcome: str = "passed") -> RunResult:
    return RunResult(
        test_id=test_id,
        outcome=outcome,  # type: ignore[arg-type]
        duration_s=0.012,
        error_repr=None if outcome == "passed" else "AssertionError: boom",
        seed_env={"PYTHONHASHSEED": str(seed), "randomly-seed": str(seed)},
        order_hash="deadbeef",
    )
