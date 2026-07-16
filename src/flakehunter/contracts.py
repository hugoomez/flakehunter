"""Immutable contracts shared by FlakeHunter components."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RunResult:
    test_id: str
    outcome: Literal["passed", "failed", "error", "skipped"]
    duration_s: float
    error_repr: str | None
    seed_env: dict[str, str]
    order_hash: str


@dataclass(frozen=True)
class FlakeVerdict:
    test_id: str
    n_runs: int
    n_failures: int
    failure_rate: float
    ci95_upper: float
    is_flaky: bool
    sample_tracebacks: list[str]


@dataclass(frozen=True)
class RootCause:
    test_id: str
    category: Literal["order", "shared_state", "randomness", "timing", "external"]
    confidence: float
    evidence: list[str]
    ast_signals: list[str]
    auto_fixable: bool


@dataclass(frozen=True)
class FixProposal:
    test_id: str
    diff: str
    rationale: str
    files_touched: list[str]
    codex_session_id: str


@dataclass(frozen=True)
class VerificationResult:
    test_id: str
    contract_passed: bool
    assertion_count_preserved: bool
    no_skip_introduced: bool
    failure_rate_before: float
    failure_rate_after: float
    ci95_upper_after: float
    suite_still_green: bool
    verdict: Literal[
        "verified_fix",
        "rejected_weakens_test",
        "rejected_still_flaky",
        "rejected_breaks_suite",
    ]
