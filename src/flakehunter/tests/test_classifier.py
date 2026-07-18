"""Classifier checks: fast unit tests plus end-to-end runs on the demo repo.

The module has two halves:

* **Unit tests** drive :class:`Classifier` through a scripted ``_StubRunner``
  and call :func:`scan_ast` directly. No subprocesses, so they are fast and
  deterministic, and they can stage failure counts the demo repo cannot
  produce (see the note on the randomness experiment below).
* **End-to-end tests** run the real :class:`SandboxRunner` against
  ``demo/tests``. Unlike the other test modules these run several *batches* of
  repeated pytest executions per case — isolated (per-run seed and fixed seed)
  plus full-suite runs (unforked, and a forked probe) — so at
  ``n_experiment_runs=30`` a single case launches ~90 pytest subprocesses and
  the whole module takes on the order of several minutes. That is expected; 30
  is chosen to keep it "reasonably fast" while still giving the Fisher exact
  comparisons enough samples to be decisive.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from flakehunter.classifier import Classifier, scan_ast
from flakehunter.contracts import FlakeVerdict, RunResult
from flakehunter.runner import SandboxRunner


REPO_ROOT = Path(__file__).resolve().parents[3]
GROUND_TRUTH_PATH = REPO_ROOT / "demo" / "GROUND_TRUTH.json"

# The full demo/tests/ suite, used as the shared context for the order and
# shared-state (in-suite) experiments.
SUITE_TEST_IDS = [
    "demo/tests/test_catalog.py::test_load_populates_cache",
    "demo/tests/test_catalog.py::test_read_from_cache",
    "demo/tests/test_cart_discounts.py::test_add_discount_to_shared_cart",
    "demo/tests/test_cart_discounts.py::test_discount_count",
    "demo/tests/test_deck.py::test_shuffle_preserves_first",
    "demo/tests/test_pricing_rounding.py::test_round_trip",
    "demo/tests/test_async_worker.py::test_worker_completes",
    "demo/tests/test_broken_feature.py::test_feature_x",
    "demo/tests/test_stable_total.py::test_total_is_correct",
]

# Each auto-detectable demo case and the category the classifier must assign,
# per demo/GROUND_TRUTH.json.
CLASSIFICATION_CASES = [
    ("demo/tests/test_catalog.py::test_read_from_cache", "order"),
    ("demo/tests/test_cart_discounts.py::test_discount_count", "shared_state"),
    ("demo/tests/test_deck.py::test_shuffle_preserves_first", "randomness"),
    ("demo/tests/test_pricing_rounding.py::test_round_trip", "randomness"),
    ("demo/tests/test_async_worker.py::test_worker_completes", "timing"),
]

AUTO_FIXABLE_CATEGORIES = {"order", "shared_state", "randomness"}


def _flake_verdict(test_id: str) -> FlakeVerdict:
    """A minimal FlakeVerdict; classify() only consumes ``test_id``."""
    return FlakeVerdict(
        test_id=test_id,
        n_runs=30,
        n_failures=15,
        failure_rate=0.5,
        ci95_upper=0.7,
        is_flaky=True,
        sample_tracebacks=[],
    )


# ======================================================================
# Unit tests: decision logic via a scripted runner
# ======================================================================

_STUB_TARGET = "pkg/test_target.py::test_target"
_STUB_SUITE = [_STUB_TARGET, "pkg/test_other.py::test_other"]


class _StubRunner:
    """A SandboxRunner that scripts failure counts per experimental condition.

    Each ``run_once`` call is bucketed into one of the four conditions the
    classifier samples, and the first ``k`` calls of a bucket report the target
    as failed. Every call is recorded so tests can assert *how* the classifier
    sampled, not just what it concluded.
    """

    def __init__(
        self,
        *,
        k_iso: int = 0,
        k_seeded: int = 0,
        k_suite: int = 0,
        k_forked: int = 0,
        fork_supported: bool = False,
        target: str = _STUB_TARGET,
        suite: list[str] | None = None,
    ) -> None:
        self.target = target
        self.suite = list(suite if suite is not None else _STUB_SUITE)
        self.fork_supported = fork_supported
        self._budget = {
            "iso": k_iso,
            "seeded": k_seeded,
            "suite": k_suite,
            "forked": k_forked,
        }
        self.calls: list[dict[str, object]] = []

    def _condition(self, test_ids: list[str], seed: int | None, forked: bool) -> str:
        if list(test_ids) == [self.target]:
            return "seeded" if seed is not None else "iso"
        return "forked" if forked else "suite"

    def run_once(
        self,
        test_ids: list[str],
        *,
        seed: int | None,
        forked: bool,
        randomize_order: bool,
    ) -> list[RunResult]:
        condition = self._condition(test_ids, seed, forked)
        self.calls.append(
            {
                "condition": condition,
                "seed": seed,
                "forked": forked,
                "randomize_order": randomize_order,
                "test_ids": list(test_ids),
            }
        )
        if forked and not self.fork_supported:
            # pytest-forked on a platform without os.fork: no real outcomes.
            return []

        target_failed = self._budget[condition] > 0
        if target_failed:
            self._budget[condition] -= 1
        return [
            RunResult(
                test_id=test_id,
                outcome=(
                    "failed" if test_id == self.target and target_failed else "passed"
                ),
                duration_s=0.01,
                error_repr=None,
                seed_env={},
                order_hash="stub",
            )
            for test_id in test_ids
        ]

    def run_isolated(self, test_id: str, *, seed: int | None) -> RunResult:
        return self.run_once(
            [test_id], seed=seed, forked=False, randomize_order=False
        )[0]

    def condition_calls(self, condition: str) -> list[dict[str, object]]:
        return [call for call in self.calls if call["condition"] == condition]


def _classify_with(runner: _StubRunner, *, repo_root: Path, n: int = 30):
    classifier = Classifier(
        runner=runner,
        suite_test_ids=runner.suite,
        n_experiment_runs=n,
        repo_root=repo_root,
    )
    return classifier.classify(_flake_verdict(runner.target))


def _write_target(tmp_path: Path, body: str) -> Path:
    """Materialise pkg/test_target.py so the AST scan has a source to read."""
    package = tmp_path / "pkg"
    package.mkdir(exist_ok=True)
    path = package / "test_target.py"
    path.write_text(body, encoding="utf-8")
    return path


_TIMING_SOURCE = """\
import time


def test_target() -> None:
    time.sleep(0.05)
    assert True
"""

_NO_SIGNAL_SOURCE = """\
def test_target() -> None:
    assert True
"""


def test_randomness_experiment_samples_fixed_seed_against_per_run_seed(
    tmp_path: Path,
) -> None:
    """BUG 1: the two randomness conditions must differ only in the seed.

    Condition A pins seed=42; condition B passes seed=None so pytest-randomly
    draws a fresh seed per run. Both run the target alone.
    """
    runner = _StubRunner(k_iso=20, k_seeded=0)

    _classify_with(runner, repo_root=tmp_path)

    seeded = runner.condition_calls("seeded")
    unseeded = runner.condition_calls("iso")
    assert len(seeded) == 30
    assert len(unseeded) == 30
    assert all(call["seed"] == 42 for call in seeded)
    assert all(call["randomize_order"] is False for call in seeded)
    assert all(call["seed"] is None for call in unseeded)
    assert all(call["randomize_order"] is True for call in unseeded)
    # Both conditions run the target in isolation, holding context constant.
    assert all(call["test_ids"] == [_STUB_TARGET] for call in seeded + unseeded)


def test_randomness_experiment_fires_on_experimental_evidence_alone(
    tmp_path: Path,
) -> None:
    """A fixed seed removing the failures yields randomness with no AST signal."""
    runner = _StubRunner(k_iso=20, k_seeded=0, k_suite=20)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.ast_signals == [], "no source on disk; AST must not contribute"
    assert cause.category == "randomness"
    assert cause.auto_fixable is True
    assert cause.confidence > 0.9
    assert any(
        "Fixing the seed reduced failures" in item for item in cause.evidence
    ), cause.evidence


def test_randomness_experiment_ignores_the_wrong_direction(tmp_path: Path) -> None:
    """A fixed seed that *adds* failures is not evidence of unseeded randomness."""
    runner = _StubRunner(k_iso=0, k_seeded=20, k_suite=0)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category != "randomness"


def test_pollution_direction_alone_yields_shared_state(tmp_path: Path) -> None:
    """Baseline for the veto: no timing signal, so the direction stands."""
    _write_target(tmp_path, _NO_SIGNAL_SOURCE)
    runner = _StubRunner(k_iso=0, k_suite=25)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category == "shared_state"
    assert cause.auto_fixable is True


def test_timing_signal_vetoes_direction_only_shared_state(tmp_path: Path) -> None:
    """BUG 2: a load-sensitive timing test must not be auto-fixed as shared state."""
    _write_target(tmp_path, _TIMING_SOURCE)
    runner = _StubRunner(k_iso=0, k_suite=25)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category == "timing"
    assert cause.auto_fixable is False
    assert any("Downgraded shared_state -> timing" in e for e in cause.evidence), (
        cause.evidence
    )


def test_timing_signal_vetoes_direction_only_order(tmp_path: Path) -> None:
    """An order verdict rests on direction alone, so the veto applies to it too."""
    _write_target(tmp_path, _TIMING_SOURCE)
    runner = _StubRunner(k_iso=25, k_suite=0)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category == "timing"
    assert cause.auto_fixable is False
    assert any("Downgraded order -> timing" in e for e in cause.evidence), (
        cause.evidence
    )


def test_corroborated_shared_state_survives_the_timing_veto(tmp_path: Path) -> None:
    """Forked experiment + in-suite direction agree: two signals beat the veto."""
    _write_target(tmp_path, _TIMING_SOURCE)
    runner = _StubRunner(k_iso=0, k_suite=25, k_forked=0, fork_supported=True)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category == "shared_state"
    assert cause.auto_fixable is True
    assert not any("Downgraded" in e for e in cause.evidence), cause.evidence


def test_randomness_verdict_is_not_vetoed_by_a_timing_signal(tmp_path: Path) -> None:
    """The veto guards direction-only verdicts, not the seed experiment."""
    _write_target(tmp_path, _TIMING_SOURCE)
    runner = _StubRunner(k_iso=20, k_seeded=0, k_suite=20)

    cause = _classify_with(runner, repo_root=tmp_path)

    assert cause.category == "randomness"
    assert cause.auto_fixable is True


# ======================================================================
# Unit tests: AST scanning
# ======================================================================


def _kinds(source: str, test_name: str | None = "test_target") -> set[str]:
    return {signal.kind for signal in scan_ast(ast.parse(source), test_name)}


def test_ast_resolves_module_alias() -> None:
    """BUG 3a: `import random as rnd` must still read as randomness."""
    source = """\
import random as rnd


def test_target() -> None:
    assert rnd.choice([1, 2]) == 1
"""
    assert "randomness" in _kinds(source)


def test_ast_resolves_from_import() -> None:
    """BUG 3a: `from time import sleep` has no `time.` prefix to match."""
    source = """\
from time import sleep


def test_target() -> None:
    sleep(0.01)
"""
    assert "timing" in _kinds(source)


def test_ast_resolves_aliased_from_import() -> None:
    source = """\
from time import sleep as nap


def test_target() -> None:
    nap(0.01)
"""
    assert "timing" in _kinds(source)


def test_ast_resolves_numpy_random_alias() -> None:
    source = """\
import numpy as np


def test_target() -> None:
    assert np.random.rand() < 0.5
"""
    assert "randomness" in _kinds(source)


def test_ast_treats_seeded_random_constructor_as_seeded() -> None:
    """BUG 3b: `random.Random(42)` is reproducible, not unseeded randomness."""
    source = """\
import random


def test_target() -> None:
    rng = random.Random(42)
    assert rng.random() < 1.0
"""
    kinds = _kinds(source)
    assert "randomness" not in kinds
    assert "seeded" in kinds


def test_ast_treats_unseeded_random_constructor_as_random() -> None:
    source = """\
import random


def test_target() -> None:
    rng = random.Random()
    assert rng.random() < 1.0
"""
    assert "randomness" in _kinds(source)


def test_ast_detects_asyncio_sleep() -> None:
    """BUG 3c: asyncio.sleep is a timing dependency."""
    source = """\
import asyncio


async def test_target() -> None:
    await asyncio.sleep(0.01)
"""
    assert "timing" in _kinds(source)


def test_ast_scan_is_scoped_to_the_target_function() -> None:
    """BUG 3d: one test's seed() must not suppress another test's signal."""
    source = """\
import random
import time


def test_seeded() -> None:
    random.seed(1234)
    assert random.random() < 1.0


def test_unseeded() -> None:
    assert random.random() < 1.0


def test_timed() -> None:
    time.sleep(0.01)
"""
    # The seeded test's own signal is "seeded", never "randomness"...
    seeded_kinds = _kinds(source, "test_seeded")
    assert "randomness" not in seeded_kinds
    assert "seeded" in seeded_kinds

    # ...and it must not suppress the unseeded test in the same file.
    assert _kinds(source, "test_unseeded") == {"randomness"}

    # Signals must not leak the other way either.
    assert _kinds(source, "test_timed") == {"timing"}


def test_ast_scan_includes_module_level_code() -> None:
    """Module constants run before every test in the file, so they stay in scope."""
    source = """\
import random

TOKEN = random.random()


def test_target() -> None:
    assert TOKEN < 1.0
"""
    assert "randomness" in _kinds(source)


def test_ast_scan_includes_requested_fixtures() -> None:
    """A fixture's body runs before the test that requests it."""
    source = """\
import time

import pytest


@pytest.fixture
def slow_setup():
    time.sleep(0.01)
    return 1


@pytest.fixture
def unused_fixture():
    import random

    return random.random()


def test_target(slow_setup) -> None:
    assert slow_setup == 1
"""
    kinds = _kinds(source)
    assert "timing" in kinds
    assert "randomness" not in kinds, "an unrequested fixture must not contribute"


def test_ast_scan_falls_back_to_the_module_for_an_unknown_test() -> None:
    source = """\
import time


def test_target() -> None:
    time.sleep(0.01)
"""
    assert "timing" in _kinds(source, "test_generated_by_a_plugin")


# ======================================================================
# End-to-end tests against the demo repo
# ======================================================================


def _make_classifier() -> Classifier:
    runner = SandboxRunner(cwd=REPO_ROOT, timeout_s=30.0)
    return Classifier(
        runner=runner,
        suite_test_ids=SUITE_TEST_IDS,
        n_experiment_runs=30,
        repo_root=REPO_ROOT,
    )


def test_ground_truth_matches_expected_cases() -> None:
    """Guard: the categories asserted here agree with GROUND_TRUTH.json."""
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    for test_id, expected_category in CLASSIFICATION_CASES:
        key = test_id.removeprefix("demo/")
        assert ground_truth[key]["category"] == expected_category


@pytest.mark.parametrize(
    ("test_id", "expected_category"),
    CLASSIFICATION_CASES,
    ids=[test_id.split("::", 1)[1] for test_id, _ in CLASSIFICATION_CASES],
)
def test_classifier_assigns_ground_truth_category(
    test_id: str, expected_category: str
) -> None:
    classifier = _make_classifier()

    cause = classifier.classify(_flake_verdict(test_id))

    assert cause.test_id == test_id
    assert cause.category == expected_category, (
        f"expected {expected_category!r} for {test_id!r}; "
        f"got {cause.category!r}. Evidence: {cause.evidence}"
    )
    assert cause.auto_fixable is (expected_category in AUTO_FIXABLE_CATEGORIES)
    assert 0.0 <= cause.confidence <= 0.99
    assert cause.evidence, "classifier must always cite evidence"

    # The randomness experiment genuinely runs for these two (see
    # test_randomness_experiment_samples_fixed_seed_against_per_run_seed), but
    # neither demo case can produce a significant result, for opposite reasons:
    #
    #   * test_shuffle_preserves_first fails ~98% of the time unseeded. Seed 42
    #     makes random.shuffle deterministic — and puts a card other than 0
    #     first — so it fails 30/30 seeded vs ~29/30 unseeded. A fixed seed
    #     makes it deterministically *fail*; only ~1 seed in 52 would flip it.
    #   * test_round_trip fails ~5% of the time. Seed 42 makes it pass, so the
    #     comparison is 0/30 vs ~1.5/30 — Fisher needs ~6/30 to reach p<0.05,
    #     which happens ~6% of the time. Separating it reliably needs n~200.
    #
    # So the AST randomness signal is what carries these two, and asserting the
    # experiment fires would just make this suite flaky. The unit tests above
    # cover the experiment firing on evidence that can actually separate.
    if expected_category == "randomness":
        assert any("randomness" in signal.lower() for signal in cause.ast_signals)
