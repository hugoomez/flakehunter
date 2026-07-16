from __future__ import annotations

import json
from pathlib import Path

import pytest

from flakehunter.detector import Detector


REPO_ROOT = Path(__file__).resolve().parents[3]
GROUND_TRUTH_PATH = REPO_ROOT / "demo" / "GROUND_TRUTH.json"


def test_detector_flags_demo_flakes_without_deterministic_false_positives() -> None:
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    flaky_test_ids = [
        "demo/tests/test_deck.py::test_shuffle_preserves_first",
        "demo/tests/test_pricing_rounding.py::test_round_trip",
        "demo/tests/test_async_worker.py::test_worker_completes",
    ]
    deterministic_test_ids = [
        "demo/tests/test_broken_feature.py::test_feature_x",
        "demo/tests/test_stable_total.py::test_total_is_correct",
    ]
    test_ids = flaky_test_ids + deterministic_test_ids

    # Production diagnostic use would choose n_runs with n_for_detection()
    # based on a target sensitivity. The demo test keeps this at 30 for speed.
    verdicts = Detector(batch_seed=6).detect(
        test_ids,
        n_runs=30,
        vary_order=True,
        vary_seed=True,
    )
    verdict_by_id = {verdict.test_id: verdict for verdict in verdicts}

    assert [verdict.test_id for verdict in verdicts] == test_ids

    for test_id in flaky_test_ids:
        verdict = verdict_by_id[test_id]
        expected_min, expected_max = ground_truth[_ground_truth_key(test_id)][
            "expected_rate"
        ]

        assert verdict.is_flaky is True
        assert expected_min <= verdict.failure_rate <= expected_max
        assert verdict.n_runs == 30
        assert verdict.n_failures == pytest.approx(verdict.failure_rate * 30)
        assert len(verdict.sample_tracebacks) <= 3

    broken_verdict = verdict_by_id["demo/tests/test_broken_feature.py::test_feature_x"]
    stable_verdict = verdict_by_id["demo/tests/test_stable_total.py::test_total_is_correct"]

    assert broken_verdict.is_flaky is False
    assert broken_verdict.failure_rate == 1.0
    assert broken_verdict.n_failures == 30

    assert stable_verdict.is_flaky is False
    assert stable_verdict.failure_rate == 0.0
    assert stable_verdict.n_failures == 0
    assert stable_verdict.sample_tracebacks == []


def _ground_truth_key(test_id: str) -> str:
    return test_id.removeprefix("demo/")
