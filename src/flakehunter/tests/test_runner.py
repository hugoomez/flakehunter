from __future__ import annotations

import time
from pathlib import Path

from flakehunter.runner import SandboxRunner


def test_demo_stable_total_passes() -> None:
    runner = SandboxRunner()

    results = runner.run_once(
        ["demo/tests/test_stable_total.py"],
        seed=None,
        forked=False,
        randomize_order=False,
    )

    assert len(results) == 1
    assert results[0].outcome == "passed"
    assert results[0].error_repr is None
    assert len(results[0].order_hash) == 64


def test_demo_broken_feature_fails() -> None:
    runner = SandboxRunner()

    results = runner.run_once(
        ["demo/tests/test_broken_feature.py"],
        seed=None,
        forked=False,
        randomize_order=False,
    )

    assert len(results) == 1
    assert results[0].outcome == "failed"
    assert results[0].error_repr is not None
    assert "Feature X is deliberately broken" in results[0].error_repr


def test_timeout_returns_error_promptly(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sleepy.py"
    test_file.write_text(
        "import time\n\n"
        "def test_sleepy() -> None:\n"
        "    time.sleep(5)\n",
        encoding="utf-8",
    )
    runner = SandboxRunner(timeout_s=1)

    started_at = time.monotonic()
    results = runner.run_once(
        [str(test_file)],
        seed=None,
        forked=False,
        randomize_order=False,
    )
    elapsed_s = time.monotonic() - started_at

    assert len(results) == 1
    assert results[0].outcome == "error"
    assert results[0].error_repr is not None
    assert "timed out" in results[0].error_repr.lower()
    assert elapsed_s < 4
