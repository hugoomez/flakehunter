from __future__ import annotations

import sys
import time
from pathlib import Path

from flakehunter.runner import SandboxRunner, _default_python_executable


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


def test_seed_is_passed_when_order_is_not_randomized() -> None:
    runner = SandboxRunner()

    command = runner._pytest_command(
        test_ids=["demo/tests/test_stable_total.py"],
        report_path=Path("report.json"),
        seed=4242,
        forked=False,
        randomize_order=False,
    )

    assert "--randomly-seed=4242" in command
    assert "--randomly-dont-reorganize" in command
    assert "no:randomly" not in command


def test_seed_is_passed_when_order_is_randomized() -> None:
    runner = SandboxRunner()

    command = runner._pytest_command(
        test_ids=["demo/tests/test_stable_total.py"],
        report_path=Path("report.json"),
        seed=4242,
        forked=False,
        randomize_order=True,
    )

    assert "--randomly-seed=4242" in command
    assert "--randomly-dont-reorganize" not in command


def test_unseeded_randomized_order_lets_plugin_choose_seed() -> None:
    runner = SandboxRunner()

    command = runner._pytest_command(
        test_ids=["demo/tests/test_stable_total.py"],
        report_path=Path("report.json"),
        seed=None,
        forked=False,
        randomize_order=True,
    )

    assert not any(arg.startswith("--randomly-seed") for arg in command)
    assert "--randomly-dont-reorganize" not in command


def test_unseeded_fixed_order_disables_reordering_only() -> None:
    runner = SandboxRunner()

    command = runner._pytest_command(
        test_ids=["demo/tests/test_stable_total.py"],
        report_path=Path("report.json"),
        seed=None,
        forked=False,
        randomize_order=False,
    )

    assert not any(arg.startswith("--randomly-seed") for arg in command)
    assert "--randomly-dont-reorganize" in command
    assert "no:randomly" not in command


def test_seed_env_records_seed_when_order_is_not_randomized() -> None:
    runner = SandboxRunner()

    seed_env = runner._seed_env(4242)

    assert seed_env["randomly-seed"] == "4242"
    assert seed_env["PYTHONHASHSEED"] == "4242"


def _write_random_probe(tmp_path: Path, output_path: Path) -> Path:
    """A test file that appends the RNG's first draw to output_path."""
    test_file = tmp_path / "test_random_probe.py"
    test_file.write_text(
        "import random\n"
        "from pathlib import Path\n"
        "\n"
        f"OUTPUT = Path({str(output_path)!r})\n"
        "\n"
        "def test_records_random_draw() -> None:\n"
        "    with OUTPUT.open('a', encoding='utf-8') as handle:\n"
        "        handle.write(f'{random.random()}\\n')\n",
        encoding="utf-8",
    )
    return test_file


def test_same_seed_reproduces_rng_without_randomizing_order(tmp_path: Path) -> None:
    output_path = tmp_path / "draws.txt"
    test_file = _write_random_probe(tmp_path, output_path)
    runner = SandboxRunner()

    for _ in range(2):
        results = runner.run_once(
            [str(test_file)],
            seed=4242,
            forked=False,
            randomize_order=False,
        )
        assert len(results) == 1
        assert results[0].outcome == "passed"

    draws = output_path.read_text(encoding="utf-8").split()
    assert len(draws) == 2
    assert draws[0] == draws[1]


def test_differing_seeds_produce_differing_rng_without_randomizing_order(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "draws.txt"
    test_file = _write_random_probe(tmp_path, output_path)
    runner = SandboxRunner()

    for seed in (4242, 17):
        results = runner.run_once(
            [str(test_file)],
            seed=seed,
            forked=False,
            randomize_order=False,
        )
        assert len(results) == 1
        assert results[0].outcome == "passed"

    draws = output_path.read_text(encoding="utf-8").split()
    assert len(draws) == 2
    assert draws[0] != draws[1]


def test_run_once_succeeds_from_a_cwd_outside_the_project(tmp_path: Path) -> None:
    """Regression test: Fixer/Verifier run pytest from a fresh temp-dir copy
    of the repo, not the project's own cwd. A trampoline-stub
    ``python_executable`` (uv-managed venvs on Windows) can fail to spawn
    from such a cwd with "uv trampoline failed to spawn Python child
    process" -- a failure that silently degraded to outcome="error" with no
    test coverage catching it.
    """
    assert tmp_path.resolve() != Path.cwd().resolve()

    test_file = tmp_path / "test_sandboxed.py"
    test_file.write_text(
        "def test_ok() -> None:\n    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )
    runner = SandboxRunner(cwd=tmp_path)

    results = runner.run_once(
        ["test_sandboxed.py"],
        seed=None,
        forked=False,
        randomize_order=False,
    )

    assert len(results) == 1
    assert results[0].outcome == "passed", results[0].error_repr
    assert results[0].error_repr is None


def test_default_python_executable_resolves_to_a_working_binary() -> None:
    _default_python_executable.cache_clear()
    try:
        executable, extra_env = _default_python_executable()
    finally:
        _default_python_executable.cache_clear()

    assert Path(executable).is_file()
    base = getattr(sys, "_base_executable", None)
    if base and base != sys.executable:
        # The base interpreter bypasses the trampoline but needs the venv's
        # site-packages added back manually.
        assert executable == base
        assert "PYTHONPATH" in extra_env
    else:
        assert executable == sys.executable
        assert extra_env == {}


def test_explicit_python_executable_bypasses_resolution() -> None:
    runner = SandboxRunner(python_executable="/some/fixed/python")
    assert runner.python_executable == "/some/fixed/python"
    assert runner.extra_env == {}


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
