"""CLI tests: argument parsing + command wiring, with components faked.

No subprocess, no Textual event loop, no Codex. The full pipeline is exercised
elsewhere; here we prove argparse produces the right namespaces and each
subcommand dispatches with the arguments the user gave.
"""

from __future__ import annotations

import pytest

from flakehunter import cli
from flakehunter.orchestrator import Pipeline

from _builders import make_run_result, make_verdict


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def test_detect_parses_defaults() -> None:
    args = cli.build_parser().parse_args(["detect", "repo", "a.py::t1", "a.py::t2"])
    assert args.command == "detect"
    assert args.repo == "repo"
    assert args.test_ids == ["a.py::t1", "a.py::t2"]
    assert args.n_runs == 30
    assert args.vary_order is True
    assert args.vary_seed is True


def test_experiment_flags_toggle() -> None:
    args = cli.build_parser().parse_args(
        ["detect", "repo", "a.py::t1", "--n-runs", "59", "--no-vary-order", "--no-vary-seed"]
    )
    assert args.n_runs == 59
    assert args.vary_order is False
    assert args.vary_seed is False


def test_target_test_ids_are_required() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["detect", "repo"])


def test_reproduce_requires_seed() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["reproduce", "repo", "a.py::t1"])


def test_no_command_prints_help_and_returns_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 1
    assert "flakehunter" in capsys.readouterr().out


# ----------------------------------------------------------------------
# reproduce
# ----------------------------------------------------------------------
class _FakeRunner:
    instances: list["_FakeRunner"] = []

    def __init__(self, *, cwd=None, timeout_s=30.0):  # type: ignore[no-untyped-def]
        self.cwd = cwd
        self.calls: list[tuple[str, int]] = []
        _FakeRunner.instances.append(self)

    def run_isolated(self, test_id, *, seed):  # type: ignore[no-untyped-def]
        self.calls.append((test_id, seed))
        return make_run_result(test_id, seed=seed)


def test_reproduce_runs_isolated_with_fixed_seed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _FakeRunner.instances = []
    monkeypatch.setattr(cli, "SandboxRunner", _FakeRunner)

    rc = cli.main(["reproduce", "myrepo", "a.py::t1", "--seed", "42"])

    assert rc == 0
    runner = _FakeRunner.instances[-1]
    assert str(runner.cwd) == "myrepo"
    assert runner.calls == [("a.py::t1", 42)]
    out = capsys.readouterr().out
    assert "outcome=passed" in out
    assert "note:" in out  # the honest fixed-seed caveat is printed


# ----------------------------------------------------------------------
# detect / classify
# ----------------------------------------------------------------------
class _FakeDetector:
    last: dict | None = None

    def __init__(self, *, runner=None, batch_seed=0):  # type: ignore[no-untyped-def]
        pass

    def detect(self, test_ids, *, n_runs, vary_order, vary_seed):  # type: ignore[no-untyped-def]
        _FakeDetector.last = {
            "test_ids": test_ids,
            "n_runs": n_runs,
            "vary_order": vary_order,
            "vary_seed": vary_seed,
        }
        return [make_verdict(t, is_flaky=True) for t in test_ids]


def test_detect_command_dispatches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "SandboxRunner", _FakeRunner)
    monkeypatch.setattr(cli, "Detector", _FakeDetector)

    rc = cli.main(["detect", "repo", "a.py::t1", "--n-runs", "12", "--no-vary-seed"])

    assert rc == 0
    assert _FakeDetector.last == {
        "test_ids": ["a.py::t1"],
        "n_runs": 12,
        "vary_order": True,
        "vary_seed": False,
    }
    assert "FLAKY" in capsys.readouterr().out


class _FakeClassifier:
    last_classified: list = []

    def __init__(self, runner, suite_test_ids, n_experiment_runs=30, *, repo_root=None):  # type: ignore[no-untyped-def]
        self.suite = suite_test_ids

    def classify(self, verdict):  # type: ignore[no-untyped-def]
        _FakeClassifier.last_classified.append(verdict.test_id)
        from _builders import make_cause

        return make_cause(verdict.test_id, category="randomness")


def test_classify_uses_suite_file_and_skips_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    _FakeClassifier.last_classified = []
    suite_file = tmp_path / "suite.txt"
    suite_file.write_text("# comment\na.py::t1\nb.py::t2\n", encoding="utf-8")

    def _no_discovery(*a, **k):  # type: ignore[no-untyped-def]
        raise AssertionError("collect_suite must not run when --suite-file is given")

    monkeypatch.setattr(cli, "SandboxRunner", _FakeRunner)
    monkeypatch.setattr(cli, "Detector", _FakeDetector)
    monkeypatch.setattr(cli, "Classifier", _FakeClassifier)
    monkeypatch.setattr(cli, "collect_suite", _no_discovery)

    rc = cli.main(
        ["classify", "repo", "a.py::t1", "--suite-file", str(suite_file)]
    )

    assert rc == 0
    # Flaky verdict -> classified; suite came from the file (comment stripped).
    assert _FakeClassifier.last_classified == ["a.py::t1"]
    assert "cause=randomness" in capsys.readouterr().out


# ----------------------------------------------------------------------
# run (TUI) wiring
# ----------------------------------------------------------------------
class _FakeApp:
    captured: dict | None = None

    def __init__(self, *, test_ids, pipeline_factory):  # type: ignore[no-untyped-def]
        _FakeApp.captured = {"test_ids": test_ids, "pipeline_factory": pipeline_factory}

    def run(self):  # type: ignore[no-untyped-def]
        return None


def test_run_builds_pipeline_with_cli_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import flakehunter.tui as tui_mod

    monkeypatch.setattr(tui_mod, "FlakeHunterApp", _FakeApp)
    monkeypatch.setattr(cli, "collect_suite", lambda repo: ["a.py::t1", "a.py::t2"])

    rc = cli.main(
        ["run", "repo", "a.py::t1", "--n-runs", "40", "--threshold", "0.05", "--model", "gpt-x"]
    )

    assert rc == 0
    captured = _FakeApp.captured
    assert captured is not None
    assert captured["test_ids"] == ["a.py::t1"]

    # The factory yields a Pipeline carrying the CLI's configuration.
    pipeline = captured["pipeline_factory"](lambda event: None)
    assert isinstance(pipeline, Pipeline)
    assert str(pipeline.repo_root) == "repo"
    assert pipeline.suite_test_ids == ["a.py::t1", "a.py::t2"]
    assert pipeline.detect_runs == 40
    assert pipeline.threshold == 0.05
    assert pipeline.model == "gpt-x"
