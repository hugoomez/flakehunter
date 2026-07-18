"""Fast orchestration tests: fake Fixer + faked components, no subprocess.

The four pipeline components are replaced with fakes (Detector/Classifier/
Verifier via monkeypatch of the module globals, Fixer via the ``fixer_factory``
injection seam) so we exercise the *sequencing* logic — phase transitions,
auto-fixable gating, and per-test exception isolation — in milliseconds and
without importing the Codex SDK.
"""

from __future__ import annotations

import pytest

from flakehunter import orchestrator
from flakehunter.orchestrator import Pipeline, PipelineEvent

from _builders import make_cause, make_proposal, make_result, make_verdict


class FakeFixer:
    """A Fixer that returns a hand-built proposal instantly."""

    def __init__(self) -> None:
        self.calls = 0

    def propose_fix(self, cause, verdict):  # type: ignore[no-untyped-def]
        self.calls += 1
        return make_proposal(cause.test_id)


def _boom_fixer_factory(root):  # type: ignore[no-untyped-def]
    raise AssertionError("fixer_factory must not be called on this path")


def _patch_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flaky: bool = True,
    category: str = "randomness",
    verify_verdict: str = "verified_fix",
    classify_exc: Exception | None = None,
) -> None:
    class FakeDetector:
        def __init__(self, *, runner=None, batch_seed=0):  # type: ignore[no-untyped-def]
            pass

        def detect(self, test_ids, *, n_runs, vary_order, vary_seed):  # type: ignore[no-untyped-def]
            return [make_verdict(t, is_flaky=flaky) for t in test_ids]

    class FakeClassifier:
        def __init__(self, runner, suite_test_ids, n_experiment_runs=30, *, repo_root=None):  # type: ignore[no-untyped-def]
            pass

        def classify(self, verdict):  # type: ignore[no-untyped-def]
            if classify_exc is not None:
                raise classify_exc
            return make_cause(verdict.test_id, category=category)

    class FakeVerifier:
        def __init__(self, *, repo_root, suite_test_ids, threshold=0.02, timeout_s=30.0, runner_factory=None):  # type: ignore[no-untyped-def]
            pass

        def verify(self, fix, before):  # type: ignore[no-untyped-def]
            return make_result(fix.test_id, verdict=verify_verdict)

    monkeypatch.setattr(orchestrator, "Detector", FakeDetector)
    monkeypatch.setattr(orchestrator, "Classifier", FakeClassifier)
    monkeypatch.setattr(orchestrator, "Verifier", FakeVerifier)


def _build(monkeypatch, *, fixer_factory=None, on_event=None, **kw) -> Pipeline:  # type: ignore[no-untyped-def]
    return Pipeline(
        repo_root=".",
        suite_test_ids=["a::t1", "a::t2"],
        fixer_factory=fixer_factory,
        runner_factory=lambda: object(),
        on_event=on_event,
        **kw,
    )


def test_not_flaky_stops_after_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_components(monkeypatch, flaky=False)
    events: list[PipelineEvent] = []
    pipeline = _build(monkeypatch, fixer_factory=_boom_fixer_factory, on_event=events.append)

    outcomes = pipeline.run(["a::t1"])

    assert [e.phase for e in events] == ["detecting", "not_flaky"]
    assert outcomes[0].phase == "not_flaky"
    assert outcomes[0].verdict is not None and not outcomes[0].verdict.is_flaky


def test_full_verified_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_components(monkeypatch, flaky=True, category="randomness", verify_verdict="verified_fix")
    fixer = FakeFixer()
    events: list[PipelineEvent] = []
    pipeline = _build(monkeypatch, fixer_factory=lambda root: fixer, on_event=events.append)

    outcomes = pipeline.run(["a::t1"])

    assert [e.phase for e in events] == [
        "detecting", "flaky", "classifying", "classified",
        "fixing", "fixed", "verifying", "verified",
    ]
    assert fixer.calls == 1
    result = outcomes[0].result
    assert outcomes[0].phase == "verified"
    assert result is not None and result.verdict == "verified_fix"


def test_not_auto_fixable_is_suggest_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_components(monkeypatch, flaky=True, category="timing")
    events: list[PipelineEvent] = []
    # The boom factory proves the Fixer is never even constructed.
    pipeline = _build(monkeypatch, fixer_factory=_boom_fixer_factory, on_event=events.append)

    outcomes = pipeline.run(["a::t1"])

    phases = [e.phase for e in events]
    assert phases == ["detecting", "flaky", "classifying", "classified", "suggest_only"]
    assert "fixing" not in phases
    assert outcomes[0].phase == "suggest_only"


def test_rejected_verdict_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_components(monkeypatch, verify_verdict="rejected_still_flaky")
    fixer = FakeFixer()
    events: list[PipelineEvent] = []
    pipeline = _build(monkeypatch, fixer_factory=lambda root: fixer, on_event=events.append)

    outcomes = pipeline.run(["a::t1"])

    assert events[-1].phase == "rejected"
    assert outcomes[0].result is not None
    assert outcomes[0].result.verdict == "rejected_still_flaky"


def test_exception_is_isolated_per_test(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_components(monkeypatch, classify_exc=RuntimeError("boom"))
    events: list[PipelineEvent] = []
    pipeline = _build(monkeypatch, fixer_factory=_boom_fixer_factory, on_event=events.append)

    # Two targets: the first raising must not abort the batch.
    outcomes = pipeline.run(["a::t1", "a::t2"])

    assert len(outcomes) == 2
    assert [o.phase for o in outcomes] == ["error", "error"]
    assert all("boom" in (o.error or "") for o in outcomes)
    assert [e.phase for e in events].count("error") == 2


def test_default_fixer_factory_builds_real_fixer(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The production seam: the default factory yields a real Fixer, lazily.

    Constructing it does not hit the Codex API (that only happens inside
    propose_fix), so this is safe without a working account.
    """
    pipeline = Pipeline(repo_root=tmp_path, suite_test_ids=[])
    fixer = pipeline._default_fixer_factory(tmp_path)
    from flakehunter.fixer import Fixer

    assert isinstance(fixer, Fixer)
    assert fixer.repo_root == tmp_path
