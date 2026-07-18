"""Wire Detector -> Classifier -> Fixer -> Verifier into one pipeline.

This module owns the *sequencing* of the four already-built components and the
seams that make that sequencing testable without a live Codex account. The UI
layers (``cli.py``, ``tui.py``) call into here; the pipeline itself knows
nothing about argparse or Textual.

Why ``collect_suite`` exists while ``test_ids`` stay mandatory
--------------------------------------------------------------
There are two different lists of test ids in play, and the asymmetry between
them is deliberate — not an inconsistency:

* The **target** ``test_ids`` are always supplied explicitly by the user and
  are never discovered. FlakeHunter is a *targeted diagnostic instrument* for a
  test (or small set) already suspected of being flaky, not a "scan the whole
  suite and find the flaky ones" scanner (see ``docs/ARCHITECTURE.md``). Every
  contract in ``contracts.py`` takes an explicit ``test_ids``; there is no
  discovery entry point, by design, because detecting a 5% failure rate at 95%
  confidence costs ~59 isolated re-runs per test and classification adds ~90
  subprocess launches — that depth does not scale to breadth.

* The **suite context** (``suite_test_ids``) is an *internal* concern that
  ``collect_suite`` discovers automatically. The Classifier's order and
  shared-state experiments run the whole suite, and the Verifier watches the
  target while re-running its siblings, so both need the full membership. That
  is plumbing the user should not have to hand-type, so we discover it via
  ``pytest --collect-only``. Discovering the *suite* is not the same as
  discovering *targets*: the tool still only ever diagnoses the tests you name.

The Fixer injection seam
------------------------
``import flakehunter.fixer`` eagerly imports the ``openai_codex`` SDK, and there
is no working Codex account this month. So ``Pipeline`` takes a
``fixer_factory`` whose default *lazily* imports ``Fixer`` only when a fix is
actually generated. Tests inject a factory returning a fake Fixer, and thereby
never import the SDK nor touch the (dead) account — the same dependency-
injection idea Fixer's ``codex_factory`` and Verifier's ``runner_factory``
already use, lifted one level up to the orchestrator.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from flakehunter.classifier import Classifier
from flakehunter.contracts import (
    Fixer as FixerContract,
    FixProposal,
    FlakeVerdict,
    RootCause,
    SandboxRunner as SandboxRunnerContract,
    VerificationResult,
)
from flakehunter.detector import Detector
from flakehunter.runner import SandboxRunner
from flakehunter.verifier import Verifier

__all__ = [
    "Phase",
    "PipelineEvent",
    "TestOutcome",
    "Pipeline",
    "collect_suite",
    "SuiteDiscoveryError",
]

# The status a test can be in as it moves through the pipeline. Emitted as
# events so a UI can render live progress; the terminal ones also become the
# ``phase`` of the returned ``TestOutcome``.
Phase = Literal[
    "detecting",
    "flaky",
    "not_flaky",
    "classifying",
    "classified",
    "fixing",
    "fixed",
    "verifying",
    "verified",
    "rejected",
    "suggest_only",
    "error",
]

FixerFactory = Callable[[Path], FixerContract]
RunnerFactory = Callable[[], SandboxRunnerContract]
EventCallback = Callable[["PipelineEvent"], None]


@dataclass(frozen=True)
class PipelineEvent:
    """A single live status update for one test as the pipeline advances."""

    test_id: str
    phase: Phase
    verdict: FlakeVerdict | None = None
    cause: RootCause | None = None
    proposal: FixProposal | None = None
    result: VerificationResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class TestOutcome:
    """The terminal state of one test after the pipeline finishes with it."""

    test_id: str
    phase: Phase
    verdict: FlakeVerdict | None = None
    cause: RootCause | None = None
    proposal: FixProposal | None = None
    result: VerificationResult | None = None
    error: str | None = None


class SuiteDiscoveryError(Exception):
    """``pytest --collect-only`` produced no usable node ids."""


def collect_suite(
    repo_root: str | Path,
    *,
    python_executable: str | Path | None = None,
) -> list[str]:
    """Discover the full suite's test ids via ``pytest --collect-only -q``.

    Returns the node ids (``path::name``) pytest reports, relative to
    ``repo_root`` — the same cwd the runner executes from, so the ids line up
    with what the components expect. This discovers *suite context*, never
    *targets*: see the module docstring for why that distinction matters.
    """
    python = str(python_executable) if python_executable is not None else sys.executable
    proc = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    # ``-q`` prints one node id per line followed by a summary line; only the
    # node ids contain "::". Parametrized ids keep their "[...]" suffix.
    test_ids = [line.strip() for line in proc.stdout.splitlines() if "::" in line]
    if not test_ids:
        raise SuiteDiscoveryError(
            "pytest --collect-only discovered no tests in "
            f"{repo_root!r} (exit {proc.returncode}). "
            "Pass --suite-file to supply the suite explicitly.\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return test_ids


class Pipeline:
    """Drive Detector -> Classifier -> Fixer -> Verifier for named test ids."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        suite_test_ids: list[str],
        detect_runs: int = 30,
        vary_order: bool = True,
        vary_seed: bool = True,
        threshold: float = 0.02,
        timeout_s: float = 30.0,
        model: str | None = None,
        batch_seed: int = 0,
        fixer_factory: FixerFactory | None = None,
        runner_factory: RunnerFactory | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.suite_test_ids = list(suite_test_ids)
        self.detect_runs = detect_runs
        self.vary_order = vary_order
        self.vary_seed = vary_seed
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.model = model
        self.batch_seed = batch_seed
        self._fixer_factory = fixer_factory or self._default_fixer_factory
        self._runner_factory = runner_factory or self._default_runner_factory
        self._on_event = on_event

    # -- injection defaults ------------------------------------------------
    def _default_fixer_factory(self, root: Path) -> FixerContract:
        # Lazy import: keeps the ``openai_codex`` SDK off the import path for
        # everything except a real fix generation, so cli.py and the test suite
        # import cleanly without a working Codex account.
        from flakehunter.fixer import Fixer

        return Fixer(repo_root=root, model=self.model)

    def _default_runner_factory(self) -> SandboxRunnerContract:
        return SandboxRunner(cwd=self.repo_root, timeout_s=self.timeout_s)

    # -- driving loop ------------------------------------------------------
    def run(self, test_ids: list[str]) -> list[TestOutcome]:
        return [self._run_one(test_id) for test_id in test_ids]

    def _run_one(self, test_id: str) -> TestOutcome:
        verdict: FlakeVerdict | None = None
        cause: RootCause | None = None
        proposal: FixProposal | None = None
        try:
            runner = self._runner_factory()

            self._emit(PipelineEvent(test_id, "detecting"))
            verdict = Detector(runner=runner, batch_seed=self.batch_seed).detect(
                [test_id],
                n_runs=self.detect_runs,
                vary_order=self.vary_order,
                vary_seed=self.vary_seed,
            )[0]

            if not verdict.is_flaky:
                self._emit(PipelineEvent(test_id, "not_flaky", verdict=verdict))
                return TestOutcome(test_id, "not_flaky", verdict=verdict)
            self._emit(PipelineEvent(test_id, "flaky", verdict=verdict))

            self._emit(PipelineEvent(test_id, "classifying", verdict=verdict))
            cause = Classifier(
                runner, self.suite_test_ids, repo_root=self.repo_root
            ).classify(verdict)
            self._emit(
                PipelineEvent(test_id, "classified", verdict=verdict, cause=cause)
            )

            if not cause.auto_fixable:
                # Categories D (timing) and E (external): a fix is suggested,
                # never auto-applied, so the pipeline stops here.
                self._emit(
                    PipelineEvent(test_id, "suggest_only", verdict=verdict, cause=cause)
                )
                return TestOutcome(test_id, "suggest_only", verdict=verdict, cause=cause)

            self._emit(PipelineEvent(test_id, "fixing", verdict=verdict, cause=cause))
            fixer = self._fixer_factory(self.repo_root)
            proposal = fixer.propose_fix(cause, verdict)
            self._emit(
                PipelineEvent(
                    test_id, "fixed", verdict=verdict, cause=cause, proposal=proposal
                )
            )

            self._emit(
                PipelineEvent(
                    test_id, "verifying", verdict=verdict, cause=cause, proposal=proposal
                )
            )
            result = Verifier(
                repo_root=self.repo_root,
                suite_test_ids=self.suite_test_ids,
                threshold=self.threshold,
                timeout_s=self.timeout_s,
            ).verify(proposal, verdict)
            phase: Phase = "verified" if result.verdict == "verified_fix" else "rejected"
            self._emit(
                PipelineEvent(
                    test_id,
                    phase,
                    verdict=verdict,
                    cause=cause,
                    proposal=proposal,
                    result=result,
                )
            )
            return TestOutcome(
                test_id,
                phase,
                verdict=verdict,
                cause=cause,
                proposal=proposal,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 - one bad test must not abort the batch
            error = f"{type(exc).__name__}: {exc}"
            self._emit(
                PipelineEvent(
                    test_id,
                    "error",
                    verdict=verdict,
                    cause=cause,
                    proposal=proposal,
                    error=error,
                )
            )
            return TestOutcome(
                test_id,
                "error",
                verdict=verdict,
                cause=cause,
                proposal=proposal,
                error=error,
            )

    def _emit(self, event: PipelineEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)
