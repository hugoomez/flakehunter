"""Flaky-test detection orchestration."""

from __future__ import annotations

import random
from typing import Iterable

from flakehunter.contracts import (
    Detector as DetectorContract,
    FlakeVerdict,
    RunResult,
    SandboxRunner as SandboxRunnerContract,
)
from flakehunter.runner import SandboxRunner
from flakehunter.stats import wilson_interval


class Detector(DetectorContract):
    """Run suspected tests repeatedly and summarize observed flakiness."""

    def __init__(
        self,
        *,
        runner: SandboxRunnerContract | None = None,
        batch_seed: int = 0,
    ) -> None:
        self.runner = runner if runner is not None else SandboxRunner()
        self.batch_seed = batch_seed

    def detect(
        self,
        test_ids: list[str],
        *,
        n_runs: int,
        vary_order: bool,
        vary_seed: bool,
    ) -> list[FlakeVerdict]:
        if n_runs <= 0:
            raise ValueError("n_runs must be positive")

        rng = random.Random(self.batch_seed)
        seeds = self._distinct_seeds(rng=rng, n_runs=n_runs)

        verdicts: list[FlakeVerdict] = []
        for test_id in test_ids:
            results: list[RunResult] = []
            for run_index in range(n_runs):
                run_results = self.runner.run_once(
                    [test_id],
                    seed=seeds[run_index] if vary_seed else None,
                    forked=False,
                    randomize_order=vary_order,
                )
                results.extend(run_results)

            verdicts.append(self._verdict(test_id=test_id, results=results, n_runs=n_runs))

        return verdicts

    def _distinct_seeds(self, *, rng: random.Random, n_runs: int) -> list[int]:
        ordered_seeds: list[int] = []
        seeds: set[int] = set()
        while len(ordered_seeds) < n_runs:
            seed = rng.randrange(1, 2**32)
            if seed in seeds:
                continue
            seeds.add(seed)
            ordered_seeds.append(seed)
        return ordered_seeds

    def _verdict(
        self,
        *,
        test_id: str,
        results: list[RunResult],
        n_runs: int,
    ) -> FlakeVerdict:
        n_failures = sum(
            1 for result in results[:n_runs] if result.outcome in {"failed", "error"}
        )
        failure_rate = n_failures / n_runs
        _, ci95_upper = wilson_interval(k=n_failures, n=n_runs)

        return FlakeVerdict(
            test_id=test_id,
            n_runs=n_runs,
            n_failures=n_failures,
            failure_rate=failure_rate,
            ci95_upper=ci95_upper,
            is_flaky=0 < n_failures < n_runs,
            sample_tracebacks=self._sample_tracebacks(results[:n_runs]),
        )

    def _sample_tracebacks(self, results: Iterable[RunResult]) -> list[str]:
        sample_tracebacks: list[str] = []
        seen: set[str] = set()
        for result in results:
            if result.outcome not in {"failed", "error"}:
                continue
            if result.error_repr is None:
                continue
            if result.error_repr in seen:
                continue

            seen.add(result.error_repr)
            sample_tracebacks.append(result.error_repr)
            if len(sample_tracebacks) == 3:
                break

        return sample_tracebacks
