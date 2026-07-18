"""Pytest subprocess runner for FlakeHunter sandbox executions."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path
from typing import Any, Literal, Mapping

from flakehunter.contracts import RunResult, SandboxRunner as SandboxRunnerContract


Outcome = Literal["passed", "failed", "error", "skipped"]

_PROBE_TIMEOUT_S = 10.0


def _venv_site_packages() -> str:
    return sysconfig.get_path("purelib")


def _can_import_pytest(executable: str, *, extra_env: Mapping[str, str]) -> bool:
    """Probe whether ``executable`` can spawn and import pytest from a cwd

    other than this process's own -- exactly the condition that silently
    broke Fixer/Verifier's temporary sandbox copies.
    """
    env = os.environ.copy()
    env.update(extra_env)
    try:
        with tempfile.TemporaryDirectory() as probe_cwd:
            result = subprocess.run(
                [executable, "-c", "import pytest"],
                cwd=probe_cwd,
                env=env,
                capture_output=True,
                timeout=_PROBE_TIMEOUT_S,
            )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@functools.lru_cache(maxsize=1)
def _default_python_executable() -> tuple[str, dict[str, str]]:
    """Resolve a Python executable that reliably spawns regardless of cwd.

    ``sys.executable`` inside a uv-managed venv on Windows can be a small
    (~45KB) trampoline stub that re-execs the real interpreter; spawning it
    from certain working directories -- notably Fixer/Verifier's temporary
    sandbox copies -- can fail with "uv trampoline failed to spawn Python
    child process". ``sys._base_executable`` (Python >= 3.11) is the real
    interpreter underneath the trampoline and spawns reliably regardless of
    cwd, but bypassing the trampoline also bypasses the venv discovery it
    performs, so the venv's site-packages must be added back explicitly via
    PYTHONPATH. Platforms without the trampoline distinction (the attribute
    is absent, or identical to ``sys.executable``) just use ``sys.executable``
    unmodified, as before.

    If neither is a working non-trampoline binary, fall back to a
    system-wide ``python``/``py`` install with the venv's site-packages on
    PYTHONPATH, same idea as ``base_executable``.
    """
    site_packages = _venv_site_packages()
    base = getattr(sys, "_base_executable", None)

    if base and base != sys.executable:
        extra_env = {"PYTHONPATH": site_packages}
        if _can_import_pytest(base, extra_env=extra_env):
            return base, extra_env

    if _can_import_pytest(sys.executable, extra_env={}):
        return sys.executable, {}

    for candidate in (shutil.which("python"), shutil.which("py")):
        if not candidate:
            continue
        extra_env = {"PYTHONPATH": site_packages}
        if _can_import_pytest(candidate, extra_env=extra_env):
            return candidate, extra_env

    # Nothing verified: keep the historical default so failures surface as
    # the familiar subprocess spawn error rather than a resolution crash.
    return sys.executable, {}


class SandboxRunner(SandboxRunnerContract):
    """Run pytest in a fresh subprocess and parse pytest-json-report output."""

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        timeout_s: float = 30.0,
        python_executable: str | Path | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.timeout_s = timeout_s
        user_env = dict(extra_env) if extra_env is not None else {}
        if python_executable is not None:
            self.python_executable = str(python_executable)
            self.extra_env = user_env
        else:
            resolved_executable, resolved_env = _default_python_executable()
            self.python_executable = resolved_executable
            self.extra_env = {**resolved_env, **user_env}

    def run_once(
        self,
        test_ids: list[str],
        *,
        seed: int | None,
        forked: bool,
        randomize_order: bool,
    ) -> list[RunResult]:
        if not test_ids:
            return []

        report_path = self._new_report_path()
        seed_env = self._seed_env(seed)
        started_at = time.monotonic()

        try:
            completed = subprocess.run(
                self._pytest_command(
                    test_ids=test_ids,
                    report_path=report_path,
                    seed=seed,
                    forked=forked,
                    randomize_order=randomize_order,
                ),
                cwd=self.cwd,
                env=self._subprocess_env(seed=seed),
                timeout=self.timeout_s,
                capture_output=True,
                text=True,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_s = time.monotonic() - started_at
            error_repr = f"pytest subprocess timed out after {self.timeout_s:.3g}s"
            if exc.cmd:
                error_repr = f"{error_repr}: {exc.cmd!r}"
            report_path.unlink(missing_ok=True)
            return [
                RunResult(
                    test_id=test_id,
                    outcome="error",
                    duration_s=duration_s,
                    error_repr=error_repr,
                    seed_env=seed_env,
                    order_hash=self._order_hash([]),
                )
                for test_id in test_ids
            ]
        except OSError as exc:
            duration_s = time.monotonic() - started_at
            report_path.unlink(missing_ok=True)
            return [
                RunResult(
                    test_id=test_id,
                    outcome="error",
                    duration_s=duration_s,
                    error_repr=f"pytest subprocess could not be started: {exc}",
                    seed_env=seed_env,
                    order_hash=self._order_hash([]),
                )
                for test_id in test_ids
            ]

        try:
            report = self._load_report(report_path)
        except (OSError, json.JSONDecodeError) as exc:
            duration_s = time.monotonic() - started_at
            error_repr = self._process_error_repr(completed=completed, exc=exc)
            return [
                RunResult(
                    test_id=test_id,
                    outcome="error",
                    duration_s=duration_s,
                    error_repr=error_repr,
                    seed_env=seed_env,
                    order_hash=self._order_hash([]),
                )
                for test_id in test_ids
            ]
        finally:
            report_path.unlink(missing_ok=True)

        return self._results_from_report(
            report=report,
            requested_test_ids=test_ids,
            seed_env=seed_env,
            fallback_duration_s=time.monotonic() - started_at,
        )

    def run_isolated(self, test_id: str, *, seed: int | None) -> RunResult:
        results = self.run_once(
            [test_id],
            seed=seed,
            forked=False,
            randomize_order=False,
        )
        if len(results) != 1:
            raise RuntimeError(
                f"isolated run for {test_id!r} produced {len(results)} results"
            )
        return results[0]

    def _pytest_command(
        self,
        *,
        test_ids: list[str],
        report_path: Path,
        seed: int | None,
        forked: bool,
        randomize_order: bool,
    ) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={report_path}",
        ]

        if seed is not None:
            command.append(f"--randomly-seed={seed}")
        if forked:
            command.append("--forked")
        if not randomize_order:
            # Keeps pytest-randomly active (so --randomly-seed still seeds the
            # RNG) while leaving collection order untouched. `-p no:randomly`
            # would disable the plugin and silently drop the seed.
            command.append("--randomly-dont-reorganize")

        command.extend(test_ids)
        return command

    def _subprocess_env(self, *, seed: int | None) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env)
        if seed is not None:
            env["PYTHONHASHSEED"] = str(seed)
        return env

    def _seed_env(self, seed: int | None) -> dict[str, str]:
        if seed is None:
            return {}
        return {
            "PYTHONHASHSEED": str(seed),
            "randomly-seed": str(seed),
        }

    def _new_report_path(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="flakehunter-", suffix=".json")
        os.close(fd)
        return Path(path)

    def _load_report(self, report_path: Path) -> dict[str, Any]:
        with report_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
        if not isinstance(report, dict):
            raise json.JSONDecodeError("pytest JSON report was not an object", "", 0)
        return report

    def _results_from_report(
        self,
        *,
        report: dict[str, Any],
        requested_test_ids: list[str],
        seed_env: dict[str, str],
        fallback_duration_s: float,
    ) -> list[RunResult]:
        tests = report.get("tests")
        if not isinstance(tests, list) or not tests:
            error_repr = self._collection_error_repr(report)
            return [
                RunResult(
                    test_id=test_id,
                    outcome="error",
                    duration_s=fallback_duration_s,
                    error_repr=error_repr,
                    seed_env=seed_env,
                    order_hash=self._order_hash([]),
                )
                for test_id in requested_test_ids
            ]

        ordered_nodeids = [
            test.get("nodeid", "")
            for test in tests
            if isinstance(test, dict) and isinstance(test.get("nodeid"), str)
        ]
        order_hash = self._order_hash(ordered_nodeids)

        results: list[RunResult] = []
        for test in tests:
            if not isinstance(test, dict):
                continue
            nodeid = test.get("nodeid")
            if not isinstance(nodeid, str):
                continue
            results.append(
                RunResult(
                    test_id=nodeid,
                    outcome=self._outcome(test),
                    duration_s=self._duration_s(test),
                    error_repr=self._error_repr(test),
                    seed_env=seed_env,
                    order_hash=order_hash,
                )
            )

        if results:
            return results

        error_repr = self._collection_error_repr(report)
        return [
            RunResult(
                test_id=test_id,
                outcome="error",
                duration_s=fallback_duration_s,
                error_repr=error_repr,
                seed_env=seed_env,
                order_hash=order_hash,
            )
            for test_id in requested_test_ids
        ]

    def _outcome(self, test: dict[str, Any]) -> Outcome:
        outcome = test.get("outcome")
        if outcome in {"passed", "failed", "skipped"}:
            return outcome
        return "error"

    def _duration_s(self, test: dict[str, Any]) -> float:
        total = 0.0
        found_stage_duration = False
        for stage_name in ("setup", "call", "teardown"):
            stage = test.get(stage_name)
            if not isinstance(stage, dict):
                continue
            duration = stage.get("duration")
            if isinstance(duration, int | float):
                found_stage_duration = True
                total += float(duration)
        if found_stage_duration:
            return total

        duration = test.get("duration")
        if isinstance(duration, int | float):
            return float(duration)
        return 0.0

    def _error_repr(self, test: dict[str, Any]) -> str | None:
        outcome = self._outcome(test)
        if outcome not in {"failed", "error"}:
            return None

        for stage_name in ("call", "setup", "teardown"):
            stage = test.get(stage_name)
            if not isinstance(stage, dict):
                continue

            longrepr = stage.get("longrepr")
            if isinstance(longrepr, str) and longrepr:
                return longrepr

            crash = stage.get("crash")
            if isinstance(crash, dict):
                message = crash.get("message")
                if isinstance(message, str) and message:
                    return message

        return f"pytest reported outcome={outcome!r}"

    def _collection_error_repr(self, report: dict[str, Any]) -> str:
        collectors = report.get("collectors")
        if isinstance(collectors, list):
            for collector in collectors:
                if not isinstance(collector, dict):
                    continue
                if collector.get("outcome") != "failed":
                    continue
                longrepr = collector.get("longrepr")
                if isinstance(longrepr, str) and longrepr:
                    return longrepr
        return "pytest JSON report contained no test results"

    def _process_error_repr(
        self,
        *,
        completed: subprocess.CompletedProcess[str],
        exc: BaseException,
    ) -> str:
        stderr = completed.stderr.strip() if completed.stderr else ""
        stdout = completed.stdout.strip() if completed.stdout else ""
        process_output = stderr or stdout
        if process_output:
            return f"pytest JSON report could not be read: {exc}; output: {process_output}"
        return f"pytest JSON report could not be read: {exc}"

    def _order_hash(self, ordered_nodeids: list[str]) -> str:
        encoded = json.dumps(
            ordered_nodeids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
