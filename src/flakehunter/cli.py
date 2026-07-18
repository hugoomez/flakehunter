"""Command-line entry point for FlakeHunter.

    flakehunter run|detect|classify|reproduce <repo> <test_ids...>

You always name the target test(s) explicitly, as positional arguments.
FlakeHunter is a *targeted diagnostic instrument* for tests already suspected
of being flaky -- it deliberately has no "scan the whole repo and find the
flaky ones" mode (see docs/ARCHITECTURE.md). The one thing it *does* discover
for you is the surrounding *suite context* (``run`` and ``classify`` call
``pytest --collect-only`` so the Classifier's order/shared-state experiments and
the Verifier's regression checks have the full suite to work with). That
asymmetry is intentional: the suite is plumbing you shouldn't hand-type, but the
targets are always yours to name -- pass ``--suite-file`` to override discovery.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path
from typing import Callable

from flakehunter.classifier import Classifier
from flakehunter.contracts import (
    FixProposal,
    FlakeVerdict,
    RootCause,
    RunResult,
)
from flakehunter.contracts import Fixer as FixerContract
from flakehunter.detector import Detector
from flakehunter.orchestrator import Pipeline, PipelineEvent, collect_suite
from flakehunter.runner import SandboxRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flakehunter",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser(
        "run",
        help="Full pipeline (detect -> classify -> fix -> verify) in the live TUI.",
    )
    _add_target_args(run_p)
    _add_experiment_args(run_p)
    _add_suite_args(run_p)
    run_p.add_argument("--threshold", type=float, default=0.02, help="Verifier target failure rate.")
    run_p.add_argument("--model", default=None, help="Codex model for the Fixer.")
    run_p.add_argument(
        "--demo-seed",
        type=int,
        default=0,
        help="Fix the internal random seed sequence for reproducible "
        "recording/demo runs (Detector's batch_seed; default 0).",
    )
    # DEMO/TESTING ONLY: not a real fix strategy, so it stays out of --help.
    # See _DemoCheatingFixer for what it actually does.
    run_p.add_argument(
        "--demo-cheating-fix",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_p.set_defaults(func=_cmd_run)

    detect_p = sub.add_parser("detect", help="Detector only; plain output.")
    _add_target_args(detect_p)
    _add_experiment_args(detect_p)
    detect_p.set_defaults(func=_cmd_detect)

    classify_p = sub.add_parser(
        "classify", help="Detector + Classifier; plain output."
    )
    _add_target_args(classify_p)
    _add_experiment_args(classify_p)
    _add_suite_args(classify_p)
    classify_p.set_defaults(func=_cmd_classify)

    reproduce_p = sub.add_parser(
        "reproduce",
        help="Deterministically re-run one test under a fixed seed.",
    )
    reproduce_p.add_argument("repo", help="Repo root (the pytest cwd).")
    reproduce_p.add_argument("test_id", help="Single node id (path::name).")
    reproduce_p.add_argument(
        "--seed", type=int, required=True, help="Fixed PYTHONHASHSEED / randomly-seed."
    )
    reproduce_p.set_defaults(func=_cmd_reproduce)

    return parser


def _add_target_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("repo", help="Repo root (the pytest cwd).")
    p.add_argument(
        "test_ids",
        nargs="+",
        metavar="TEST_ID",
        help="One or more node ids (path::name) already suspected flaky.",
    )


def _add_experiment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-runs", type=int, default=30, help="Detection re-runs per test.")
    p.add_argument(
        "--no-vary-order", dest="vary_order", action="store_false",
        help="Do not randomize collection order across detection runs.",
    )
    p.add_argument(
        "--no-vary-seed", dest="vary_seed", action="store_false",
        help="Do not vary the seed across detection runs.",
    )


def _add_suite_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--suite-file",
        default=None,
        help="File of suite node ids (one per line) to use instead of "
        "auto-discovering the suite via pytest --collect-only.",
    )


def _resolve_suite(args: argparse.Namespace, repo: Path) -> list[str]:
    if getattr(args, "suite_file", None):
        text = Path(args.suite_file).read_text(encoding="utf-8")
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return collect_suite(repo)


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------
def _cmd_detect(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    detector = Detector(runner=SandboxRunner(cwd=repo))
    verdicts = detector.detect(
        args.test_ids,
        n_runs=args.n_runs,
        vary_order=args.vary_order,
        vary_seed=args.vary_seed,
    )
    for verdict in verdicts:
        _print_verdict(verdict)
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    runner = SandboxRunner(cwd=repo)
    suite = _resolve_suite(args, repo)
    verdicts = Detector(runner=runner).detect(
        args.test_ids,
        n_runs=args.n_runs,
        vary_order=args.vary_order,
        vary_seed=args.vary_seed,
    )
    classifier = Classifier(runner, suite, repo_root=repo)
    for verdict in verdicts:
        _print_verdict(verdict)
        if verdict.is_flaky:
            _print_cause(classifier.classify(verdict))
    return 0


def _cmd_reproduce(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    result = SandboxRunner(cwd=repo).run_isolated(args.test_id, seed=args.seed)
    _print_run_result(result, seed=args.seed)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    # Import here so detect/classify/reproduce never load Textual.
    from flakehunter.tui import FlakeHunterApp

    repo = Path(args.repo)
    suite = _resolve_suite(args, repo)

    def make_pipeline(on_event: Callable[[PipelineEvent], None]) -> Pipeline:
        return Pipeline(
            repo_root=repo,
            suite_test_ids=suite,
            detect_runs=args.n_runs,
            vary_order=args.vary_order,
            vary_seed=args.vary_seed,
            threshold=args.threshold,
            model=args.model,
            batch_seed=args.demo_seed,
            fixer_factory=_DemoCheatingFixer if args.demo_cheating_fix else None,
            on_event=on_event,
        )

    FlakeHunterApp(test_ids=args.test_ids, pipeline_factory=make_pipeline).run()
    return 0


# ----------------------------------------------------------------------
# Plain-text rendering
# ----------------------------------------------------------------------
def _print_verdict(verdict: FlakeVerdict) -> None:
    flag = "FLAKY" if verdict.is_flaky else "stable"
    print(
        f"[{verdict.test_id}] {flag}  "
        f"failures={verdict.n_failures}/{verdict.n_runs}  "
        f"rate={verdict.failure_rate:.0%}  "
        f"ci95_upper={verdict.ci95_upper:.3f}"
    )
    for traceback in verdict.sample_tracebacks:
        first_line = traceback.strip().splitlines()[0] if traceback.strip() else ""
        print(f"    sample: {first_line}")


def _print_cause(cause: RootCause) -> None:
    print(
        f"    -> cause={cause.category}  confidence={cause.confidence:.2f}  "
        f"auto_fixable={cause.auto_fixable}"
    )
    for item in cause.evidence:
        print(f"       evidence: {item}")


def _print_run_result(result: RunResult, *, seed: int) -> None:
    print(f"[{result.test_id}] outcome={result.outcome}  duration={result.duration_s:.3f}s")
    print(f"    seed_env={result.seed_env}")
    if result.error_repr:
        print(f"    error: {result.error_repr.strip().splitlines()[0]}")
    print(
        f"    note: seed={seed} fixes the outcome deterministically, but for a "
        "randomness-driven flake that deterministic outcome is often a PASS. "
        "Reproducing the failure requires a seed already known to trigger it."
    )


# ----------------------------------------------------------------------
# DEMO/TESTING ONLY -- not a real fix strategy
# ----------------------------------------------------------------------
class _DemoCheatingFixer(FixerContract):
    """Fakes a "fix" that just skips the flaky test, never touching Codex.

    Wired in only by the hidden ``--demo-cheating-fix`` flag, to prove on
    demand that the Verifier's structural check genuinely rejects a fix that
    weakens the test (adds ``pytest.mark.skip``) rather than trusting
    whatever the Fixer hands it. The resulting ``FixProposal`` still flows
    through the real ``Verifier.verify`` unmodified -- the rejection is
    measured, not staged.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)

    def propose_fix(self, cause: RootCause, verdict: FlakeVerdict) -> FixProposal:
        source_rel = cause.test_id.split("::", 1)[0]
        before = (self.repo_root / source_rel).read_text(encoding="utf-8")
        test_name = cause.test_id.split("::")[-1].split("[", 1)[0]
        after = _add_skip_marker(before, test_name)

        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{source_rel}",
                tofile=f"b/{source_rel}",
            )
        )
        return FixProposal(
            test_id=cause.test_id,
            diff=diff,
            rationale="Marked as skip to avoid intermittent failures.",
            files_touched=[source_rel],
            codex_session_id="demo-cheating-fix",
        )


def _add_skip_marker(source: str, test_name: str) -> str:
    """Insert ``@pytest.mark.skip`` above ``test_name`` (plus the import if needed)."""
    tree = ast.parse(source)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == test_name
        ),
        None,
    )
    if target is None:
        raise ValueError(f"could not locate function {test_name!r} in source")

    lines = source.splitlines(keepends=True)
    def_lineno = (
        target.decorator_list[0].lineno if target.decorator_list else target.lineno
    )
    indent = " " * target.col_offset
    insertions = [(def_lineno - 1, f'{indent}@pytest.mark.skip(reason="flaky")\n')]

    has_pytest_import = any(
        (isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
        for node in tree.body
    )
    if not has_pytest_import:
        import_nodes = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
        if import_nodes:
            insertions.append((max(n.end_lineno for n in import_nodes), "import pytest\n"))
        else:
            insertions.append((0, "import pytest\n\n"))

    for index, text in sorted(insertions, key=lambda item: item[0], reverse=True):
        lines.insert(index, text)
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 1
    result: int = args.func(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
