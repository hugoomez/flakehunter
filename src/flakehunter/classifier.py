"""Root-cause classification for flaky tests via differential experiments.

The classifier takes a :class:`FlakeVerdict` produced by the detector and
decides *why* the test is flaky. It runs three controlled 2x2 experiments,
each summarised with a Fisher exact p-value comparing failure counts between
two conditions, and complements them with a static AST scan of the test's
source.

1. **Order dependency** — failures isolated vs. failures inside the full
   suite. Both conditions run unseeded with order randomisation on, so the
   only factor that varies is the surrounding suite context.
2. **Shared mutable state** — failures in the suite without vs. with process
   isolation (``--forked``). If forking *reduces* failures, the test is being
   polluted by shared mutable state rather than merely depending on ordering.
3. **Unseeded randomness** — failures under a fixed seed vs. under a seed that
   varies per run. Both conditions run the target alone, so the seed is the
   only factor that varies. If fixing the seed *reduces* failures, unseeded
   randomness is the driver.

Distinguishing "order" from "shared_state"
------------------------------------------
Both causes are context dependent, so experiment 1 fires for both. Two things
separate them:

* When ``--forked`` is available, experiment 2 is decisive: process isolation
  *removes* the pollution behind a shared-state flake but does nothing for a
  test that legitimately depends on setup performed by another test.
* ``pytest-forked`` needs ``os.fork``, which does not exist on Windows, so the
  forked experiment cannot run everywhere. Fortunately the *direction* of
  experiment 1 also separates the two causes and works on every platform:

  - a test **polluted** by shared state passes in isolation and fails in the
    suite — it fails *more* in-suite (``k_suite > k_iso``) → ``shared_state``;
  - a test that **depends on setup** from another test fails in isolation and
    passes in the suite — it fails *more* isolated (``k_iso > k_suite``) →
    ``order``.

  (Note: the SPEC's illustrative order example — "22/30 in-suite vs 0/30
  isolated" — is actually the pollution pattern, which ``demo/GROUND_TRUTH``
  labels ``shared_state``. Where the prose and the ground truth disagree we
  follow the ground truth, and use the forked experiment as the primary,
  stronger shared-state signal whenever fork is supported.)

The timing veto
---------------
A directional failure-rate difference is *weak* evidence: a load-sensitive
race also fails more often in-suite than alone, simply because the suite keeps
the machine busy. Read naively that pattern says ``shared_state``, which the
SPEC marks auto-fixable — so a timing flake would be handed to the fixer even
though the SPEC makes category D (timing) detect-only.

So before a ``shared_state`` or ``order`` verdict is finalised, it is vetoed
when *both* hold: the only evidence is a direction (no second, independent
experiment corroborates it), and the test carries a timing AST signal. Those
verdicts are downgraded to ``timing`` with ``auto_fixable=False``. A
``shared_state`` verdict on which the forked experiment and the in-suite
direction *agree* is corroborated and survives the veto.

Causes with no significant experimental signature
-------------------------------------------------
A test whose RNG makes it fail ~98% of the time (``random.shuffle`` asserting
a fixed first element) fails at essentially the same rate seeded or unseeded —
a fixed seed makes it *deterministically fail*, not pass — so experiment 3
stays flat. The same holds, for the opposite reason, for a test that fails
only ~5% of the time: at ``n=30`` the unseeded condition yields ~1.5 failures
against the seeded condition's 0, which Fisher cannot separate (it needs ~6).
For those the AST randomness signal (a ``random.*`` call with no fixed seed)
identifies the cause; the experiments merely fail to contradict it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from flakehunter.contracts import (
    Classifier as ClassifierContract,
    FlakeVerdict,
    RootCause,
    RunResult,
    SandboxRunner as SandboxRunnerContract,
)
from flakehunter.stats import fisher_exact_pvalue

# Two-sided significance threshold shared by all three experiments.
_ALPHA = 0.05
# Fixed seed used for the "seeded" condition of the randomness experiment.
_FIXED_SEED = 42
# Confidence used when only a static AST signal supports the diagnosis.
_AST_CONFIDENCE = 0.5
# Confidence used when nothing at all points to a specific cause.
_DEFAULT_CONFIDENCE = 0.3
# Experimental confidence is capped so a p-value of ~0 never reads as certainty.
_MAX_CONFIDENCE = 0.99

_REAL_OUTCOMES = frozenset({"passed", "failed"})
_FAILURE_OUTCOMES = frozenset({"failed", "error"})

_AUTO_FIXABLE_CATEGORIES = frozenset({"order", "shared_state", "randomness"})
# Verdicts that rest on a directional failure-rate difference, and so are
# subject to the timing veto.
_VETOABLE_CATEGORIES = frozenset({"order", "shared_state"})

Category = Literal["order", "shared_state", "randomness", "timing", "external"]


class Classifier(ClassifierContract):
    """Classify a flaky test's root cause from differential experiments + AST."""

    def __init__(
        self,
        runner: SandboxRunnerContract,
        suite_test_ids: list[str],
        n_experiment_runs: int = 30,
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        if n_experiment_runs <= 0:
            raise ValueError("n_experiment_runs must be positive")

        self.runner = runner
        self.suite_test_ids = list(suite_test_ids)
        self.n_experiment_runs = n_experiment_runs
        # Test ids carry a repo-relative path component (``path::name``); the
        # runner executes pytest from this directory, so source files resolve
        # relative to it. Fall back to the runner's cwd, then the process cwd.
        if repo_root is not None:
            self.repo_root = Path(repo_root)
        else:
            self.repo_root = Path(getattr(runner, "cwd", None) or Path.cwd())

    def classify(self, verdict: FlakeVerdict) -> RootCause:
        test_id = verdict.test_id
        n = self.n_experiment_runs

        # --- Experiment inputs -------------------------------------------
        # ``k_iso`` runs the target alone, unseeded, with order randomisation
        # on. It is sampled once and serves two comparisons, each of which
        # varies exactly one factor against it:
        #   * vs. ``k_suite`` (same seeding and ordering, full suite) — the
        #     suite context is the only difference;
        #   * vs. ``k_seeded`` (fixed seed) — the seed is the only *effective*
        #     difference, since with a single test id there is nothing for
        #     order randomisation to reorder.
        # Under seed=None pytest-randomly draws its own fresh seed per run, so
        # ``k_iso`` genuinely samples "seed varies" rather than "no plugin".
        k_iso = self._count_isolated_failures(test_id, seed=None, randomize_order=True)
        k_seeded = self._count_isolated_failures(
            test_id, seed=_FIXED_SEED, randomize_order=False
        )
        k_suite = self._count_suite_failures(test_id, forked=False)
        k_forked, forked_supported = self._count_forked_failures(test_id)

        # --- Experiment 1: order / context dependence --------------------
        p_order = fisher_exact_pvalue(k1=k_suite, n1=n, k2=k_iso, n2=n)
        order_significant = p_order < _ALPHA
        # Direction splits pollution (fails more in-suite) from missing setup
        # (fails more isolated).
        pollution = order_significant and k_suite > k_iso
        missing_setup = order_significant and k_iso > k_suite

        # --- Experiment 2: shared mutable state --------------------------
        p_shared = fisher_exact_pvalue(k1=k_forked, n1=n, k2=k_suite, n2=n)
        forked_shared = forked_supported and p_shared < _ALPHA and k_forked < k_suite

        # --- Experiment 3: unseeded randomness ---------------------------
        p_random = fisher_exact_pvalue(k1=k_seeded, n1=n, k2=k_iso, n2=n)
        random_significant = p_random < _ALPHA and k_seeded < k_iso

        ast_signals = self._ast_signals(test_id)
        ast_evidence = [s.description for s in ast_signals]
        has_random_ast = any(s.kind == "randomness" for s in ast_signals)
        has_timing_ast = any(s.kind == "timing" for s in ast_signals)
        has_external_ast = any(s.kind == "external" for s in ast_signals)

        evidence = [
            f"Fisher exact p={p_order:.4f}: {k_suite}/{n} failures in-suite "
            f"vs {k_iso}/{n} isolated (order / context dependence)",
            self._shared_experiment_evidence(
                p_shared, k_forked, k_suite, n, forked_supported
            ),
            f"Fisher exact p={p_random:.4f}: {k_seeded}/{n} failures with fixed "
            f"seed={_FIXED_SEED} vs {k_iso}/{n} with a per-run seed (randomness)",
        ]

        # --- Decision -----------------------------------------------------
        # Shared state wins when either the forked experiment confirms it or
        # experiment 1 shows the pollution direction (fails more in-suite).
        # A pure order dependency shows the opposite direction (fails more
        # isolated). Anything not explained experimentally falls back to AST.
        category: Category
        # True when the verdict rests on a single directional difference with
        # no independent corroboration — the weak evidence the timing veto
        # guards against.
        direction_only = False
        if forked_shared or pollution:
            category = "shared_state"
            direction_only = not (forked_shared and pollution)
            if forked_shared:
                confidence = self._experimental_confidence(p_shared)
                evidence.append(
                    f"Process isolation reduced failures "
                    f"({k_suite}/{n} -> {k_forked}/{n}); root cause is shared state"
                )
            else:
                confidence = self._experimental_confidence(p_order)
                evidence.append(
                    f"Fails more in-suite than isolated "
                    f"({k_suite}/{n} vs {k_iso}/{n}); polluted by shared state"
                )
            if not direction_only:
                evidence.append(
                    "Forked experiment and in-suite direction agree; shared "
                    "state is corroborated by two independent experiments"
                )
        elif missing_setup:
            category = "order"
            direction_only = True
            confidence = self._experimental_confidence(p_order)
            evidence.append(
                f"Fails more isolated than in-suite "
                f"({k_iso}/{n} vs {k_suite}/{n}); depends on setup by another test"
            )
        elif random_significant:
            category = "randomness"
            confidence = self._experimental_confidence(p_random)
            evidence.append(
                f"Fixing the seed reduced failures "
                f"({k_iso}/{n} -> {k_seeded}/{n}); root cause is unseeded randomness"
            )
        elif has_random_ast:
            # No experiment separated the conditions, but the source uses an
            # unseeded RNG. Covers random assertions the seeded condition
            # cannot separate: near-always-failing ones (a fixed seed only
            # makes the failure deterministic) and rare ones (too few failures
            # at this n for Fisher to resolve).
            category = "randomness"
            confidence = _AST_CONFIDENCE
            evidence.append(
                "No experiment was significant; classified from AST randomness "
                "signal (unseeded RNG usage)"
            )
        elif has_timing_ast:
            category = "timing"
            confidence = _AST_CONFIDENCE
            evidence.append(
                "No experiment was significant; classified from AST timing signal"
            )
        elif has_external_ast:
            category = "external"
            confidence = _AST_CONFIDENCE
            evidence.append(
                "No experiment was significant; classified from AST external "
                "dependency signal"
            )
        else:
            # Nothing pointed to a specific cause. Default to timing (a race we
            # could not reproduce under our controlled conditions) with low
            # confidence rather than asserting a fixable category.
            category = "timing"
            confidence = _DEFAULT_CONFIDENCE
            evidence.append(
                "No experiment was significant and no AST signal was found; "
                "defaulting to timing with low confidence"
            )

        # --- Timing veto ---------------------------------------------------
        if category in _VETOABLE_CATEGORIES and direction_only and has_timing_ast:
            evidence.append(
                f"Downgraded {category} -> timing: the only evidence was a "
                f"directional failure-rate difference, which a load-sensitive "
                f"race reproduces just as well (the suite keeps the machine "
                f"busy), and the test carries a timing signal. No second "
                f"independent experiment corroborated {category}, so the "
                f"weaker-but-detect-only diagnosis wins and no fix is proposed"
            )
            category = "timing"
            confidence = _AST_CONFIDENCE

        evidence.extend(ast_evidence)

        return RootCause(
            test_id=test_id,
            category=category,
            confidence=confidence,
            evidence=evidence,
            ast_signals=ast_evidence,
            auto_fixable=category in _AUTO_FIXABLE_CATEGORIES,
        )

    # ------------------------------------------------------------------
    # Experiment execution helpers
    # ------------------------------------------------------------------
    def _count_isolated_failures(
        self, test_id: str, *, seed: int | None, randomize_order: bool
    ) -> int:
        failures = 0
        for _ in range(self.n_experiment_runs):
            results = self.runner.run_once(
                [test_id],
                seed=seed,
                forked=False,
                randomize_order=randomize_order,
            )
            if self._failed(results, test_id):
                failures += 1
        return failures

    def _count_suite_failures(self, test_id: str, *, forked: bool) -> int:
        failures = 0
        for _ in range(self.n_experiment_runs):
            results = self.runner.run_once(
                self.suite_test_ids,
                seed=None,
                forked=forked,
                randomize_order=True,
            )
            if self._failed(results, test_id):
                failures += 1
        return failures

    def _count_forked_failures(self, test_id: str) -> tuple[int, bool]:
        """Count forked-suite failures, short-circuiting if fork is unavailable.

        ``pytest-forked`` needs ``os.fork`` and produces no results at all on
        platforms without it (e.g. Windows). The first run doubles as a
        capability probe: if it yields no real pytest outcomes we stop and
        report the experiment as unsupported rather than spending
        ``n_experiment_runs`` fruitless subprocesses.
        """
        first = self.runner.run_once(
            self.suite_test_ids,
            seed=None,
            forked=True,
            randomize_order=True,
        )
        if not any(r.outcome in _REAL_OUTCOMES for r in first):
            return self.n_experiment_runs, False

        failures = 1 if self._failed(first, test_id) else 0
        for _ in range(self.n_experiment_runs - 1):
            results = self.runner.run_once(
                self.suite_test_ids,
                seed=None,
                forked=True,
                randomize_order=True,
            )
            if self._failed(results, test_id):
                failures += 1
        return failures, True

    @staticmethod
    def _failed(results: list[RunResult], test_id: str) -> bool:
        for result in results:
            if result.test_id == test_id:
                return result.outcome in _FAILURE_OUTCOMES
        # The target did not appear in the report (e.g. suite-level error or
        # timeout). Treat that as a failure for the target under test.
        return True

    @staticmethod
    def _experimental_confidence(p_value: float) -> float:
        return min(1.0 - p_value, _MAX_CONFIDENCE)

    @staticmethod
    def _shared_experiment_evidence(
        p_shared: float,
        k_forked: int,
        k_suite: int,
        n: int,
        forked_supported: bool,
    ) -> str:
        if not forked_supported:
            return (
                "Shared-state (forked) experiment unsupported on this platform "
                "(os.fork unavailable); relying on experiment 1 direction"
            )
        return (
            f"Fisher exact p={p_shared:.4f}: {k_forked}/{n} failures forked "
            f"vs {k_suite}/{n} unforked in-suite (shared mutable state)"
        )

    # ------------------------------------------------------------------
    # AST analysis
    # ------------------------------------------------------------------
    def _ast_signals(self, test_id: str) -> list["_AstSignal"]:
        source_path = self._source_path(test_id)
        if source_path is None:
            return []
        try:
            source = source_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_path))
        except (OSError, SyntaxError):
            return []
        return scan_ast(tree, self._test_function_name(test_id))

    def _source_path(self, test_id: str) -> Path | None:
        path_component = test_id.split("::", 1)[0]
        if not path_component:
            return None
        candidate = Path(path_component)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return candidate if candidate.is_file() else None

    @staticmethod
    def _test_function_name(test_id: str) -> str | None:
        """Extract the test function's name from ``path::[Class::]name[param]``."""
        parts = test_id.split("::")
        if len(parts) < 2:
            return None
        return parts[-1].split("[", 1)[0] or None


class _AstSignal:
    """A single static signal detected in a test's source."""

    __slots__ = ("kind", "description")

    def __init__(self, kind: str, description: str) -> None:
        self.kind = kind
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_AstSignal(kind={self.kind!r}, description={self.description!r})"


# Canonical (alias-resolved) dotted names that indicate a timing dependency.
_TIMING_CALLS = frozenset(
    {
        "time.sleep",
        "asyncio.sleep",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
    }
)
# Canonical prefixes for RNG modules.
_RANDOM_PREFIXES = ("random.", "numpy.random.")
# Canonical names that fix an RNG's seed when given a literal argument.
_SEEDING_CALLS = frozenset(
    {
        "random.seed",
        "random.Random",
        "numpy.random.seed",
        "numpy.random.default_rng",
        "numpy.random.RandomState",
    }
)
# Canonical prefixes for external/network dependencies.
_EXTERNAL_PREFIXES = ("requests.", "socket.", "urllib.", "httpx.")


class _Symbols:
    """Maps locally bound names to their canonical dotted origin.

    Built by walking ``Import``/``ImportFrom`` so the scanner can recognise
    ``rnd.choice`` (``import random as rnd``) and ``sleep()`` (``from time
    import sleep``) rather than only literal ``random.``/``time.`` prefixes.
    """

    def __init__(self) -> None:
        self._aliases: dict[str, str] = {}

    def add(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    # import numpy.random as npr -> npr == numpy.random
                    self._aliases[alias.asname] = alias.name
                else:
                    # import numpy.random binds only the top-level "numpy".
                    top = alias.name.split(".", 1)[0]
                    self._aliases[top] = top
            return

        # Relative imports resolve outside this module; leave them unmapped.
        if node.level:
            return
        module = node.module or ""
        for alias in node.names:
            bound = alias.asname or alias.name
            self._aliases[bound] = f"{module}.{alias.name}" if module else alias.name

    def resolve(self, dotted: str) -> str:
        """Rewrite a dotted call target onto its canonical origin."""
        head, _, tail = dotted.partition(".")
        base = self._aliases.get(head)
        if base is None:
            return dotted
        return f"{base}.{tail}" if tail else base


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted call target (e.g. ``numpy.random.choice``) or None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _has_literal_seed(node: ast.Call) -> bool:
    """True when the call fixes a seed with a literal (``random.Random(42)``).

    A bare ``random.seed()`` reseeds from system entropy and a
    ``random.Random(compute_seed())`` may too, so neither counts as seeded.
    """
    for arg in (*node.args, *(kw.value for kw in node.keywords)):
        if isinstance(arg, ast.Constant) and isinstance(
            arg.value, int | float | str | bytes
        ):
            return True
    return False


def _fixed_path_argument(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _find_test_function(
    tree: ast.Module, test_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Locate the test's own def, including inside a test class."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == test_name
        ):
            return node
    return None


def _executes_at_import(node: ast.stmt) -> bool:
    """True for module-level statements whose body runs at import time.

    A ``def``'s body does not run at import, and a ``class``'s method bodies do
    not either — only the class-level statements do, which are handled
    separately.
    """
    return not isinstance(
        node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    )


def _is_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = _dotted_name(target)
        if name is not None and name.split(".")[-1] == "fixture":
            return True
    return False


def _is_autouse_fixture(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if (
                keyword.arg == "autouse"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return True
    return False


def _argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    args = node.args
    return {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}


def _scope_nodes(tree: ast.Module, test_name: str | None) -> list[ast.AST]:
    """Nodes whose code actually runs when ``test_name`` runs.

    Scanning the whole file would let one test's calls speak for every other
    test in it — including its ``random.seed()``, which would suppress the
    randomness signal for unrelated tests. The scope is the test's own def,
    plus the module-level code that runs before it: import-time statements,
    class-level constants, and the fixtures it requests.
    """
    if test_name is None:
        return [tree]
    target = _find_test_function(tree, test_name)
    if target is None:
        # Unknown test (e.g. generated by a plugin): fall back to the module so
        # a missing scope never silently drops every signal.
        return [tree]

    nodes: list[ast.AST] = [target]
    requested = _argument_names(target)

    for node in tree.body:
        if _executes_at_import(node):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            nodes.extend(stmt for stmt in node.body if _executes_at_import(stmt))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node is target:
                continue
            if _is_fixture(node) and (
                node.name in requested or _is_autouse_fixture(node)
            ):
                nodes.append(node)

    return nodes


def scan_ast(tree: ast.Module, test_name: str | None = None) -> list[_AstSignal]:
    """Detect timing, randomness, and external-dependency signals for a test.

    ``test_name`` scopes the scan to that test (see :func:`_scope_nodes`); when
    omitted the whole module is scanned. Import aliases are always resolved
    module-wide, since an import binds a module-level name regardless of which
    test uses it.
    """
    symbols = _Symbols()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            symbols.add(node)

    calls: list[tuple[str, ast.Call]] = []
    for scope in _scope_nodes(tree, test_name):
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            raw = _dotted_name(node.func)
            if raw is not None:
                calls.append((symbols.resolve(raw), node))

    signals: list[_AstSignal] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, description: str) -> None:
        key = (kind, description)
        if key not in seen:
            seen.add(key)
            signals.append(_AstSignal(kind, description))

    # A fixed seed in scope suppresses the randomness signal: the RNG draws are
    # reproducible, so they cannot be what makes the test flaky.
    has_seed_call = any(
        name in _SEEDING_CALLS and _has_literal_seed(node) for name, node in calls
    )

    for name, node in calls:
        if name in _TIMING_CALLS:
            add("timing", f"calls {name} (timing dependency)")
        elif name in _SEEDING_CALLS and _has_literal_seed(node):
            add("seeded", f"calls {name} with a fixed seed (RNG is reproducible)")
        elif name.startswith(_RANDOM_PREFIXES) and not has_seed_call:
            add("randomness", f"calls {name} without a fixed seed (unseeded randomness)")
        elif name.startswith(_EXTERNAL_PREFIXES):
            add("external", f"calls {name} (external network dependency)")
        elif name == "open":
            fixed_path = _fixed_path_argument(node)
            if fixed_path is not None:
                add(
                    "external",
                    f"opens fixed path {fixed_path!r} (external IO dependency)",
                )

    return signals
