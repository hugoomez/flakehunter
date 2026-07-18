"""Tests for the :class:`Verifier` — the Verification Contract.

The split mirrors ``test_classifier.py`` and ``test_fixer.py``: fast tests carry
the *reasoning*, slow "reality" tests carry the *truth*, and neither subsumes the
other. Crucially, **no test here touches Codex or the Fixer** — every
``FixProposal`` is hand-built, so the whole module runs against the real
``SandboxRunner`` and the demo repo with no Codex account.

* **Fast tests** exercise the structural (AST) check and the fail-fast wiring
  with an injected ``runner_factory`` that *raises* if pytest is ever launched,
  plus pure-function unit tests of the assertion counter and masking detector.
  Milliseconds, no subprocesses.
* **Slow "reality" tests** (``@pytest.mark.slow``) apply hand-built diffs to the
  real demo repo and run the actual ``SandboxRunner`` re-run batteries, covering
  all four verdicts. These launch tens–hundreds of pytest subprocesses and take
  minutes; run them with ``-m slow`` deliberately.
"""

from __future__ import annotations

import ast
import difflib

import pytest

from flakehunter.contracts import FixProposal, FlakeVerdict, RunResult
from flakehunter.runner import SandboxRunner
from flakehunter.stats import wilson_interval
from flakehunter.verifier import (
    Verifier,
    _count_assertions,
    _masking_markers,
    _structural_check,
    _test_function_name,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]

# --- Demo targets -----------------------------------------------------
_ROUND_TRIP = "demo/tests/test_pricing_rounding.py::test_round_trip"
_ROUND_TRIP_REL = "demo/tests/test_pricing_rounding.py"
_DECK = "demo/tests/test_deck.py::test_shuffle_preserves_first"
_DECK_REL = "demo/tests/test_deck.py"
_STABLE = "demo/tests/test_stable_total.py::test_total_is_correct"
_CART_REL = "demo/src/shopcart/cart.py"

# random.seed(0) makes random.choice pick a 0.104 candidate, so test_round_trip
# passes deterministically. Confirmed empirically (seeds 0-4 all pass; 5 fails).
_SEED_LITERAL = 0


# ======================================================================
# Helpers
# ======================================================================
def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _make_diff(rel: str, before: str, after: str) -> str:
    """A `git apply`-able unified diff, same shape the Fixer emits."""
    chunk = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )
    if chunk and not chunk.endswith("\n"):
        chunk += "\n"
    return chunk


def _seed_round_trip(seed: int = _SEED_LITERAL) -> str:
    """The genuine fix: seed the RNG at the top of the test body."""
    before = _read(_ROUND_TRIP_REL)
    return before.replace(
        "    rounding_candidates",
        f"    random.seed({seed})\n    rounding_candidates",
        1,
    )


def _fix(
    diff: str, *, test_id: str = _ROUND_TRIP, files_touched: list[str] | None = None
) -> FixProposal:
    return FixProposal(
        test_id=test_id,
        diff=diff,
        rationale="hand-built test fixture",
        files_touched=files_touched if files_touched is not None else [],
        codex_session_id="hand-built-fixture",
    )


def _verdict(
    *,
    test_id: str = _ROUND_TRIP,
    n_runs: int = 200,
    n_failures: int = 10,
) -> FlakeVerdict:
    _, ci = wilson_interval(k=n_failures, n=n_runs)
    return FlakeVerdict(
        test_id=test_id,
        n_runs=n_runs,
        n_failures=n_failures,
        failure_rate=n_failures / n_runs,
        ci95_upper=ci,
        is_flaky=0 < n_failures < n_runs,
        sample_tracebacks=[],
    )


class _ExplodingRunner:
    """A runner that fails the test if pytest is ever launched.

    Proves the structural fail-fast path never reaches the subprocess battery.
    """

    def __init__(self, cwd) -> None:
        self.cwd = cwd

    def run_once(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("run_once must not be called after a structural rejection")

    def run_isolated(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("run_isolated must not be called after a structural rejection")


def _exploding_factory():
    calls: list[object] = []

    def factory(cwd):
        runner = _ExplodingRunner(cwd)
        calls.append(runner)
        return runner

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


# ======================================================================
# Fast tests: structural fail-fast (no pytest subprocess)
# ======================================================================
def _fast_verifier(runner_factory) -> Verifier:
    return Verifier(
        repo_root=REPO_ROOT,
        suite_test_ids=[_ROUND_TRIP, _STABLE],
        threshold=0.02,
        runner_factory=runner_factory,
    )


def test_skip_decorator_is_rejected_before_any_run() -> None:
    before = _read(_ROUND_TRIP_REL)
    after = before.replace(
        "def test_round_trip",
        "@pytest.mark.skip(reason='flaky')\ndef test_round_trip",
        1,
    )
    after = "import pytest\n" + after
    factory = _exploding_factory()

    result = _fast_verifier(factory).verify(
        _fix(_make_diff(_ROUND_TRIP_REL, before, after)), _verdict()
    )

    assert result.verdict == "rejected_weakens_test"
    assert result.no_skip_introduced is False
    assert result.contract_passed is False
    # Fail-fast: no runner was ever built, so no subprocess battery ran.
    assert factory.calls == []


def test_removed_assertion_is_rejected_before_any_run() -> None:
    before = _read(_ROUND_TRIP_REL)
    after = before.replace("    assert round(adjustment, 2) == 0.10\n", "", 1)
    factory = _exploding_factory()

    result = _fast_verifier(factory).verify(
        _fix(_make_diff(_ROUND_TRIP_REL, before, after)), _verdict()
    )

    assert result.verdict == "rejected_weakens_test"
    assert result.assertion_count_preserved is False
    assert factory.calls == []


def test_try_except_wrapping_assert_is_rejected_before_any_run() -> None:
    before = _read(_ROUND_TRIP_REL)
    after = before.replace(
        "    assert round(adjustment, 2) == 0.10\n",
        "    try:\n"
        "        assert round(adjustment, 2) == 0.10\n"
        "    except AssertionError:\n"
        "        pass\n",
        1,
    )
    factory = _exploding_factory()

    result = _fast_verifier(factory).verify(
        _fix(_make_diff(_ROUND_TRIP_REL, before, after)), _verdict()
    )

    assert result.verdict == "rejected_weakens_test"
    assert result.no_skip_introduced is False
    assert factory.calls == []


def test_autouse_cleanup_fixture_is_not_a_weakening() -> None:
    """The explicit false-positive guard: a legit autouse fixture must pass."""
    before = _read(_ROUND_TRIP_REL)
    after = (
        "import pytest\n"
        + before
        + "\n\n@pytest.fixture(autouse=True)\ndef _reset_state() -> None:\n    pass\n"
    )

    preserved, no_skip = _structural_check(before, after, "test_round_trip")

    assert preserved is True
    assert no_skip is True


# ======================================================================
# Fast tests: assertion counter (pure function)
# ======================================================================
def _count(src: str, name: str = "test_target") -> int:
    tree = ast.parse(src)
    from flakehunter.verifier import _find_test_function

    fn = _find_test_function(tree, name)
    return _count_assertions(fn if fn is not None else tree)


def test_counts_bare_assert() -> None:
    src = "def test_target():\n    assert 1 == 1\n    assert True\n"
    assert _count(src) == 2


def test_counts_pytest_raises_and_warns() -> None:
    src = (
        "import pytest\n"
        "def test_target():\n"
        "    with pytest.raises(ValueError):\n"
        "        raise ValueError\n"
        "    with pytest.warns(UserWarning):\n"
        "        pass\n"
    )
    assert _count(src) == 2


def test_counts_unittest_style_asserts() -> None:
    src = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_target(self):\n"
        "        self.assertEqual(1, 1)\n"
        "        self.assertTrue(True)\n"
        "        self.assertIn(1, [1])\n"
    )
    assert _count(src) == 3


def test_counter_is_scoped_to_the_target_function() -> None:
    src = (
        "def test_target():\n    assert 1 == 1\n\n"
        "def test_other():\n    assert 2 == 2\n    assert 3 == 3\n"
    )
    assert _count(src, "test_target") == 1


# ======================================================================
# Fast tests: masking-marker detector (pure function)
# ======================================================================
def _markers(src: str, name: str = "test_target") -> set[str]:
    tree = ast.parse(src)
    from flakehunter.verifier import _find_test_function

    return _masking_markers(tree, _find_test_function(tree, name))


def test_detects_skip_decorator() -> None:
    src = (
        "import pytest\n"
        "@pytest.mark.skip\n"
        "def test_target():\n    assert True\n"
    )
    assert any("skip" in m for m in _markers(src))


def test_detects_xfail_and_flaky_decorators() -> None:
    src = (
        "import pytest\n"
        "@pytest.mark.xfail\n"
        "@pytest.mark.flaky(reruns=3)\n"
        "def test_target():\n    assert True\n"
    )
    markers = _markers(src)
    assert any("xfail" in m for m in markers)
    assert any("flaky" in m for m in markers)


def test_detects_module_level_pytestmark_skip() -> None:
    src = (
        "import pytest\n"
        "pytestmark = pytest.mark.skipif(True, reason='x')\n"
        "def test_target():\n    assert True\n"
    )
    assert any("skip" in m for m in _markers(src))


def test_fixture_and_parametrize_are_not_masking() -> None:
    src = (
        "import pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def _setup():\n    pass\n\n"
        "@pytest.mark.parametrize('x', [1, 2])\n"
        "def test_target(x):\n    assert x\n"
    )
    assert _markers(src) == set()


def test_test_function_name_parsing() -> None:
    assert _test_function_name("a/b.py::test_x") == "test_x"
    assert _test_function_name("a/b.py::TestC::test_x[param]") == "test_x"
    assert _test_function_name("a/b.py") is None


# ======================================================================
# Reality tests: real SandboxRunner against the demo repo (slow)
# ======================================================================
def _real_verifier(suite: list[str], threshold: float) -> Verifier:
    return Verifier(
        repo_root=REPO_ROOT,
        suite_test_ids=suite,
        threshold=threshold,
        timeout_s=30.0,
    )


@pytest.mark.slow
def test_genuine_seed_fix_is_verified() -> None:
    """A real seed fix drives test_round_trip to 0 failures -> verified_fix.

    threshold=0.02 => N=150, so rule_of_three_upper(150)=0.02 passes the gate at
    k_after=0, and before=(10/200) gives Fisher enough power to be significant.
    The suite is kept to two tests to bound the ~150 in-suite runs.
    """
    before = _read(_ROUND_TRIP_REL)
    after = _seed_round_trip()
    diff = _make_diff(_ROUND_TRIP_REL, before, after)

    result = _real_verifier([_ROUND_TRIP, _STABLE], threshold=0.02).verify(
        _fix(diff, files_touched=[_ROUND_TRIP_REL]), _verdict()
    )

    assert result.verdict == "verified_fix", result
    assert result.contract_passed is True
    assert result.suite_still_green is True
    assert result.assertion_count_preserved is True
    assert result.no_skip_introduced is True
    assert result.failure_rate_after == 0.0
    _, expected_ci = wilson_interval(k=0, n=150)
    assert result.ci95_upper_after == pytest.approx(expected_ci)


@pytest.mark.slow
def test_noop_diff_stays_flaky_is_rejected() -> None:
    """A structurally-valid diff that doesn't fix the flake -> rejected_still_flaky.

    Target is the ~98%-failing deck test, so it fails in essentially every run;
    even at threshold=0.2 (N=15) k_after is high, so both the CI bound and the
    Fisher test reject. Deterministic precisely because it fails near-certainly.
    """
    before = _read(_DECK_REL)
    after = _insert_first_body_line(before, "    _noop = 1  # does not fix the flake\n")
    diff = _make_diff(_DECK_REL, before, after)

    result = _real_verifier([_DECK, _STABLE], threshold=0.2).verify(
        _fix(diff, test_id=_DECK, files_touched=[_DECK_REL]),
        _verdict(test_id=_DECK, n_runs=30, n_failures=29),
    )

    assert result.verdict == "rejected_still_flaky", result
    assert result.contract_passed is False
    assert result.failure_rate_after > 0.2


@pytest.mark.slow
def test_fix_that_breaks_a_sibling_is_rejected() -> None:
    """Fixes the target but breaks a deterministic sibling -> rejected_breaks_suite.

    The diff seeds test_round_trip (genuine fix, passes structural + statistical)
    and also breaks shopcart.cart.Cart.total so the stable test_total_is_correct
    (normally 13.50) regresses. Deterministic because the broken sibling is stable.
    """
    rt_before = _read(_ROUND_TRIP_REL)
    rt_after = _seed_round_trip()
    cart_before = _read(_CART_REL)
    cart_after = _break_cart_total(cart_before)

    diff = _make_diff(_ROUND_TRIP_REL, rt_before, rt_after) + _make_diff(
        _CART_REL, cart_before, cart_after
    )

    result = _real_verifier([_ROUND_TRIP, _STABLE], threshold=0.02).verify(
        _fix(diff, files_touched=[_ROUND_TRIP_REL, _CART_REL]), _verdict()
    )

    assert result.verdict == "rejected_breaks_suite", result
    assert result.contract_passed is False
    assert result.suite_still_green is False


# ======================================================================
# Reality-test source-manipulation helpers
# ======================================================================
def _insert_first_body_line(source: str, line: str) -> str:
    """Insert ``line`` as the first statement of the file's single test function."""
    lines = source.splitlines(keepends=True)
    for i, text in enumerate(lines):
        if text.lstrip().startswith("def test_"):
            # Find the first indented body line after the def and insert before it.
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    return "".join(lines[:j] + [line] + lines[j:])
    raise AssertionError("no test function found to modify")


def _break_cart_total(cart_source: str) -> str:
    """Make Cart.total ignore discounts so the stable total test regresses."""
    broken = cart_source.replace(
        "        return round(subtotal * (1 - discount), 2)",
        "        return round(subtotal, 2)  # bug: discount dropped",
        1,
    )
    assert broken != cart_source, "cart.py total() line changed shape; update the break"
    return broken
