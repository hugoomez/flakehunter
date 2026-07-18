# FlakeHunter

**FlakeHunter detects, classifies, fixes (via Codex), and — critically — statistically *proves* whether a flaky Python test is actually fixed, rejecting anything it can't prove.**

## Try it in 60 seconds

```bash
uv sync
```

That's it for setup — no build step, `uv sync` resolves and installs everything (`openai-codex`, `scipy`, `statsmodels`, `textual`, plus the dev/test tooling) from `pyproject.toml`/`uv.lock` into `.venv`, and wires up the `flakehunter` console script. Activate the venv, then run the exact command below from the repo root:

```bash
flakehunter run . demo/tests/test_pricing_rounding.py::test_round_trip
```

This launches a live 3-panel Textual TUI against `demo/tests/test_pricing_rounding.py::test_round_trip` — a real, deliberately seeded ~5% flake in the `demo/` shopping-cart repo (see `docs/DEMO_CASES.md`, Case 4). Watch it move through `detecting → flaky → classifying → classified → fixing → verifying → verified` live: N isolated re-runs with a Wilson confidence interval, three Fisher-exact experiments plus an AST scan to diagnose *why* it's flaky, a Codex-generated fix, and a real statistical verification battery — no mocks, every stage launches actual pytest subprocesses against a throwaway copy of the repo.

Two honest caveats: the `fixing` stage calls the real Codex SDK (`fixer.py`), so it needs Codex API access to complete — `flakehunter classify . demo/tests/test_pricing_rounding.py::test_round_trip` runs detection + diagnosis only, no Codex account required, if you just want to see the statistical engine. And the full pipeline is statistically thorough by design (the verifier alone re-runs the suite `n_for_verification(threshold)` times — 150 runs at the default `threshold=0.02`), so end-to-end completion takes a few minutes, not 60 seconds; getting it running and watching it work live is the 60-second part.

## The problem — and why statistical rigor matters

A flaky test's outcome on an *unchanged* commit isn't a fixed fact — it's a Bernoulli trial with some unknown true failure probability `p`. "I ran it 10 times and it passed" says almost nothing: a test that truly fails 20% of the time still passes ten times in a row about 11% of the time by chance alone. FlakeHunter's entire design follows from taking that seriously — flakiness is a parameter to be *estimated*, with a confidence interval, not a yes/no fact established by eyeballing a few runs. `stats.py` exists so every claim FlakeHunter makes ("this test is flaky," "this fix reduced the failure rate," "this is the root cause") is backed by an interval or a p-value instead of a run count (`docs/ARCHITECTURE.md` §1).

That statistical honesty has a direct architectural consequence: FlakeHunter is a **targeted diagnostic instrument for a test (or small set) already suspected of being flaky**, not a whole-suite CI scanner. Every contract in `contracts.py` takes an explicit `test_ids` — there's no "scan the repo and find the flaky ones" mode. Detecting a test with a 5% failure rate at 95% confidence needs on the order of dozens of isolated re-runs (`n_for_detection(0.05, beta=0.05) == 59`), and classification adds roughly 90 more subprocess launches per test at the default experiment size. That cost is worth paying for a test a developer or CI system has already flagged; it does not scale to re-running an entire suite dozens of times over. FlakeHunter optimizes for depth on a named target, deliberately, rather than breadth across a suite.

## Architecture: Detector → Classifier → Fixer → Verifier

```
 test_ids (explicit, user-named — never auto-discovered)
        │
        ▼
 ┌─────────────────────┐
 │  Detector             │  runner.py launches N fresh, isolated pytest
 │  (detector.py)        │  subprocesses, optionally varying seed/order
 └──────────┬────────────┘  → FlakeVerdict (n_failures, failure_rate,
            │                 Wilson 95% CI, is_flaky)
            │ is_flaky? ──── no ──▶ stop: "not_flaky"
            │ yes
            ▼
 ┌─────────────────────┐
 │  Classifier           │  3× Fisher-exact 2×2 experiments (isolated vs.
 │  (classifier.py)      │  in-suite, forked vs. unforked, seeded vs.
 └──────────┬────────────┘  unseeded) + a scoped AST scan
            │                 → RootCause (category, confidence, evidence)
            │ auto_fixable? (order / shared_state / randomness)
            │ no (timing / external) ──▶ stop: "suggest_only"
            │ yes
            ▼
 ┌─────────────────────┐
 │  Fixer                │  Prompts the Codex SDK with the diagnosis,
 │  (fixer.py)           │  category-specific instructions, and hard
 └──────────┬────────────┘  constraints (never weaken the assertion)
            │                 → FixProposal (a unified diff)
            ▼
 ┌─────────────────────┐
 │  Verifier              │  Applies the diff to a throwaway copy of the
 │  (verifier.py)         │  repo and runs the Verification Contract below
 └──────────┬────────────┘
            ▼
   VerificationResult: verified_fix
   or rejected_weakens_test / rejected_still_flaky / rejected_breaks_suite
```

`orchestrator.py`'s `Pipeline` owns exactly this sequencing (nothing about argparse or Textual) and emits a `PipelineEvent` at every phase transition, which `tui.py` renders live and `cli.py`'s `run` subcommand drives. The `Classifier`'s order/shared-state experiments and the `Verifier`'s regression check both need the surrounding suite, so `collect_suite()` discovers that automatically via `pytest --collect-only`. That's the one thing FlakeHunter *does* discover for you; the targets you're diagnosing are always yours to name explicitly.

## The Verification Contract

A `FixProposal` — however it was generated — is never trusted on inspection. `Verifier.verify()` runs three checks, in a fixed fail-fast order (each cheaper than the next), and a fix is `verified_fix` only if **all three** pass:

1. **Structural.** Does the diff weaken the test? Checked with a pure AST pass, no subprocess involved: the assertion count in the target test's own scope must not decrease, and no `skip`/`xfail`/`flaky`/rerun marker or `try/except` wrapping an assertion may be newly introduced. A legitimate `@pytest.fixture(autouse=True)` a real fix adds is explicitly *not* treated as masking. Fails this stage → `rejected_weakens_test`, immediately, with no pytest subprocess ever launched.
2. **Statistical.** Re-run the *whole suite* (not just the target in isolation — an order or shared-state flake only reproduces alongside its siblings) `n_for_verification(threshold)` times with randomized order, and require **both**: the post-fix failure rate's confidence bound falls below the target threshold, *and* the before/after improvement is significant by Fisher's exact test. A numerically lower failure rate alone is not enough — it has to be a statistically real improvement. Fails this stage → `rejected_still_flaky`.
3. **Regression.** Run the suite once on the pristine tree and once on the fixed tree; every test that passed before must still read `"passed"` after (missing from the report, newly skipped, or failed all count as a regression). Fails this stage → `rejected_breaks_suite`.

Every `VerificationResult` therefore states *which* clause of the contract failed, not a bare pass/fail — the same discipline the Verifier applies to Codex-generated fixes applies to every claim FlakeHunter makes about itself.

## Where Codex accelerated development

Codex plays two distinct roles in this project, worth separating clearly: it is both a **coding agent** that helped build FlakeHunter, and a **runtime dependency** the finished product calls (`fixer.py` imports `openai_codex` to generate the actual test fixes). This section is about the first role.

Per `AGENTS.md`'s documented workflow ("work on a `codex/<module>` branch for substantial features; I will review and merge") and the commit history, **Codex built the foundational modules against the frozen `contracts.py` interfaces** — the ones where the shape of the answer is already pinned down by a fixed spec and a known statistical formula, so an autonomous agent has to get the arithmetic and plumbing right rather than invent the design:

- `contracts.py` — the five frozen dataclasses (`feat(contracts): add component interface stubs (hand-reviewed)`)
- `stats.py` and the first pass of `runner.py` — the Wilson/Fisher/rule-of-three primitives and the sandboxed subprocess runner (`feat(runner)`)
- `detector.py` — the N-run isolated detection loop (`feat(detector)`)
- `verifier.py` — the full three-stage Verification Contract (`feat(verifier)`)

**This Claude Code session did the auditing, integration, and cross-file work** — which is also, concretely, why `classifier.py`, `fixer.py`, `orchestrator.py`, `tui.py`, the `cli.py` rewrite, and their whole test suite remain uncommitted in this working tree rather than sitting behind their own `feat(...)` commits:

- **Found and fixed a real statistical bug.** `docs/ARCHITECTURE.md` §4 records that an independent audit of `classifier.py` found its randomness experiment was statistically void — `PYTHONHASHSEED` doesn't seed Python's `random` module, and the runner only passed `--randomly-seed` when order randomization was also on, so the "fixed seed" condition was silently identical to "unseeded." The fix touched `runner.py`'s `_pytest_command`/`_subprocess_env` to decouple seeding from order randomization — which is why `runner.py` shows as modified against its original `feat(runner)` commit instead of sitting untouched.
- **Built `fixer.py`'s Codex SDK integration**, including a behavior its own docstring documents finding by running the real thing rather than reading type signatures: a failed Codex turn never actually returns `TurnStatus.failed` — the SDK raises a bare `RuntimeError` instead — "found by running the real E2E test... It is the reason the fake in `test_fixer.py` now models the *raise*."
- **Wired all four components into `orchestrator.py`**, whose module docstring is explicit about the constraint that shaped its design: "`import flakehunter.fixer` eagerly imports the `openai_codex` SDK, and there is no working Codex account this month" — hence the injectable `fixer_factory` seam that lets the Pipeline, `cli.py`'s `detect`/`classify` paths, and the test suite run without one.
- **Built `tui.py`'s live 3-panel view and the real `cli.py` command surface** (`run|detect|classify|reproduce`) that `docs/SPEC.md` named but that stayed a `print("flakehunter placeholder")` stub through the modules above.

## Statistical rigor (`stats.py`)

| Function | Chosen because | Reference check |
|---|---|---|
| `wilson_interval(k, n)` | The naive Wald (normal-approximation) interval is known to misbehave — including bounds outside `[0, 1]` — exactly where FlakeHunter operates: small `n`, proportions near 0 or 1. Wilson's score interval stays reliable there. | `wilson_interval(k=0, n=200)` upper bound ≈ `0.0188` |
| `rule_of_three_upper(n)` | The standard 95%-confidence upper bound on a failure rate after observing *zero* failures in `n` trials — lets the Verifier say "the true rate is below X%" instead of overclaiming "it never fails." | `rule_of_three_upper(200) == 0.015` |
| `n_for_detection(p, beta)` | Turns "run it a bunch of times" into an evidence-based sample size: how many isolated runs are needed so that, at true failure rate `p`, seeing zero failures by chance is at most `beta`. | `n_for_detection(p=0.05, beta=0.05) == 59` |
| `n_for_verification(threshold)` | The inverse of `rule_of_three_upper` — how many clean re-runs are needed post-fix to support a stated `threshold` claim. | `n_for_verification(0.02) == 150` |
| `fisher_exact_pvalue(k1, n1, k2, n2)` | An *exact* small-sample test, chosen over chi-squared/z-tests because the experiment sizes involved (`n=30` by default) and expected cell counts are too small for the chi-squared asymptotic approximation to be trustworthy. | `15/20` vs `0/20` → `p < 0.001`; `5/20` vs `5/20` → `p > 0.5` |
| `bh_fdr_correction(pvalues)` | The Classifier runs three Fisher tests per candidate cause; judging each against a raw `α=0.05` independently would inflate the overall false-discovery rate across those comparisons. Benjamini-Hochberg correction keeps the *set* of significant findings credible. | every corrected value `>=` its raw input; `[] -> []` |

## Setup

Requires Python ≥3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs runtime dependencies (`openai-codex`, `scipy`, `statsmodels`, `textual`) and the dev/test group (`mypy`, `pytest`, `pytest-forked`, `pytest-json-report`, `pytest-randomly`, `pytest-repeat`) from `pyproject.toml`/`uv.lock`, and registers the `flakehunter` console script (`flakehunter = "flakehunter.cli:main"`).

```bash
uv run pytest src/flakehunter/tests -q
```

runs the fast unit/end-to-end suite (milliseconds to a few minutes; no subprocess-heavy batteries). Tests marked `@pytest.mark.slow` (real repeated `SandboxRunner` batteries against the demo repo, minutes each) opt in with `-m slow`; tests marked `@pytest.mark.codex` (the real Codex API) opt in with `FLAKEHUNTER_CODEX_E2E=1`.

## Known limitations

Pulled directly from `docs/ARCHITECTURE.md` §5 — current, not hypothetical:

- **The randomness experiment is underpowered at the default `n_experiment_runs=30`, and structurally blind to near-always-failing RNG tests.** A test failing ~5% of the time needs `n≈200` for Fisher's test to separate seeded from unseeded runs; a test failing ~98% of the time can't be separated at *any* `n`, because a fixed seed makes it deterministically *fail*, not pass. In practice the AST randomness signal is what classifies both demo cases, and the experiment merely fails to contradict it.
- **The timing veto is deliberately asymmetric.** A genuinely shared-state-polluted test that also calls `time.sleep` gets downgraded to detect-only `timing` whenever it isn't corroborated by the forked experiment — which, on a fork-less platform (Windows, this environment included), it never can be. The trade is intentional: a missed auto-fix is a cost, a wrongly auto-fixed timing test is a correctness violation.
- **A seed argument that isn't a literal reads as unseeded.** `random.seed(SOME_CONSTANT)` with a module-level constant still emits a randomness signal, since resolving module-level constants isn't implemented. Deliberate direction to err in — over-flagging is safer than suppressing for a flake hunter.
- **Classifying multiple tests from the same suite redundantly re-runs the shared suite battery.** `_count_suite_failures`/`_count_forked_failures` are recomputed from scratch inside every `classify()` call even though several targets in `CLASSIFICATION_CASES` share the same suite; no caching exists yet.
- **Not a full-suite CI scanner, by design** — see "The problem" above; every contract takes an explicit `test_ids`, with no discovery mode for flakiness itself.
- **The forked shared-state experiment is unsupported on this development platform.** `pytest-forked` needs `os.fork`, unavailable on Windows; the capability probe correctly falls back to the direction-only signal rather than hanging, but the stronger of the two shared-state signals is never actually exercised here.
- **The Verifier's regression check trusts a single before/after suite run on each side.** A pre-existing flaky sibling that happens to flip pass→fail between those two specific runs — independent of the fix under test — reads as a false `rejected_breaks_suite`. Accepted per the SPEC's regression clause ("once"), not overlooked; callers are expected to scope `suite_test_ids` to tests that are themselves stable.

## Codex Session ID

`<PASTE_CODEX_SESSION_ID_HERE>`
