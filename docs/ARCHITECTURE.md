# FlakeHunter — Architecture & Design Log

## 1. Problem framing

A flaky test is one whose outcome on an unchanged commit is not a fixed
function of the code — it is a Bernoulli trial with some unknown true
failure probability `p`. "I ran it 10 times and it passed" says almost
nothing about `p`: a test that fails 20% of the time passes ten times in a
row about 11% of the time by chance alone, so a handful of green runs is
consistent with a genuinely flaky test. Treating flakiness this way — as a
parameter to be *estimated*, with a confidence interval, rather than a
yes/no fact established by eyeballing a few runs — is the premise the whole
project is built on. `stats.py` exists so every claim FlakeHunter makes
("this test is flaky," "this fix reduced the failure rate," "this is the
root cause") is backed by an interval or a p-value instead of a run count.

FlakeHunter is deliberately scoped as a **targeted diagnostic instrument
for a test (or small set of tests) already suspected of being flaky**, not
a continuous, whole-suite CI scanner. Every component signature in
`contracts.py` takes an explicit `test_ids: list[str]` — there is no "scan
this repo and find the flaky ones" entry point. This framing is a direct
consequence of the statistics: detecting a test with a 5% failure rate at
95% confidence needs on the order of dozens of isolated re-runs
(`n_for_detection(0.05, beta=0.05) == 59`, see §2), and classifying its root
cause needs several more batches of suite-level and isolated runs on top of
that (`test_classifier.py` documents ~90 subprocess launches for a single
test at `n_experiment_runs=30`). That cost is affordable for a handful of
tests a developer or CI system has already flagged; it does not scale to
re-running every test in a large suite dozens of times each. FlakeHunter
therefore optimizes for depth on a small, named target rather than breadth
across an entire suite.

## 2. Statistical foundations (`stats.py`)

**`wilson_interval(k, n, confidence=0.95)`** computes a Wilson score
confidence interval for a binomial proportion, via
`statsmodels.stats.proportion.proportion_confint(method="wilson")`. Wilson's
interval is used instead of the naive Wald (normal-approximation) interval
because Wald is known to perform badly — including producing bounds outside
`[0, 1]` — exactly in the regime FlakeHunter operates in: small `n` and
proportions near 0 or 1 (a test that fails 1 time out of 30). `Detector`
attaches a Wilson `ci95_upper` to every `FlakeVerdict` (`detector.py:82`),
and `VerificationResult.ci95_upper_after` in the spec's contract is the same
computation applied post-fix. Verified in `test_stats.py` against a
reference value: `wilson_interval(k=0, n=200)` has an upper bound of
`≈0.0188`, and the function rejects `n <= 0`.

**`rule_of_three_upper(n)`** returns `3/n`, the standard approximation for
the upper 95% confidence bound on a failure rate after observing zero
failures in `n` trials (an even simpler special case of Wilson at `k=0`
for large `n`). This is what lets the verifier say something like "after
300 clean reruns, the true failure rate is below 1%" rather than "it never
failed." Verified in `test_stats.py`: `rule_of_three_upper(200) == 0.015`
and `rule_of_three_upper(100) == 0.03`.

**`n_for_detection(p, beta=0.05)`** answers "how many isolated runs are
needed so that, if the true failure rate is `p`, the chance of observing
zero failures by chance is at most `beta`?" via
`n ≥ log(beta) / log(1 - p)`. This turns "run it a bunch of times" into an
evidence-based sample size tied to the sensitivity FlakeHunter wants to
guarantee, rather than an arbitrary constant. Verified against the
reference value `n_for_detection(p=0.05, beta=0.05) == 59`.

**`n_for_verification(threshold)`** is the inverse of `rule_of_three_upper`:
`ceil(3 / threshold)`, the number of clean (zero-failure) reruns needed to
support a claim that the post-fix failure rate is below `threshold`. This is
what the Verification Contract's statistical clause is meant to consume when
it exists. Verified against reference values: `n_for_verification(0.01) ==
300` and `n_for_verification(0.02) == 150`.

**`fisher_exact_pvalue(k1, n1, k2, n2)`** runs a two-sided Fisher exact test
on the 2x2 table of failures/successes under two conditions, via
`scipy.stats.fisher_exact`. Fisher's exact test is used rather than a
chi-squared or z-test because the sample sizes involved (`n_experiment_runs`
is 30 by default) and expected cell counts are too small for the
chi-squared approximation to be trustworthy — Fisher's test is exact rather
than asymptotic. This is the core tool `Classifier` uses to decide whether
two experimental conditions (isolated vs. in-suite, forked vs. unforked,
seeded vs. unseeded) produced meaningfully different failure counts, and
`Verifier` (per the spec) is meant to reuse it to confirm a before/after
improvement is significant, not noise. Verified against reference values: a
clear difference (`k1=15/20` vs `k2=0/20`) yields `p < 0.001`; identical
proportions (`5/20` vs `5/20`) yield `p > 0.5`.

**`bh_fdr_correction(pvalues)`** applies the Benjamini-Hochberg
false-discovery-rate procedure via
`statsmodels.stats.multitest.multipletests(method="fdr_bh")`. Because
`Classifier` runs three separate Fisher tests per test under investigation
(order, shared-state, randomness), and a full diagnostic session may cover
several suspected tests at once, evaluating each p-value against a raw 0.05
threshold would inflate the overall false-positive rate across comparisons.
BH correction keeps the *set* of "significant" experimental findings
credible. Verified in `test_stats.py`: every corrected value is `>=` its raw
input, and an empty list returns `[]`.

## 3. Component architecture

### `SandboxRunner` (`runner.py`)

`SandboxRunner.run_once` (`runner.py:42`) always launches pytest in a
**fresh subprocess** (`subprocess.run`, `runner.py:58`) rather than running
tests in-process. This is necessary, not just cautious: order-dependency and
shared-mutable-state flakiness (categories A/B) are properties of process
state accumulated across test execution, so in-process execution would
either mask or artificially create exactly the effects FlakeHunter is
trying to isolate and measure. Every subprocess call carries an explicit
`timeout_s` (default 30s, `runner.py:28`) enforced by
`subprocess.run(..., timeout=self.timeout_s)`; a `TimeoutExpired` is caught
and turned into `RunResult(outcome="error", ...)` for every requested test
id rather than propagating, satisfying the non-negotiable "never hang
indefinitely" (`test_timeout_returns_error_promptly` in `test_runner.py`
confirms this returns in well under the 5-second sleep it's timing out on).

Results are parsed exclusively from **`pytest-json-report`** output
(`--json-report`, `runner.py:157`, parsed by `_load_report` /
`_results_from_report`), never from stdout text — the SPEC's "never parse
pytest's text output" non-negotiable. `_outcome`, `_duration_s`, and
`_error_repr` (`runner.py:265-310`) all read structured fields (`outcome`,
per-stage `duration`, `longrepr`/`crash.message`) out of the JSON report,
with explicit fallback paths when a stage is missing or the report itself
is malformed (`_collection_error_repr`, `_process_error_repr`), so a
malformed report or a collection-time failure still produces a `RunResult`
per requested test id instead of raising.

`run_once` takes independent `seed`, `forked`, and `randomize_order`
parameters and translates them into `pytest-randomly` and `pytest-forked`
flags plus environment variables in `_pytest_command` /
`_subprocess_env` (`runner.py:144-179`). This three-way independence is what
makes it possible to run controlled differential experiments (vary exactly
one axis, hold the others fixed) — see §4 for why this decoupling had to be
fixed explicitly.

### `Detector` (`detector.py`)

`Detector.detect` (`detector.py:30`) runs each requested test id
`n_runs` times, drawing a set of distinct pseudo-random seeds up front from
a `random.Random(self.batch_seed)` seeded RNG (`_distinct_seeds`,
`detector.py:60`) so a detection run is reproducible given the same
`batch_seed`. Each run is isolated (`forked=False`, single test id per
`run_once` call) and optionally varies order and/or seed per the caller's
flags. `_verdict` (`detector.py:71`) turns the resulting `RunResult` list
into a `FlakeVerdict`: `n_failures` counts `"failed"` and `"error"`
outcomes together, `failure_rate` is the raw proportion, `ci95_upper` comes
from `wilson_interval` (§2), and `is_flaky` is exactly `0 < n_failures <
n_runs` — a test that always fails is a deterministic bug, not flakiness,
and this predicate is what keeps the two apart. `_sample_tracebacks`
(`detector.py:94`) caps stored tracebacks at 3 distinct `error_repr`
values, so a verdict carries evidence without unbounded growth.
`test_detector.py` exercises this against the real demo repo (not mocks):
30 runs each of the three flaky demo cases plus the two deterministic
distractors, asserting `is_flaky` and failure-rate-in-range for the flaky
cases and `is_flaky is False` with exact `0/30` or `30/30` counts for the
distractors — i.e., a real check for both sensitivity and the absence of
false positives.

### `Classifier` (`classifier.py`)

`Classifier.classify` (`classifier.py:133`) takes a `FlakeVerdict` and
determines *why* the test is flaky by running three 2x2 differential
experiments, each compared with `fisher_exact_pvalue` (§2), plus a static
AST scan scoped to the test:

1. **Order** — failures isolated (`k_iso`) vs. failures inside the
   full randomized-order suite (`k_suite`) (`_count_isolated_failures`,
   `_count_suite_failures`).
2. **Shared state** — failures in-suite forked (`k_forked`, via
   `--forked`/process isolation) vs. unforked (`k_suite`)
   (`_count_forked_failures`, `classifier.py:323`).
3. **Randomness** — failures running the target alone under a fixed seed
   (`k_seeded`, `_FIXED_SEED = 42`, `randomize_order=False`) vs. running it
   alone with `seed=None` so `pytest-randomly` draws its own fresh seed per
   repetition (`k_iso`, `randomize_order=True`). With a single test id there
   is nothing for order randomization to reorder, so the seed is the only
   *effective* variable between the two conditions.

`k_iso` is sampled once and serves two comparisons, each varying exactly one
factor against it: the suite context (vs. `k_suite`, which also runs unseeded
with order randomization on) and the seed (vs. `k_seeded`). Sharing the
baseline this way keeps the subprocess budget at four batches while making
the order experiment *cleaner* than a design that held order fixed on the
isolated side only — that would have confounded seeding with context.

Because both order-dependency and shared-state pollution make experiment 1
significant, the *direction* of the difference is what separates them
(`classifier.py:20-42`, module docstring): a test polluted by shared state fails
*more* in-suite than isolated (`pollution`, `k_suite > k_iso`); a test that
legitimately depends on setup performed by an earlier test fails *more*
isolated than in-suite (`missing_setup`, `k_iso > k_suite`). The forked
experiment is treated as the stronger, primary signal for shared state when
`os.fork` is available, since process isolation directly removes shared-state
pollution; `_count_forked_failures` probes once and short-circuits with
`forked_supported=False` if the platform can't produce real pytest outcomes
under `--forked` (this is the normal, expected path on Windows, since
`pytest-forked` needs `os.fork`), falling back to the direction-only signal
from experiment 1.

**The timing veto** (`classifier.py:268`) guards the weakness in that
direction-only reasoning. A directional difference is *weak* evidence,
because a load-sensitive race reproduces the same pattern for an unrelated
reason: the suite keeps the machine busy, so a test with a fixed `time.sleep`
budget fails more often in-suite than alone — indistinguishable from
pollution if direction is all you look at. So `classify()` tracks a
`direction_only` flag (`classifier.py:196`) alongside the verdict. A
`shared_state` verdict is *corroborated* only when the forked experiment and
the in-suite direction independently agree; `order` is always direction-only,
since no second experiment speaks to it. When a `shared_state`/`order`
verdict is direction-only *and* the target carries a timing AST signal, it is
downgraded to `timing` with `auto_fixable=False`, and the evidence records
why. This makes the SPEC's "category D is detect-only" a property the code
enforces rather than one the priority order happens to preserve.

`_ast_signals` (`classifier.py:387`) parses the test's source with the
standard library `ast` module. Two properties make it trustworthy — it
recognizes calls however they were imported, and it attributes them to the
right test:

* **Import aliases are resolved.** `_Symbols` (`classifier.py:454`) walks
  `Import`/`ImportFrom` into a map from locally bound name to canonical
  dotted origin, so `rnd.choice` (`import random as rnd`), a bare `sleep()`
  (`from time import sleep`), and `np.random.rand` all canonicalize before
  matching. Matching the literal text at the call site — the previous
  behavior — silently skipped every aliased or from-imported call.
* **The scan is scoped to the target test.** `_scope_nodes`
  (`classifier.py:582`) collects the test's own `def`, the module-level
  statements that run at import (constants, class-level assignments), and the
  fixtures the test actually requests by name or that are `autouse=True`.
  Scanning the whole file let one test's `random.seed(...)` suppress the
  randomness signal for every *other* test in the same file. Aliases are
  still resolved module-wide, since an import binds a module-level name
  regardless of which test uses it.

It emits `time.sleep`/`asyncio.sleep`/`datetime.now` calls (timing), unseeded
`random.*`/`numpy.random.*` calls (randomness), seeded RNG constructs
(`seeded`, which suppresses the randomness signal in scope), and
`requests.*`/`socket.*`/`urllib.*`/`httpx.*` calls or `open()` on a fixed
string path (external). A construct counts as seeded only when given a
*literal* argument (`_has_literal_seed`, `classifier.py:506`), so
`random.Random(42)` is reproducible but a bare `random.seed()` — which
reseeds from system entropy — correctly is not.

AST signals serve three roles: input to the timing veto above; a fallback
when no experiment is significant (see §4, Finding 1's aftermath); and
supplementary evidence appended to every verdict regardless. `auto_fixable`
is `True` only for `order`/`shared_state`/`randomness`, matching the SPEC's
taxonomy.

### `Verifier` (`verifier.py`)

`Verifier.verify` (`verifier.py:190`) enforces the SPEC's **Verification
Contract** against an already-generated `FixProposal`: a fix is
`verified_fix` only if all three stages pass, in a fixed fail-fast order,
each cheaper than the next.

1. **Structural** (`_structural_check`, `verifier.py:381`) — an AST-only
   check, no subprocess. Assertion count in the target test's own scope may
   not decrease (`_count_assertions`, matching both bare `assert` and
   `unittest`/`pytest` assertion-equivalent calls by their final attribute
   name), and no `skip`/`xfail`/`flaky`/rerun decorator, module-level
   `pytestmark`, or `try/except` wrapping an `assert` may be newly
   introduced (`_masking_markers`, `_masking_try_count`). A legitimate
   `@pytest.fixture(autouse=True)` added by a real fix is explicitly not
   masking (`test_autouse_cleanup_fixture_is_not_a_weakening`). Failing this
   stage returns `rejected_weakens_test` immediately — no runner is ever
   constructed, which `test_*_is_rejected_before_any_run`'s `_ExplodingRunner`
   (a factory that raises if `run_once` is ever called) proves directly
   rather than by inference from timing.
2. **Statistical** — re-runs the whole `suite_test_ids` batch (unseeded,
   randomized order) `n_for_verification(threshold)` times, watching the
   target, and gates on both a confidence bound and a significance test (see
   "the Wilson-vs-rule-of-three gate" and "in-suite re-run" below).
3. **Regression** (`_suite_still_green`, `verifier.py:311`) — runs the
   pristine (pre-fix) and post-fix trees once each and requires every test
   that passed pristine (other than the target itself) to still read
   `"passed"` after. Missing from the post-fix report at all (crash, timeout,
   newly skipped) counts as a regression, not a pass, since `.get(tid) ==
   "passed"` is the only accepted value.

Each stage short-circuits on failure with its own verdict
(`rejected_weakens_test` / `rejected_still_flaky` / `rejected_breaks_suite`),
so a `VerificationResult` always states *which* clause of the contract
failed rather than a bare pass/fail.

**The Wilson-vs-rule-of-three gate.** The statistical gate does not simply
threshold the stored Wilson `ci95_upper_after`. `N_verify =
n_for_verification(threshold)` is constructed as the inverse of
`rule_of_three_upper` (§2), so `rule_of_three_upper(N_verify) <= threshold`
holds exactly by that construction — but the Wilson upper bound for the same
zero-failure outcome is slightly *higher* (e.g. `0.0249` vs. `0.02` at
`threshold=0.02`, `N=150`). Gating the reported Wilson bound directly would
therefore reject a fix that produced a perfect `0/150` re-run — the exact
outcome the verifier exists to accept. `verify()` (`verifier.py:240-244`)
resolves this by gating `k_after == 0` on `rule_of_three_upper(n_verify)` and
falling back to the (necessarily present, since `k_after > 0`) Wilson bound
otherwise, while `VerificationResult.ci95_upper_after` always reports the
honest Wilson value regardless of which bound gated. This means a genuinely
verified fix's *reported* CI can read a hair above `threshold` — documented
in the module docstring as intentional, and pinned by
`test_genuine_seed_fix_is_verified`'s exact assertion of
`wilson_interval(k=0, n=150)` against the reported value.

**The in-suite re-run methodology.** Unlike `Detector`, which isolates the
target test for its re-run battery, both the statistical and regression
stages of `verify()` re-run the *entire* `suite_test_ids` batch and read the
target's outcome out of it (`_count_target_failures`,
`_suite_still_green`). This is necessary because `verify()` receives no
root-cause category from `Classifier` — an order-dependency or shared-state
flake only reproduces when run alongside the rest of the suite, so isolating
the target would report a false zero-failure "pass" regardless of whether
the fix did anything. Running in-suite covers randomness flakes too (they
still fail when run as part of a larger batch), so one re-run protocol
serves every SPEC category the verifier has to handle. The cost is real —
each of the `n_for_verification(threshold)` statistical iterations launches
one suite-wide subprocess rather than one single-test subprocess — but
correctness of the measurement takes priority over its cost, matching
`Classifier._count_suite_failures`'s same choice (§3).

**Why this module never imports the Fixer or Codex.** `verify(fix, before)`
consumes an already-generated `FixProposal`; it has no reason to construct
one, so it deliberately never imports `fixer.py`. This is enforced, not just
avoided by convention: `fixer.py` pulls in the Codex SDK at module load time,
so importing anything from it — even an unrelated helper — would drag that
dependency into every verifier import and every verifier test. The small
git-tracked-tree-copy and `git apply` helpers at the bottom of `verifier.py`
(`_copy_tracked_tree`, `_git_apply`) are therefore reimplemented locally
rather than shared with `fixer.py`, which needs the same operations for the
same reason (SPEC repo layout note in the module docstring: `verifier.py:20-29`).
The whole test suite proves this holds in practice rather than just in
principle: every `FixProposal` in `test_verifier.py` is hand-built with
`codex_session_id="hand-built-fixture"` (`test_verifier.py:84-93`), so all 16
tests — including the three "reality" tests that apply real diffs to the
real demo repo — run against the actual `SandboxRunner` with no Codex
account, API key, or network access required.

## 4. The Classifier audit and remediation (case study)

Before `classifier.py` was accepted, an independent audit (a second agent,
not the one that wrote the module) reviewed it against `stats.py` and the
SPEC's category definitions, specifically looking for statistically invalid
experiments. It found two real issues.

**Finding 1 — the randomness differential experiment was statistically
void.** The classifier's randomness experiment compares failures with a
"fixed seed" condition against an "unseeded" condition. The audit pointed
out that `PYTHONHASHSEED` does not seed Python's `random` module — it only
affects hash randomization (`str`/`bytes` hash order) — and that the runner
only passed `--randomly-seed` (the flag that actually seeds `random` via
`pytest-randomly`) when `randomize_order` was also `True`. Since the
randomness experiment always runs with `randomize_order=False` (order must
be held fixed to isolate the seeding variable), the "fixed seed" condition
was never actually receiving a seed — it was statistically identical to the
"unseeded" condition. The audit demonstrated this empirically: five runs
under the nominally "fixed" condition produced five different RNG draws,
i.e. zero reproducibility, when a genuinely fixed seed should have produced
five identical draws.

**Finding 2 — the priority-based decision chain could misclassify a
load-sensitive timing test as auto-fixable.** Because experiment 1's
direction-only signal (`pollution`, `k_suite > k_iso`) is checked before any
AST signal in the decision chain, a timing-dependent test (SPEC category D,
detect-only) that happens to fail more often when run as part of the full
suite — plausible for a test with a fixed `time.sleep` budget under
suite-level load — could trip the `pollution` branch and be classified
`shared_state`, which is auto-fixable. This would violate the SPEC's
requirement that timing dependencies are surfaced for review, never
auto-fixed.

**What was changed for Finding 1:** `runner.py`'s `_pytest_command`
(`runner.py:144`) and `_subprocess_env` (`runner.py:174`) were restructured
so RNG seeding is fully decoupled from order randomization: `seed` is
passed to `--randomly-seed` and `PYTHONHASHSEED` whenever a seed is given,
*regardless* of `randomize_order`. To keep collection order fixed without
disabling the seed, the runner uses `--randomly-dont-reorganize` (which
keeps the `pytest-randomly` plugin active, so `--randomly-seed` still takes
effect) instead of `-p no:randomly` (which would disable the plugin
entirely and silently drop the seed) — see the comment at
`runner.py:166-169`. This is exercised directly in `test_runner.py`:
`test_seed_is_passed_when_order_is_not_randomized` asserts
`--randomly-seed=4242` and `--randomly-dont-reorganize` both appear on the
command line with `no:randomly` absent, and
`test_same_seed_reproduces_rng_without_randomizing_order` /
`test_differing_seeds_produce_differing_rng_without_randomizing_order` run
an actual probe test twice with fixed order and assert that the same seed
reproduces an identical `random.random()` draw while different seeds
produce different draws — closing the exact gap the audit demonstrated.

**What was changed on the classifier side for Finding 1:** decoupling the
runner flags was necessary but not sufficient — the classifier still had to
*ask* for the two conditions. `classify()` now samples them explicitly:
condition A pins `seed=42`, condition B passes `seed=None` so
`pytest-randomly` draws a fresh seed per repetition, and both run the target
alone (§3). The mechanism is proven by a stub-runner unit test
(`test_randomness_experiment_samples_fixed_seed_against_per_run_seed`), which
asserts the recorded calls rather than the verdict: 30 calls at `seed=42`,
30 at `seed=None`, all with `test_ids == [target]`.

**Finding 1's aftermath — the demo cases cannot exercise the fixed
experiment.** A genuinely-seeded condition turned out *still* not to separate
either demo randomness case, for opposite reasons that are worth recording,
because they are properties of the tests rather than of the code:

* `test_shuffle_preserves_first` fails ~98% of the time unseeded. Seed 42
  makes `random.shuffle` deterministic — and puts a card other than 0 first —
  so it fails **30/30** seeded vs. ~29/30 unseeded. A fixed seed makes it
  deterministically *fail*, not pass; the direction the experiment looks for
  (`k_seeded < k_iso`) is unreachable. Only ~1 seed in 52 would flip it, and
  choosing that seed would be overfitting to the demo.
* `test_round_trip` fails ~5% of the time. Seed 42 makes it pass, so the
  comparison is 0/30 vs. ~1.5/30. Fisher needs ~6/30 to reach `p<0.05`
  (`0/30` vs `4/30` is `p=0.11`; vs `6/30` is `p=0.024`), which occurs ~6% of
  the time at the true rate. Separating it reliably needs `n≈200`.

So the AST randomness signal is what carries both — not because of the runner
bug, but because there is no experimental signature to find. This is why the
end-to-end test asserts the AST signal backs a `randomness` verdict while the
*experiment firing* is covered by stub-runner unit tests that can stage
counts the demo cannot produce. Asserting the experiment fires end-to-end
would have made FlakeHunter's own suite flaky ~94% of the time — a fitting
hazard for this project to have caught in review.

**What was changed for Finding 2:** `classify()` now tracks whether a verdict
rests on a directional difference alone and vetoes it against the timing AST
signal before finalizing — see "the timing veto" in §3 for the mechanism.
`test_timing_signal_vetoes_direction_only_shared_state` and
`test_timing_signal_vetoes_direction_only_order` cover the downgrade,
`test_corroborated_shared_state_survives_the_timing_veto` covers the
forked+direction case that must *not* downgrade, and
`test_pollution_direction_alone_yields_shared_state` pins the baseline so the
veto can't silently swallow every shared-state verdict. All four use a
scripted stub runner, so they stage the exact 2x2 counts in milliseconds
rather than hoping the demo reproduces them.

This episode is worth stating plainly as evidence of the project's rigor,
not a flaw to downplay: a real statistical bug — one that would have caused
the classifier to report a spurious, high-confidence root cause based on an
experiment that could never have shown a difference — was caught by an
independent audit before it reached the (not-yet-built) verification stage.
That is precisely the verify-don't-trust principle the SPEC's `Verifier`
component is meant to apply to Codex-generated fixes: FlakeHunter's own
statistically load-bearing code was held to the same standard before this
project trusted it.

## 5. Known limitations (current, not hypothetical)

- **The randomness experiment is underpowered at `n_experiment_runs=30`, and
  is structurally blind to near-always-failing RNG tests.** As quantified in
  §4, a test failing ~5% of the time needs `n≈200` for Fisher to separate
  seeded from unseeded, and a test failing ~98% of the time cannot be
  separated at *any* `n`, because a fixed seed makes it deterministically fail
  rather than pass. Both demo randomness cases fall in these gaps, so in
  practice the AST signal is what classifies them and the experiment only
  fails to contradict it. A "does the seed *determine* the outcome" design
  (comparing outcome variance across several fixed seeds) would catch the
  98% case in principle, but not for these demo tests specifically: ~51 of 52
  seeds fail, so nearly every seed pair agrees. This is a real sensitivity
  limit, not a bug — the AST fallback is what covers it.
- **The timing veto is deliberately asymmetric and will produce false
  `timing` verdicts.** A genuinely shared-state-polluted test that also calls
  `time.sleep` is downgraded to detect-only `timing` whenever it is not
  corroborated by the forked experiment — which, on Windows, it never can be
  (see the fork limitation below). The trade is intentional: the SPEC makes
  category D detect-only, so a missed auto-fix is a cost and a wrongly
  auto-fixed timing test is a correctness violation. On a fork-capable
  platform the corroboration path recovers these cases.
- **A seed argument that is not a literal reads as unseeded.**
  `_has_literal_seed` (`classifier.py:506`) requires a literal constant, so
  `random.seed(SEED_CONST)` with a module-level constant emits a randomness
  signal. This is the deliberate direction to err in (over-flagging randomness
  is safer than suppressing it for a flake hunter, and the signal is a
  hypothesis rather than a verdict), but resolving module-level constants
  would remove the false positive.
- **Classifying multiple tests from the same suite redundantly re-runs the
  shared suite battery.** `_count_suite_failures` and
  `_count_forked_failures` are recomputed from scratch inside every
  `classify()` call, even though they execute the same `suite_test_ids`
  batch regardless of which single test id is under investigation.
  `test_classifier.py`'s module docstring documents that a single case at
  `n_experiment_runs=30` launches on the order of 90 pytest subprocesses;
  `CLASSIFICATION_CASES` covers 5 test ids drawn from one shared suite, so
  the current implementation pays that suite-battery cost independently for
  each of the 5 cases instead of computing it once and reusing it across
  all targets in the same suite. No caching exists yet.
- **Not a full-suite CI scanner, by design.** Every contract in
  `contracts.py` (`Detector.detect`, `Classifier.classify`) operates on an
  explicit, caller-supplied `test_ids`/`FlakeVerdict` — there is no
  discovery mode that walks an entire test suite looking for flakiness. See
  §1 for why (the per-test run cost does not scale to a full suite).
  `Fixer`, `Verifier`, `tui.py`, and the real `cli.py` command surface
  (`flakehunter run|detect|classify|reproduce`) named in the SPEC's repo
  layout are not implemented yet; `cli.py` is currently a placeholder
  (`print("flakehunter placeholder")`).
- **The forked shared-state experiment is unsupported on this development
  platform.** `pytest-forked` requires `os.fork`, unavailable on Windows.
  `_count_forked_failures`'s capability probe (`classifier.py:323-351`)
  correctly falls back to the direction-only signal from experiment 1
  (`_shared_experiment_evidence`, `classifier.py:367`) rather than hanging
  or crashing, but this means the stronger of the two shared-state signals
  described in §3 is not actually exercised in this environment — and, per
  the veto limitation above, no `shared_state` verdict reached here can ever
  be corroborated. The stub-runner unit tests cover the fork-capable paths
  (`fork_supported=True`) that this platform cannot execute for real.
- **The Verifier's regression check trusts a single before/after suite run
  on each side.** `_suite_still_green` (`verifier.py:311`) compares one
  pristine-tree run against one post-fix-tree run; a *pre-existing* flaky
  sibling elsewhere in `suite_test_ids` that happens to flip pass→fail
  between those two specific runs — independent of the fix under test —
  reads as `rejected_breaks_suite`. The module docstring
  (`verifier.py:52-60`) records this as accepted, not overlooked: the SPEC's
  regression clause says "once," and callers are expected to scope
  `suite_test_ids` to tests that are themselves stable rather than the
  verifier re-confirming a candidate regression with repeat runs.
- **`collect_suite()`'s auto-discovered ids can be mismatched with what the
  runner expects, when the target repo has no pytest config of its own.**
  `collect_suite` (`orchestrator.py:124`) shells out to
  `pytest --collect-only` with `cwd=repo_root`; pytest's own rootdir search
  then climbs *past* `repo_root` looking for a config file, and if an outer
  directory has a `pyproject.toml`/`setup.cfg`/`tox.ini` (as `demo/`'s parent
  does), the reported node ids come back relative to that outer rootdir —
  one or more path segments longer than the `repo_root`-relative ids
  `SandboxRunner` expects when it runs pytest with `cwd=repo_root`. A target
  `test_id` the user passed in `repo_root`-relative form then will not
  appear in `suite_test_ids`, and `Verifier.verify` raises rather than
  silently guessing. Workaround: pass `--suite-file` with ids already in the
  `repo_root`-relative form (see `demo/suite_ids.txt`), bypassing
  `collect_suite` entirely. Not fixed this session, to avoid a late change
  during feature freeze.

## 6. Testing philosophy

`stats.py` is tested with **reference-value assertions**, not
runs-without-error smoke tests: `test_n_for_detection_reference_value`
asserts the exact integer `n_for_detection(p=0.05, beta=0.05) == 59`,
`test_n_for_verification_reference_values` asserts `n_for_verification(0.01)
== 300` and `n_for_verification(0.02) == 150`, and
`test_rule_of_three_upper_reference_values` checks both `200` and `100`
against their known closed-form answers. This matters because these
functions gate detection sensitivity and verification sample sizes
directly — a silently wrong constant would silently under-power every
downstream detection or verification decision.

`Detector` and `Classifier` are tested against the **ground-truth-labeled
demo repo** (`demo/GROUND_TRUTH.json`, per `docs/DEMO_CASES.md`) rather than
synthetic mocks. `test_detector.py` runs the real demo suite and checks both
that the three genuinely flaky cases are flagged `is_flaky` with
failure rates inside their labeled expected ranges, *and* that the two
deliberately deterministic distractors (`test_broken_feature.py`, always
fails; `test_stable_total.py`, always passes) are never flagged flaky —
an explicit false-positive check, not just a sensitivity check.
`test_classifier.py` does the same for root-cause category:
`test_ground_truth_matches_expected_cases` guards that the categories
asserted in the test file itself agree with `GROUND_TRUTH.json` (so the two
can't silently drift apart), and
`test_classifier_assigns_ground_truth_category` runs the real classifier
against each labeled demo case end-to-end, asserting the assigned category,
`auto_fixable` flag, and that a `randomness` verdict is specifically backed
by the AST randomness signal (not just any evidence).

**Ground-truth tests alone are not enough, and `test_classifier.py` now says
so structurally.** The demo repo can only exercise the paths its five labeled
cases happen to reach; it cannot stage a 2x2 table on demand, cannot produce
a corroborated `shared_state` verdict on a fork-less platform, and — per §4 —
cannot make the randomness experiment fire at all. So the module pairs its
end-to-end cases with **stub-runner unit tests**: `_StubRunner` scripts
failure counts per experimental condition and records every `run_once` call,
letting the decision logic, the timing veto, and the sampling *protocol*
itself be asserted directly, in ~1s and with no subprocesses. The division is
deliberate — the demo tests answer "does this work on real flaky tests," the
stub tests answer "is the reasoning correct," and neither question subsumes
the other. The AST scanner is likewise tested by calling `scan_ast` on parsed
source literals (aliases, `from` imports, `random.Random(42)`,
`asyncio.sleep`, and a three-test file asserting signals don't leak between
functions in either direction), since these are pure static-analysis
properties with no reason to pay for a subprocess.

The most statistically complex module, `Classifier`, was additionally
subjected to an **independent second-agent audit** rather than relying on
its own test suite alone — see §4. This caught a class of bug (a
differential experiment that looked reasonable but could never actually
show a difference, because of an unrelated flag-coupling bug in the
runner) that would have been easy for the module's own author, and its own
tests, to miss, since the tests would have needed to specifically check RNG
reproducibility across seeds rather than just checking that classification
output matched ground truth.

The audit's *remedy* then got the same treatment, which is the part worth
keeping. It prescribed asserting that the repaired randomness experiment
fires on the two demo cases. Checking that claim against the actual demo
tests before implementing it — rather than implementing it and watching the
suite go red — showed it was unreachable for one case and ~6%-reliable for
the other (§4). An audit finding is a hypothesis about the code, not a fact
about it; the same verify-don't-trust standard the SPEC aims at
Codex-generated fixes applies to review feedback, including feedback that is
mostly right. Findings 1 and 2 were both real and are both fixed; the
prescribed test for Finding 1 was not, and adopting it verbatim would have
made FlakeHunter's own suite flaky.

`test_verifier.py` applies the same fast/slow split as `test_classifier.py`,
with one addition specific to a component whose job is to *reject* things:
the fast tests prove the structural fail-fast path never reaches a
subprocess at all, not just that it returns the right verdict. Each
`test_*_is_rejected_before_any_run` injects an `_ExplodingRunner` factory
that raises `AssertionError` if `run_once` is ever called, so
`rejected_weakens_test` is verified to short-circuit by construction rather
than by the absence of a slow-test timing signal. The three
`@pytest.mark.slow` "reality" tests apply hand-built unified diffs (the same
shape `Fixer` emits, via `difflib.unified_diff`) to the real demo repo and
run the genuine `SandboxRunner`, deliberately choosing demo cases whose
statistics make each outcome deterministic rather than probable: a literal
`random.seed(0)` insertion drives `test_round_trip` to exactly `0/150` for
`verified_fix`; the ~98%-failing deck test with a no-op line inserted stays
almost-certainly flaky for `rejected_still_flaky`; and a diff that
legitimately seeds `test_round_trip` while also breaking
`shopcart.cart.Cart.total` (a stable, never-flaky test) demonstrates
`rejected_breaks_suite` without depending on any test's flake rate to
cooperate. No test in the module constructs a `FixProposal` any other way
than by hand — see §3's Verifier subsection for why that is load-bearing,
not incidental.

## Development Log

### 2026-07-16

Read `docs/SPEC.md`, `docs/DEMO_CASES.md`, `contracts.py`, `stats.py` (+
tests), `runner.py` (+ tests), `detector.py` (+ tests), `classifier.py` (+
tests), and `cli.py`, and wrote this document from scratch. Documented the
statistical rationale for each `stats.py` function against its reference-value
tests; the actual implementation decisions in `SandboxRunner`, `Detector`,
and `Classifier`; and the independent audit of `Classifier`'s differential
experiments as a case study (§4). Verified against the code, not just the
audit's own account, that Finding 1 (seed/order decoupling in `runner.py`)
is fixed and tested, while Finding 2 (timing-AST downgrade in the
classifier's decision chain) is not yet implemented — recorded as an open
item in §5 rather than a closed one. Also noted, from direct code reading,
two limitations not previously called out anywhere: AST signal detection
does not resolve import aliases/`from`-imports, and classifying multiple
tests from one suite redundantly re-runs the shared suite battery once per
target instead of once per suite.

Later the same day, closed the two classifier gaps this document had recorded
as open. `classify()` now samples the randomness experiment's two conditions
explicitly (fixed `seed=42` vs. `seed=None`, both on the isolated target),
and reuses the unseeded isolated baseline for the order experiment so that
comparison varies only the suite context. Added the timing veto: a
`shared_state`/`order` verdict resting on a directional difference alone is
downgraded to detect-only `timing` when the target carries a timing AST
signal, closing Finding 2. Reworked the AST scanner to resolve import aliases
through a symbol map, scope the scan to the target test's own node plus the
module-level code and fixtures that run before it, treat
`random.Random(<literal>)` as seeded, and recognize `asyncio.sleep`.
Rewrote §3, §4, §5 and §6 accordingly, and removed the two limitations this
work closed.

The prescribed test change did **not** survive review. The audit asked for an
assertion that the repaired randomness experiment fires on `test_round_trip`
and `test_shuffle_preserves_first`; checking the demo tests' actual
statistics first showed seed 42 makes `test_shuffle_preserves_first` fail
30/30 (unreachable direction) and gives `test_round_trip` a 0/30-vs-~1.5/30
comparison that clears `p<0.05` only ~6% of the time. Kept the end-to-end
tests asserting what is true — the AST signal carries these two — and proved
the experiment's mechanism instead with stub-runner unit tests that record
the sampled conditions and can stage counts the demo cannot produce. Recorded
the underlying sensitivity limit as a new §5 limitation, along with the timing
veto's deliberate asymmetry and the non-literal-seed false positive. Verified:
20 fast classifier unit tests (1.2s), 5 end-to-end demo cases (4m46s, all
matching ground truth, including `test_worker_completes` now reaching `timing`
through the veto), the other 19 tests in the suite, and mypy clean.

### 2026-07-18

Read `docs/ARCHITECTURE.md` and `verifier.py` (+ `test_verifier.py`) and
documented the `Verifier` component: a new §3 subsection covering the
three-stage Verification Contract (structural AST check, in-suite
statistical re-run, single-run regression check), each stage short-circuiting
to its own rejection verdict; the Wilson-vs-rule-of-three gate (`N_verify`
gates a zero-failure re-run on `rule_of_three_upper`, not the slightly higher
stored Wilson bound, so a genuine `0/N` fix isn't rejected by its own honestly-
reported confidence interval); and the in-suite (not isolated) re-run
methodology, necessary because `verify()` gets no root-cause category from
`Classifier` and an order/shared-state flake would falsely read as fixed in
isolation. Added a §5 limitation for the regression check's single-run
before/after comparison (a coincidentally-flipping unrelated sibling can
produce a false `rejected_breaks_suite`) and a §6 paragraph on
`test_verifier.py`'s testing approach.

This session deliberately left the Verifier **not wired to `Fixer` or
Codex** — `verify()` takes an already-built `FixProposal` and has no reason
to import either, and `fixer.py` pulls in the Codex SDK at load time, so the
verifier reimplements its own tree-copy/`git apply` helpers rather than
sharing them. Confirmed by reading, not just asserting: every `FixProposal`
in `test_verifier.py` is hand-built (`codex_session_id="hand-built-fixture"`),
and `verifier.py` has no import of `fixer` anywhere. This keeps the module
independently testable and reviewable — the whole battery below runs with no
Codex account, API key, or network access.

Verified: the full `test_verifier.py` battery is 16/16 passing — 13 fast
tests (structural fail-fast via an exploding runner factory, plus pure-
function unit tests of the assertion counter and masking-marker detector) in
under a second, and 3 `@pytest.mark.slow` reality tests that apply hand-built
diffs to the real demo repo and drive the actual `SandboxRunner` through all
three rejection paths plus `verified_fix` — `test_genuine_seed_fix_is_verified`,
`test_noop_diff_stays_flaky_is_rejected`, and
`test_fix_that_breaks_a_sibling_is_rejected` — in 4m39s total for the real
subprocess battery.
