\# STATS\_SPEC.md — `src/flakehunter/stats.py`



\## Role in the system

This module is the statistical inference layer of FlakeHunter. It is consumed

by three components:

\- \*\*Detector\*\*: turns raw pass/fail counts into a confidence-quantified

&#x20; failure rate (Wilson interval), and determines how many runs are needed

&#x20; before detection starts (power analysis).

\- \*\*Classifier\*\*: uses a two-proportion hypothesis test (Fisher exact) to

&#x20; decide, with a p-value, whether an isolated-vs-suite (or forked-vs-not,

&#x20; seeded-vs-not) difference in failure rate is real or noise — this is how

&#x20; root-cause categories are diagnosed rigorously instead of heuristically.

\- \*\*Verifier\*\*: determines how many re-runs are needed to claim a target

&#x20; post-fix failure rate (power analysis), computes the final confidence

&#x20; bound shown in the report (rule of three), and re-uses the Fisher exact

&#x20; test to confirm the before/after improvement is statistically significant,

&#x20; not incidental.



Every function must be pure, deterministic given its inputs, and independently

testable against known reference values — this module has no side effects and

does not depend on `contracts.py`.



\## Required dependencies

```

uv add scipy statsmodels

```

\- `scipy.stats.fisher\_exact` — exact test for 2x2 contingency tables, correct

&#x20; for the small sample sizes used throughout this project (do not substitute

&#x20; a chi-squared approximation).

\- `statsmodels.stats.proportion.proportion\_confint(..., method="wilson")` —

&#x20; Wilson score interval.

\- `statsmodels.stats.multitest.multipletests(..., method="fdr\_bh")` —

&#x20; Benjamini-Hochberg FDR correction.



\## Functions to implement (exact signatures)



\### `wilson\_interval(k: int, n: int, confidence: float = 0.95) -> tuple\[float, float]`

Wilson score interval for a binomial proportion `k/n`. Must NOT use the naive

normal-approximation (Wald) interval — it is invalid near p=0/p=1 and for

small n, which are exactly the regimes this project operates in (a fixed test

should show failure\_rate near 0; demo-scale sample sizes are small). Raise

`ValueError` if `n <= 0`.



\### `rule\_of\_three\_upper(n: int) -> float`

Upper bound on the true failure rate when 0 failures are observed in `n`

runs: approximately `3/n` (the informal "rule of three", the k=0 special case

of the Wilson/Clopper-Pearson interval). Raise `ValueError` if `n <= 0`.



\### `n\_for\_detection(p: float, beta: float = 0.05) -> int`

Minimum number of runs to detect flakiness of true rate `p` with confidence

`1-beta`. Derivation: P(zero failures in n runs) = (1-p)^n; require this

≤ beta; solve n ≥ log(beta)/log(1-p); round up. Raise `ValueError` if `p` is

not strictly between 0 and 1.



\### `n\_for\_verification(threshold: float) -> int`

Minimum number of runs to claim (assuming 0 observed failures) that the

post-fix failure rate is ≤ `threshold`, via the rule of three solved for n:

`n ≥ 3/threshold`, rounded up. Raise `ValueError` if `threshold` is not

strictly between 0 and 1.



\### `fisher\_exact\_pvalue(k1: int, n1: int, k2: int, n2: int) -> float`

Two-tailed Fisher exact test p-value comparing two failure proportions, via

the 2x2 contingency table:

```

&#x20;           failures    successes

&#x20; group1       k1         n1-k1

&#x20; group2       k2         n2-k2

```

Used both for root-cause classification (isolated vs. in-suite, forked vs.

not, seeded vs. not) and for fix verification (before vs. after failure

counts).



\### `bh\_fdr\_correction(pvalues: list\[float]) -> list\[float]`

Benjamini-Hochberg FDR-corrected p-values, same length and order as the

input. Return `\[]` for empty input.



\## Required tests — `src/flakehunter/tests/test\_stats.py`

Each test asserts against a known reference value, not just "runs without

error":



1\. `wilson\_interval(k=0, n=200)` → lower ≈ 0.0, upper ≈ 0.0188 (±0.002).

2\. `rule\_of\_three\_upper(200)` == 0.015 (±1e-9); `rule\_of\_three\_upper(100)` ==

&#x20;  0.03 (±1e-9).

3\. `n\_for\_detection(p=0.05, beta=0.05)` == 59 (i.e. ceil(log(0.05)/log(0.95))).

4\. `n\_for\_verification(threshold=0.01)` == 300; `n\_for\_verification(threshold=0.02)`

&#x20;  == 150.

5\. `fisher\_exact\_pvalue(k1=15, n1=20, k2=0, n2=20)` < 0.001 (clear order-dependency

&#x20;  signal: high in-suite failure rate vs. zero isolated).

6\. `fisher\_exact\_pvalue(k1=5, n1=20, k2=5, n2=20)` > 0.5 (identical proportions,

&#x20;  should NOT be flagged significant).

7\. `bh\_fdr\_correction(\[0.001, 0.02, 0.04, 0.9])` has same length as input, and

&#x20;  every corrected value is >= its corresponding raw p-value (BH correction

&#x20;  never decreases a p-value).

8\. `wilson\_interval(k=0, n=0)` raises `ValueError`.



\## Non-negotiables

\- No implicit rounding or truncation that hides precision loss — return raw

&#x20; floats, let the caller decide display precision.

\- No global state, no I/O, no logging inside this module.

\- Full type hints; every public function has a docstring explaining not just

&#x20; \*what\* it computes but \*why\* (see role descriptions above) — this module

&#x20; will be read by technical reviewers evaluating the project.

