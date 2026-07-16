"""Statistical primitives used to detect, classify, and verify flaky tests."""

import math

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def wilson_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a binomial failure rate.

    The detector uses this interval to report uncertainty around observed
    pass/fail counts. Wilson's method remains reliable for the small samples
    and near-zero failure rates common when investigating flaky tests, unlike
    the naive Wald normal approximation.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    lower, upper = proportion_confint(
        count=k,
        nobs=n,
        alpha=1.0 - confidence,
        method="wilson",
    )
    return float(lower), float(upper)


def rule_of_three_upper(n: int) -> float:
    """Compute the rule-of-three upper failure-rate bound after zero failures.

    The verifier uses this bound to express what a clean sequence of reruns
    supports statistically: even with no observed failures, the true rate is
    bounded at approximately ``3 / n`` rather than assumed to be zero.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


def n_for_detection(p: float, beta: float = 0.05) -> int:
    """Return runs needed to detect a true failure rate with confidence ``1-beta``.

    The detector must run a test often enough that the chance of seeing no
    failures when its true flaky rate is ``p`` is at most ``beta``. This makes
    the decision to begin detection evidence-based rather than arbitrary.

    Raises:
        ValueError: If ``p`` is not strictly between zero and one.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    return math.ceil(math.log(beta) / math.log(1.0 - p))


def n_for_verification(threshold: float) -> int:
    """Return clean reruns needed to support a target post-fix failure rate.

    The verifier uses the rule of three after observing no failures, requiring
    enough reruns that its upper bound is no greater than ``threshold``. This
    turns a claim that a fix worked into a stated statistical guarantee.

    Raises:
        ValueError: If ``threshold`` is not strictly between zero and one.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    return math.ceil(3.0 / threshold)


def fisher_exact_pvalue(k1: int, n1: int, k2: int, n2: int) -> float:
    """Compute a two-tailed Fisher exact p-value for two failure proportions.

    The classifier uses this exact small-sample test to distinguish meaningful
    condition-dependent failure-rate changes from noise; the verifier reuses
    it to show that a before/after improvement is statistically significant.
    """
    _, pvalue = fisher_exact(
        [[k1, n1 - k1], [k2, n2 - k2]],
        alternative="two-sided",
    )
    return float(pvalue)


def bh_fdr_correction(pvalues: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg FDR-adjusted p-values in input order.

    Classification can evaluate several possible flaky-test causes at once.
    This correction controls the expected false-discovery rate across those
    comparisons, so reported root-cause signals remain credible collectively.
    """
    if not pvalues:
        return []

    _, corrected, _, _ = multipletests(pvalues, method="fdr_bh")
    return [float(value) for value in corrected]
