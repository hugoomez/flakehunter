import pytest

from flakehunter.stats import (
    bh_fdr_correction,
    fisher_exact_pvalue,
    n_for_detection,
    n_for_verification,
    rule_of_three_upper,
    wilson_interval,
)


def test_wilson_interval_for_zero_failures() -> None:
    lower, upper = wilson_interval(k=0, n=200)

    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(0.0188, abs=0.002)


def test_rule_of_three_upper_reference_values() -> None:
    assert rule_of_three_upper(200) == pytest.approx(0.015, abs=1e-9)
    assert rule_of_three_upper(100) == pytest.approx(0.03, abs=1e-9)


def test_n_for_detection_reference_value() -> None:
    assert n_for_detection(p=0.05, beta=0.05) == 59


def test_n_for_verification_reference_values() -> None:
    assert n_for_verification(threshold=0.01) == 300
    assert n_for_verification(threshold=0.02) == 150


def test_fisher_exact_detects_clear_difference() -> None:
    assert fisher_exact_pvalue(k1=15, n1=20, k2=0, n2=20) < 0.001


def test_fisher_exact_does_not_flag_identical_proportions() -> None:
    assert fisher_exact_pvalue(k1=5, n1=20, k2=5, n2=20) > 0.5


def test_bh_fdr_correction_is_not_smaller_than_raw_pvalues() -> None:
    pvalues = [0.001, 0.02, 0.04, 0.9]
    corrected = bh_fdr_correction(pvalues)

    assert len(corrected) == len(pvalues)
    assert all(adjusted >= raw for adjusted, raw in zip(corrected, pvalues))


def test_wilson_interval_rejects_zero_observations() -> None:
    with pytest.raises(ValueError):
        wilson_interval(k=0, n=0)
