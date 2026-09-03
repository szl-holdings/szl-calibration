import pytest

from szl_calibration import metrics as M


def test_brier_perfect():
    assert M.brier_score([0.0, 1.0], [0, 1]) == 0.0


def test_brier_worst():
    assert M.brier_score([1.0, 0.0], [0, 1]) == 1.0


def test_log_loss_perfect():
    assert M.log_loss([1e-15, 1.0 - 1e-15], [0, 1]) < 1e-12


def test_ece_perfect_calibration():
    probs = [0.5] * 100
    labels = [i % 2 for i in range(100)]
    assert M.expected_calibration_error(probs, labels, 10) == pytest.approx(0.0)


def test_ece_overconfident():
    probs = [0.9] * 100
    labels = [0] * 100
    assert M.expected_calibration_error(probs, labels, 10) == pytest.approx(0.9)


def test_mce_picks_worst_bin():
    probs = [0.1, 0.1, 0.9, 0.9]
    labels = [1, 1, 0, 0]
    assert M.maximum_calibration_error(probs, labels, 10) == pytest.approx(0.9)


def test_bins_skip_empty():
    bins = M.reliability_bins([0.05, 0.95], [0, 1], 10)
    assert len(bins) == 2


def test_auroc_perfect_and_reversed():
    assert M.auroc([0.1, 0.9], [0, 1]) == 1.0
    assert M.auroc([0.9, 0.1], [0, 1]) == 0.0


def test_auroc_ties_half():
    assert M.auroc([0.5, 0.5], [0, 1]) == 0.5


def test_validate_rejects_nan_and_empty():
    with pytest.raises(ValueError):
        M.brier_score([float("nan")], [1])
    with pytest.raises(ValueError):
        M.brier_score([], [])
