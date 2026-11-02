"""Binary log loss must remain defined at supported clipping boundaries."""
import math

import pytest

from szl_calibration.metrics import log_loss


@pytest.mark.parametrize("eps", [0.0, -1.0, 0.75, 1.0, float("nan"),
                                 float("inf"), -float("inf"), True, False,
                                 None, "0.01", 10**400])
def test_invalid_clipping_epsilon_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        log_loss([0.2, 0.8], [0, 1], eps=eps)


@pytest.mark.parametrize("eps", [1e-15, 1e-20, 1e-100, math.ulp(0.0)])
def test_endpoint_losses_use_observed_class_probability(eps):
    # Correct endpoint predictions do not evaluate an unused log(0).
    assert log_loss([0.0, 1.0], [0, 1], eps=eps) == -math.log(1.0 - eps)
    # Incorrect endpoints are clipped directly to eps, not 1 - (1 - eps).
    assert log_loss([0.0, 1.0], [1, 0], eps=eps) == -math.log(eps)
    assert math.isfinite(log_loss([0.0, 1.0], [1, 0], eps=eps))


def test_unclipped_probabilities_preserve_binary_log_loss():
    probs, labels = [0.2, 0.8, 0.3, 0.7], [0, 1, 1, 0]
    expected = -(math.log(0.8) + math.log(0.8) + math.log(0.3) + math.log(1 - 0.7)) / 4
    assert log_loss(probs, labels) == expected
    assert probs == [0.2, 0.8, 0.3, 0.7]
    assert labels == [0, 1, 1, 0]


def test_valid_epsilon_near_one_half_stays_symmetric():
    eps = math.nextafter(0.5, 0.0)
    assert log_loss([0.0], [1], eps=eps) == log_loss([1.0], [0], eps=eps)


def test_one_half_clipping_retains_existing_degenerate_interval():
    assert log_loss([0.0, 0.2, 0.8, 1.0], [0, 1, 0, 1], eps=0.5) == -math.log(0.5)
