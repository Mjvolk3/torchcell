# tests/torchcell/losses/test_logcosh.py
# [[tests.torchcell.losses.test_logcosh]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_logcosh.py
"""Exact values for ``LogCoshLoss`` on a three-element residual.

Residuals 0, 1, 2 give log(cosh(r)) = 0, 0.4337808305, 1.3250027473 (cosh(1) =
1.5430806348, cosh(2) = 3.7621956911); the gradient of log(cosh(r)) is tanh(r) =
0, 0.7615941560, 0.9640275801.
"""

import pytest
import torch

from torchcell.losses.logcosh import LogCoshLoss

INPUT = torch.tensor([1.0, 2.0, 3.0])
TARGET = torch.tensor([1.0, 1.0, 1.0])
PER_ELEMENT = torch.tensor([0.0, 0.4337808305, 1.3250027473])


def test_none_reduction_returns_the_per_element_log_cosh() -> None:
    """``reduction="none"`` keeps one value per residual."""
    loss = LogCoshLoss(reduction="none")(INPUT, TARGET)
    torch.testing.assert_close(loss, PER_ELEMENT, atol=1e-6, rtol=0)


def test_mean_and_sum_reductions() -> None:
    """Mean = 1.7587835778 / 3 = 0.5862611926; sum = 1.7587835778."""
    mean = LogCoshLoss(reduction="mean")(INPUT, TARGET)
    total = LogCoshLoss(reduction="sum")(INPUT, TARGET)
    assert mean.item() == pytest.approx(0.5862611926, abs=1e-6)
    assert total.item() == pytest.approx(1.7587835778, abs=1e-6)
    assert mean.shape == () and total.shape == ()


def test_default_reduction_is_mean() -> None:
    """The constructor default matches ``reduction="mean"``."""
    assert LogCoshLoss().reduction == "mean"
    assert LogCoshLoss()(INPUT, TARGET).item() == pytest.approx(0.5862611926, abs=1e-6)


def test_gradient_is_tanh_of_the_residual_over_n() -> None:
    """d/dx mean(log cosh(x - t)) = tanh(x - t) / 3 per element."""
    x = INPUT.clone().requires_grad_(True)
    LogCoshLoss()(x, TARGET).backward()
    assert x.grad is not None
    expected = torch.tensor([0.0, 0.7615941560, 0.9640275801]) / 3
    torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=0)


def test_invalid_reduction_raises() -> None:
    """Anything but none/mean/sum is rejected at construction."""
    with pytest.raises(ValueError, match="Invalid reduction mode: median"):
        LogCoshLoss(reduction="median")
