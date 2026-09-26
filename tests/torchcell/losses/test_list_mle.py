# tests/torchcell/losses/test_list_mle.py
# [[tests.torchcell.losses.test_list_mle]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_list_mle.py
"""Exact values for ``ListMLELoss`` as implemented, and the ranking bug it carries.

The module sorts predictions by the true ranking, takes one ``log_softmax`` over the
list, and returns the negated sum averaged over the batch. For predictions [1, 2, 3]:
logsumexp = 3 + log(1 + e^-1 + e^-2) = 3 + log(1.5032147244) = 3.4076059644, so
log_softmax = [-2.4076059644, -1.4076059644, -0.4076059644] and the loss is their
negated sum, 4.2228178933. A second list [0, 0, 0] scores 3 log 3 = 3.2958368660.

Because a single ``log_softmax`` is used instead of the cumulative Plackett-Luce terms,
-sum_i log_softmax(y)_i = -sum(y) + n * logsumexp(y), which is the same for every
permutation: the implemented loss never reads the true ranking. The Plackett-Luce value
on the fixture is -[(1 - lse(1, 2, 3)) + (2 - lse(2, 3)) + (3 - 3)] = 2.4076059644 +
1.3132616875 = 3.7208676519. ``ListMLELoss`` is live (``torchcell.trainers.
neo_regression`` uses it), so the correct value is a strict xfail: the day the loss is
fixed, that test passes and the two pinned-behavior tests fail, which is the intended
signal (quality audit, Phase 1 of [[plan.test-suite-buildout.2026.09.25]]).
"""

import math

import pytest
import torch

from torchcell.losses.list_mle import ListMLELoss

Y_PRED = torch.tensor([[1.0, 2.0, 3.0]])
Y_TRUE_DESC = torch.tensor([[3.0, 2.0, 1.0]])
IMPLEMENTED = 4.2228178933
PLACKETT_LUCE = 3.7208676519
FLAT_LIST = 3 * math.log(3)  # 3.2958368660


def test_exact_value_on_a_three_item_list() -> None:
    """-(sum log_softmax([1, 2, 3])) = 4.2228178933."""
    loss = ListMLELoss()(Y_PRED, Y_TRUE_DESC)
    assert loss.item() == pytest.approx(IMPLEMENTED, abs=1e-6)


def test_one_dimensional_inputs_are_treated_as_a_single_list() -> None:
    """1-D tensors are unsqueezed to batch size one and give the same value."""
    loss = ListMLELoss()(Y_PRED[0], Y_TRUE_DESC[0])
    assert loss.item() == pytest.approx(IMPLEMENTED, abs=1e-6)


def test_batch_loss_is_the_mean_over_distinct_lists() -> None:
    """Rows [1, 2, 3] and [0, 0, 0] average to (4.2228178933 + 3 log 3) / 2."""
    y_pred = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y_true = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    loss = ListMLELoss()(y_pred, y_true)
    assert loss.item() == pytest.approx((IMPLEMENTED + FLAT_LIST) / 2, abs=1e-6)


def test_integer_inputs_are_cast_to_float() -> None:
    """Integer predictions and scores give the float value."""
    loss = ListMLELoss()(torch.tensor([[1, 2, 3]]), torch.tensor([[3, 2, 1]]))
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(IMPLEMENTED, abs=1e-6)


def test_loss_does_not_depend_on_the_true_ranking() -> None:
    """Pinned defect: a reversed ranking gives the same loss (see the module docstring)."""
    descending = ListMLELoss()(Y_PRED, Y_TRUE_DESC)
    ascending = ListMLELoss()(Y_PRED, torch.tensor([[1.0, 2.0, 3.0]]))
    assert ascending.item() == pytest.approx(descending.item(), abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="ListMLELoss uses one log_softmax over the list, not the cumulative "
    "Plackett-Luce terms; the true ranking never enters the value",
)
def test_plackett_luce_value() -> None:
    """The ListMLE of the paper on the fixture is 3.7208676519."""
    loss = ListMLELoss()(Y_PRED, Y_TRUE_DESC)
    assert loss.item() == pytest.approx(PLACKETT_LUCE, abs=1e-6)


def test_gradient_flows_to_the_predictions() -> None:
    """D loss / d y_pred = 3 * softmax(y_pred) - 1 for a single list."""
    y_pred = Y_PRED.clone().requires_grad_(True)
    ListMLELoss()(y_pred, Y_TRUE_DESC).backward()
    assert y_pred.grad is not None
    expected = 3 * torch.softmax(Y_PRED, dim=1) - 1
    torch.testing.assert_close(y_pred.grad, expected, atol=1e-6, rtol=0)
