# tests/torchcell/losses/test_losses_dcell.py
# [[tests.torchcell.losses.test_losses_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_losses_dcell.py
"""Exact values for ``DCellLoss``: root MSE plus alpha times the SUM of subsystem MSEs.

Named ``test_losses_dcell.py`` because ``tests/torchcell/models/test_dcell.py`` exists and
pytest's prepend import mode cannot hold two ``test_dcell`` modules; the pairing is
recorded in ``[tool.torchcell.test_exceptions.pairs]``.

Ma et al. 2018 (mirror ``maUsingDeepLearning2018/paper.md`` line 196) define the
objective as ``Loss(Linear(O^(r)), y) + alpha * sum_{t != r} Loss(Linear(O^(t)), y)``
with alpha = 0.3, so the default ``aux_reduction="sum"`` sums; ``"mean"`` is the
pre-2026-09-30 behavior every 005/006 run trained with (issue #554).

Fixture: predictions [1, 2] against targets [0, 0] give a primary MSE of (1 + 4) / 2 =
2.5. Subsystem outputs GO:1 = [0, 0] (MSE 0) and GO:2 = [2, 2] (MSE 4) sum to 4.0, so
alpha 0.3 weights them to 1.2 and the total is 3.7; their mean is 2.0, weighted 0.6,
total 3.1.
"""

from typing import Any, Literal

import pytest
import torch

from torchcell.losses.dcell import DCellLoss

PREDICTIONS = torch.tensor([1.0, 2.0])
TARGET = torch.tensor([0.0, 0.0])


def _outputs() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "linear_outputs": {
            "GO:ROOT": PREDICTIONS,
            "GO:1": torch.tensor([0.0, 0.0]),
            "GO:2": torch.tensor([2.0, 2.0]),
        }
    }


def test_default_is_the_paper_sum_over_non_root_subsystems() -> None:
    """2.5 + 0.3 * (0 + 4) = 3.7, with every component reported."""
    loss = DCellLoss()
    assert loss.alpha == 0.3
    assert loss.aux_reduction == "sum"
    total, parts = loss(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(3.7, abs=1e-6)
    assert parts["primary_loss"].item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == pytest.approx(4.0, abs=1e-6)
    assert parts["weighted_auxiliary_loss"].item() == pytest.approx(1.2, abs=1e-6)
    assert set(parts) == {"primary_loss", "auxiliary_loss", "weighted_auxiliary_loss"}


def test_mean_reduction_reproduces_the_pre_fix_runs() -> None:
    """aux_reduction="mean": 2.5 + 0.3 * mean(0, 4) = 3.1 (effective alpha 0.3 / 2)."""
    total, parts = DCellLoss(alpha=0.3, aux_reduction="mean")(
        PREDICTIONS, _outputs(), TARGET
    )
    assert total.item() == pytest.approx(3.1, abs=1e-6)
    assert parts["auxiliary_loss"].item() == pytest.approx(2.0, abs=1e-6)
    assert parts["weighted_auxiliary_loss"].item() == pytest.approx(0.6, abs=1e-6)


def test_unknown_reduction_is_refused() -> None:
    """Any reduction other than "sum" or "mean" raises with the exact message."""
    unknown: Any = "max"
    with pytest.raises(
        ValueError, match=r"^aux_reduction must be 'sum' or 'mean', got 'max'$"
    ):
        DCellLoss(aux_reduction=unknown)


def test_root_is_skipped_by_key_and_alias_only() -> None:
    """GO:ROOT and its alias GO:9 are skipped; GO:7, equal in value, is counted.

    ``DCell`` stores the root head under ``GO:<root index>`` and binds the same object
    to ``GO:ROOT``. GO:7 is a distinct tensor holding the root's values, so it is a
    non-root subsystem with MSE 2.5: total = 2.5 + 0.3 * 2.5 = 3.25. The old
    ``torch.equal`` test dropped it and returned 2.5.
    """
    root = PREDICTIONS.clone()
    outputs = {"linear_outputs": {"GO:9": root, "GO:ROOT": root, "GO:7": root.clone()}}
    total, parts = DCellLoss(alpha=0.3)(root, outputs, TARGET)
    assert parts["auxiliary_loss"].item() == pytest.approx(2.5, abs=1e-6)
    assert total.item() == pytest.approx(3.25, abs=1e-6)


def test_go_root_key_is_skipped_whatever_its_value() -> None:
    """GO:ROOT = [5, 5] differs from the predictions and is still skipped by its key."""
    outputs = {"linear_outputs": {"GO:ROOT": torch.tensor([5.0, 5.0])}}
    total, parts = DCellLoss(alpha=0.3)(PREDICTIONS, outputs, TARGET)
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == 0.0


def test_auxiliary_losses_can_be_switched_off() -> None:
    """With use_auxiliary_losses=False the total is the primary MSE alone."""
    total, parts = DCellLoss(alpha=0.3, use_auxiliary_losses=False)(
        PREDICTIONS, _outputs(), TARGET
    )
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == 0.0
    assert parts["weighted_auxiliary_loss"].item() == 0.0


def test_missing_linear_outputs_gives_the_primary_loss() -> None:
    """An outputs dict without subsystem outputs contributes nothing auxiliary."""
    total, _ = DCellLoss()(PREDICTIONS, {}, TARGET)
    assert total.item() == pytest.approx(2.5, abs=1e-6)


def test_alpha_scales_only_the_auxiliary_term() -> None:
    """Alpha = 1.0 gives 2.5 + (0 + 4) = 6.5."""
    total, _ = DCellLoss(alpha=1.0)(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(6.5, abs=1e-6)


@pytest.mark.parametrize(("reduction", "go2_grad"), [("sum", 0.6), ("mean", 0.3)])
def test_gradient_reaches_predictions_and_subsystem_outputs(
    reduction: Literal["sum", "mean"], go2_grad: float
) -> None:
    """d/dp of the primary MSE is 2(p - t)/2 = p: [1, 2]; GO:2 carries alpha * 2(2)/2.

    Sum: 0.3 * (2 * 2 / 2) = 0.6 per element; mean over the 2 subsystems halves it to 0.3.
    """
    predictions = PREDICTIONS.clone().requires_grad_(True)
    go2 = torch.tensor([2.0, 2.0], requires_grad=True)
    outputs = {"linear_outputs": {"GO:1": torch.tensor([0.0, 0.0]), "GO:2": go2}}
    loss = DCellLoss(alpha=0.3, aux_reduction=reduction)
    total, _ = loss(predictions, outputs, TARGET)
    total.backward()
    assert predictions.grad is not None and go2.grad is not None
    torch.testing.assert_close(predictions.grad, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(go2.grad, torch.tensor([go2_grad, go2_grad]))
