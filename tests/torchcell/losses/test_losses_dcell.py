# tests/torchcell/losses/test_losses_dcell.py
# [[tests.torchcell.losses.test_losses_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_losses_dcell.py
"""Exact values for ``DCellLoss``: root MSE plus alpha times the SUM of subsystem MSEs.

Named ``test_losses_dcell.py`` because ``tests/torchcell/models/test_dcell.py`` exists and
pytest's prepend import mode cannot hold two ``test_dcell`` modules; the pairing is
recorded in ``[tool.torchcell.test_exceptions.pairs]``.

Ma et al. 2018 (mirror ``maUsingDeepLearning2018/paper.md`` line 196) define the
objective as ``Loss(Linear(O^(r)), y) + alpha * sum_{t != r} Loss(Linear(O^(t)), y)``
with alpha = 0.3; ``aux_reduction`` is required, ``"sum"`` is the paper and ``"mean"``
divides by the number of counted subsystems (issue #554).

Fixture: predictions [1, 2] against targets [0, 0] give a primary MSE of (1 + 4) / 2 =
2.5. Subsystem outputs GO:1 = [0, 0] (MSE 0) and GO:2 = [2, 2] (MSE 4) sum to 4.0, so
alpha 0.3 weights them to 1.2 and the total is 3.7; their mean is 2.0, weighted 0.6,
total 3.1. ``root_key`` declares GO:0 as the root's own key.
"""

from typing import Any, Literal

import pytest
import torch

from torchcell.losses.dcell import DCellLoss

PREDICTIONS = torch.tensor([1.0, 2.0])
TARGET = torch.tensor([0.0, 0.0])


def _outputs() -> dict[str, Any]:
    return {
        "linear_outputs": {
            "GO:0": PREDICTIONS,
            "GO:ROOT": PREDICTIONS,
            "GO:1": torch.tensor([0.0, 0.0]),
            "GO:2": torch.tensor([2.0, 2.0]),
        },
        "root_key": "GO:0",
    }


def test_sum_is_the_paper_objective_over_non_root_subsystems() -> None:
    """2.5 + 0.3 * (0 + 4) = 3.7, with every component reported."""
    loss = DCellLoss(aux_reduction="sum")
    assert (loss.alpha, loss.use_auxiliary_losses, loss.aux_reduction) == (
        0.3,
        True,
        "sum",
    )
    total, parts = loss(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(3.7, abs=1e-6)
    assert parts["primary_loss"].item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == pytest.approx(4.0, abs=1e-6)
    assert parts["weighted_auxiliary_loss"].item() == pytest.approx(1.2, abs=1e-6)
    assert set(parts) == {"primary_loss", "auxiliary_loss", "weighted_auxiliary_loss"}


def test_mean_reduction_divides_the_sum_by_the_counted_subsystems() -> None:
    """aux_reduction="mean": 2.5 + 0.3 * (0 + 4) / 2 = 3.1 (effective alpha 0.3 / 2)."""
    total, parts = DCellLoss(alpha=0.3, aux_reduction="mean")(
        PREDICTIONS, _outputs(), TARGET
    )
    assert total.item() == pytest.approx(3.1, abs=1e-6)
    assert parts["auxiliary_loss"].item() == pytest.approx(2.0, abs=1e-6)
    assert parts["weighted_auxiliary_loss"].item() == pytest.approx(0.6, abs=1e-6)


def test_aux_reduction_is_required_and_keyword_only() -> None:
    """No default: omitting it, or passing it positionally, is a TypeError."""
    loss_class: Any = DCellLoss
    with pytest.raises(
        TypeError, match="required keyword-only argument: 'aux_reduction'"
    ):
        loss_class()
    positional: tuple[Any, ...] = (0.3, True, "sum")
    with pytest.raises(TypeError, match="takes from 1 to 3 positional arguments"):
        loss_class(*positional)


def test_unknown_reduction_is_refused() -> None:
    """Any reduction other than "sum" or "mean" raises with the exact message."""
    unknown: Any = "max"
    with pytest.raises(
        ValueError, match=r"^aux_reduction must be 'sum' or 'mean', got 'max'$"
    ):
        DCellLoss(aux_reduction=unknown)


def test_root_is_skipped_by_its_declared_key_not_by_identity_or_value() -> None:
    """The declared root GO:9 is skipped although it is a DISTINCT object from both
    the predictions and GO:ROOT; GO:7, a non-root head equal in value to the root, is
    counted with MSE 2.5: total = 2.5 + 0.3 * 2.5 = 3.25. (Identity skipping counted
    GO:9 too, 3.25 + 0.75 = 4.0; the old ``torch.equal`` test dropped GO:7, 2.5.)
    """
    outputs = {
        "linear_outputs": {
            "GO:9": PREDICTIONS.clone(),
            "GO:ROOT": PREDICTIONS.clone(),
            "GO:7": PREDICTIONS.clone(),
        },
        "root_key": "GO:9",
    }
    total, parts = DCellLoss(alpha=0.3, aux_reduction="sum")(
        PREDICTIONS.clone(), outputs, TARGET
    )
    assert parts["auxiliary_loss"].item() == pytest.approx(2.5, abs=1e-6)
    assert total.item() == pytest.approx(3.25, abs=1e-6)


def test_go_root_key_is_skipped_whatever_its_value() -> None:
    """GO:ROOT = [5, 5] differs from the predictions and is still skipped by its key."""
    outputs = {
        "linear_outputs": {"GO:ROOT": torch.tensor([5.0, 5.0])},
        "root_key": "GO:ROOT",
    }
    total, parts = DCellLoss(alpha=0.3, aux_reduction="sum")(
        PREDICTIONS, outputs, TARGET
    )
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == 0.0


def test_linear_outputs_without_a_declared_root_key_are_refused() -> None:
    """With no ``root_key`` the loss cannot tell the root from a subsystem: it raises."""
    outputs = {"linear_outputs": {"GO:1": torch.tensor([0.0, 0.0])}}
    with pytest.raises(
        ValueError,
        match=r"^DCellLoss: outputs has 'linear_outputs' but no 'root_key'; the "
        r"model must declare which head is the root$",
    ):
        DCellLoss(aux_reduction="sum")(PREDICTIONS, outputs, TARGET)


def test_a_head_shaped_unlike_the_target_is_refused_not_broadcast() -> None:
    """A ``[2, 1]`` head against a ``[2]`` target would broadcast to a ``[2, 2]`` grid."""
    outputs = _outputs()
    outputs["linear_outputs"]["GO:2"] = torch.tensor([[2.0], [2.0]])
    with pytest.raises(
        ValueError,
        match=r"^DCellLoss: GO:2 has shape \(2, 1\) but the target has shape \(2,\); "
        r"shapes must match exactly$",
    ):
        DCellLoss(aux_reduction="sum")(PREDICTIONS, outputs, TARGET)


def test_predictions_shaped_unlike_the_target_are_refused() -> None:
    """``[2, 1]`` predictions against a ``[2]`` target raise even with auxiliaries off."""
    with pytest.raises(
        ValueError,
        match=r"^DCellLoss: predictions has shape \(2, 1\) but the target has shape "
        r"\(2,\); shapes must match exactly$",
    ):
        DCellLoss(use_auxiliary_losses=False, aux_reduction="sum")(
            PREDICTIONS.unsqueeze(1), {}, TARGET
        )


def test_auxiliary_losses_can_be_switched_off() -> None:
    """With use_auxiliary_losses=False the total is the primary MSE alone."""
    total, parts = DCellLoss(
        alpha=0.3, use_auxiliary_losses=False, aux_reduction="sum"
    )(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == 0.0
    assert parts["weighted_auxiliary_loss"].item() == 0.0


def test_missing_linear_outputs_gives_the_primary_loss() -> None:
    """An outputs dict without subsystem outputs contributes nothing auxiliary."""
    total, _ = DCellLoss(aux_reduction="sum")(PREDICTIONS, {}, TARGET)
    assert total.item() == pytest.approx(2.5, abs=1e-6)


def test_alpha_scales_only_the_auxiliary_term() -> None:
    """Alpha = 1.0 gives 2.5 + (0 + 4) = 6.5."""
    total, _ = DCellLoss(alpha=1.0, aux_reduction="sum")(
        PREDICTIONS, _outputs(), TARGET
    )
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
    outputs = {
        "linear_outputs": {"GO:1": torch.tensor([0.0, 0.0]), "GO:2": go2},
        "root_key": "GO:0",
    }
    total, _ = DCellLoss(alpha=0.3, aux_reduction=reduction)(
        predictions, outputs, TARGET
    )
    total.backward()
    assert predictions.grad is not None and go2.grad is not None
    torch.testing.assert_close(predictions.grad, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(go2.grad, torch.tensor([go2_grad, go2_grad]))
