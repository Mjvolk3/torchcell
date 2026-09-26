# tests/torchcell/losses/test_losses_dcell.py
# [[tests.torchcell.losses.test_losses_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_losses_dcell.py
"""Exact values for ``DCellLoss``: root MSE plus alpha-weighted mean of subsystem MSEs.

Named ``test_losses_dcell.py`` because ``tests/torchcell/models/test_dcell.py`` exists and
pytest's prepend import mode cannot hold two ``test_dcell`` modules; the pairing is
recorded in ``[tool.torchcell.test_exceptions.pairs]``.

Fixture: predictions [1, 2] against targets [0, 0] give a primary MSE of (1 + 4) / 2 =
2.5. Subsystem outputs GO:1 = [0, 0] (MSE 0) and GO:2 = [2, 2] (MSE 4) average to 2.0;
with alpha 0.3 the weighted auxiliary term is 0.6 and the total 3.1.
"""

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


def test_total_is_primary_plus_alpha_times_mean_auxiliary() -> None:
    """2.5 + 0.3 * mean(0, 4) = 3.1, with every component reported."""
    total, parts = DCellLoss(alpha=0.3)(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(3.1, abs=1e-6)
    assert parts["primary_loss"].item() == pytest.approx(2.5, abs=1e-6)
    assert parts["auxiliary_loss"].item() == pytest.approx(2.0, abs=1e-6)
    assert parts["weighted_auxiliary_loss"].item() == pytest.approx(0.6, abs=1e-6)
    assert set(parts) == {"primary_loss", "auxiliary_loss", "weighted_auxiliary_loss"}


def test_root_is_skipped_by_name_and_by_equality() -> None:
    """GO:ROOT is excluded by key; a subsystem equal to the predictions is excluded too."""
    # GO:ROOT is excluded by its key even though its value [5, 5] differs from the
    # predictions; GO:7 is excluded by equality with them.
    outputs = {
        "linear_outputs": {
            "GO:ROOT": torch.tensor([5.0, 5.0]),
            "GO:7": PREDICTIONS.clone(),
        }
    }
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
    """Alpha = 1.0 gives 2.5 + 2.0 = 4.5."""
    total, _ = DCellLoss(alpha=1.0)(PREDICTIONS, _outputs(), TARGET)
    assert total.item() == pytest.approx(4.5, abs=1e-6)


def test_gradient_reaches_predictions_and_subsystem_outputs() -> None:
    """d/dp of the primary MSE is 2(p - t)/2 = p: [1, 2]; GO:2's gradient is 0.3 * 2 * 2 / 2 / 2."""
    predictions = PREDICTIONS.clone().requires_grad_(True)
    go2 = torch.tensor([2.0, 2.0], requires_grad=True)
    outputs = {"linear_outputs": {"GO:1": torch.tensor([0.0, 0.0]), "GO:2": go2}}
    total, _ = DCellLoss(alpha=0.3)(predictions, outputs, TARGET)
    total.backward()
    assert predictions.grad is not None and go2.grad is not None
    torch.testing.assert_close(predictions.grad, torch.tensor([1.0, 2.0]))
    # mean over 2 subsystems, MSE mean over 2 elements: 0.3 * (1/2) * (2 * 2 / 2) = 0.3
    torch.testing.assert_close(go2.grad, torch.tensor([0.3, 0.3]))
