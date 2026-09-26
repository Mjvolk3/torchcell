# tests/torchcell/losses/test_isomorphic_cell_loss.py
# [[tests.torchcell.losses.test_isomorphic_cell_loss]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/losses/test_isomorphic_cell_loss.py
"""``ICLoss`` composition: MSE by hand, the other two terms by their reported values.

Predictions [[1, 0], [0, 1], [2, 2]] against targets [[0, 1], [1, 0], [2, 3]]: squared
errors are (1, 1, 0) in dimension 0 and (1, 1, 1) in dimension 1, so the per-dimension
MSEs are 2/3 and 1 and their mean is 5/6. The targets vary within each dimension on
purpose: the distribution term fits a Gaussian KDE and a constant column makes its
covariance singular. The distribution and SupCR terms have no short closed form, so the
tests pin how they are combined: total = mse + lambda_dist * dist + lambda_supcr * supcr,
the normalized shares sum to one, and switching both lambdas off leaves the MSE alone.
"""

import pytest
import torch

from torchcell.losses.isomorphic_cell_loss import ICLoss

PREDICTIONS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
TARGETS = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])
MSE_DIM = torch.tensor([2.0 / 3.0, 1.0])
MSE = 5.0 / 6.0


def _embeddings() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(3, 4)


def test_zero_lambdas_reduce_the_total_to_the_mse() -> None:
    """With both lambdas 0 the total is exactly the 5/6 MSE."""
    total, parts = ICLoss(lambda_dist=0.0, lambda_supcr=0.0)(
        PREDICTIONS, TARGETS, _embeddings()
    )
    assert total.item() == pytest.approx(MSE, abs=1e-6)
    assert parts["mse_loss"].item() == pytest.approx(MSE, abs=1e-6)
    torch.testing.assert_close(parts["mse_dim_losses"], MSE_DIM)
    assert parts["weighted_dist"].item() == 0.0
    assert parts["weighted_supcr"].item() == 0.0


def test_total_is_the_lambda_weighted_sum_of_the_three_terms() -> None:
    """Total = mse + 0.5 dist + 2 supcr; the reported weighted parts match."""
    total, parts = ICLoss(lambda_dist=0.5, lambda_supcr=2.0)(
        PREDICTIONS, TARGETS, _embeddings()
    )
    dist = parts["dist_loss"].item()
    supcr = parts["supcr_loss"].item()
    assert dist > 0 and supcr > 0
    assert parts["weighted_dist"].item() == pytest.approx(0.5 * dist, rel=1e-6)
    assert parts["weighted_supcr"].item() == pytest.approx(2.0 * supcr, rel=1e-6)
    assert total.item() == pytest.approx(MSE + 0.5 * dist + 2.0 * supcr, rel=1e-6)
    assert parts["total_loss"] is total
    assert parts["total_weighted"].item() == pytest.approx(total.item(), rel=1e-6)


def test_normalized_shares_sum_to_one_in_both_weightings() -> None:
    """norm_weighted_* and norm_unweighted_* are fractions of their respective totals."""
    _, parts = ICLoss(lambda_dist=0.5, lambda_supcr=2.0)(
        PREDICTIONS, TARGETS, _embeddings()
    )
    weighted = sum(parts[f"norm_weighted_{k}"].item() for k in ("mse", "dist", "supcr"))
    unweighted = sum(
        parts[f"norm_unweighted_{k}"].item() for k in ("mse", "dist", "supcr")
    )
    assert weighted == pytest.approx(1.0, abs=1e-6)
    assert unweighted == pytest.approx(1.0, abs=1e-6)
    assert parts["norm_weighted_mse"].item() == pytest.approx(
        MSE / parts["total_weighted"].item(), rel=1e-6
    )


def test_gradient_of_the_mse_term_is_the_residual_over_three() -> None:
    """With both lambdas 0, d total / d p = 2 (p - t) / (3 samples * 2 dims) = (p - t) / 3."""
    predictions = PREDICTIONS.clone().requires_grad_(True)
    total, _ = ICLoss(lambda_dist=0.0, lambda_supcr=0.0)(
        predictions, TARGETS, _embeddings()
    )
    total.backward()
    assert predictions.grad is not None
    expected = (PREDICTIONS - TARGETS) / 3  # [[1/3, -1/3], [-1/3, 1/3], [0, -1/3]]
    torch.testing.assert_close(predictions.grad, expected, atol=1e-6, rtol=0)


def test_full_loss_gradient_reaches_predictions_and_embeddings() -> None:
    """The dist and SupCR terms add finite gradient to both inputs (embeddings only via SupCR)."""
    predictions = PREDICTIONS.clone().requires_grad_(True)
    embeddings = _embeddings().requires_grad_(True)
    total, _ = ICLoss(lambda_dist=0.5, lambda_supcr=2.0)(
        predictions, TARGETS, embeddings
    )
    total.backward()
    assert predictions.grad is not None and embeddings.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert torch.isfinite(embeddings.grad).all()
    # the MSE-only gradient is (p - t) / 3; the extra terms must change it somewhere
    assert not torch.allclose(predictions.grad, (PREDICTIONS - TARGETS) / 3)
    assert embeddings.grad.abs().sum().item() > 0


def test_dimension_weights_reweight_the_mse() -> None:
    """Weights [3, 1] normalize to [0.75, 0.25]: 0.75 * 2/3 + 0.25 * 1 = 0.75."""
    weights = torch.tensor([3.0, 1.0])
    _, parts = ICLoss(lambda_dist=0.0, lambda_supcr=0.0, weights=weights)(
        PREDICTIONS, TARGETS, _embeddings()
    )
    assert parts["mse_loss"].item() == pytest.approx(0.75, abs=1e-6)
    torch.testing.assert_close(parts["mse_dim_losses"], MSE_DIM)
