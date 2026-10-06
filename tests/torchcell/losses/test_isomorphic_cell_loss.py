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


# ---------------------------------------------------------------------------
# 2026.10.06 - Phase 21: ICLoss zero-total branches and ICLossStd
#
# ICLossStd's own SupCR call cannot run (finding below), so the closed-form tests swap
# ``supcr_fn`` for ``_SupCRStub``, an nn.Module returning loss 0.3 with per-dimension
# losses [0.2, 0.4], and set lambda_dist = 0 so the per-task base losses are
# mse_dim + 0.5 * supcr_dim = [2/3 + 0.1, 1 + 0.2] = [0.766667, 1.2].
# ---------------------------------------------------------------------------

import math  # noqa: E402
import re  # noqa: E402
import warnings  # noqa: E402

import torch.nn as nn  # noqa: E402

from torchcell.losses.isomorphic_cell_loss import ICLossStd  # noqa: E402


class _SupCRStub(nn.Module):
    """Stand-in SupCR accepting ICLossStd's three arguments; records them."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(
        self, z_p: torch.Tensor, z_i: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append((z_p, z_i, targets))
        return torch.tensor(0.3), torch.tensor([0.2, 0.4])


def _std_loss(**kwargs: float | torch.Tensor) -> tuple[ICLossStd, _SupCRStub]:
    loss = ICLossStd(lambda_dist=0.0, lambda_supcr=0.5, **kwargs)  # type: ignore[arg-type, unused-ignore]
    stub = _SupCRStub()
    loss.supcr_fn = stub  # type: ignore[assignment, unused-ignore]
    return loss, stub


def test_zero_totals_report_integer_zero_shares() -> None:
    """All-NaN targets make MSE, dist and SupCR all 0, so both totals are 0 and every
    norm_* entry is the int 0 (lines 72 and 81). With predictions equal to targets and
    lambdas 0 only the weighted total is 0: weighted shares are int 0, unweighted ones
    are the tensors dist / (dist + supcr) and supcr / (dist + supcr).
    """
    nan = torch.full((3, 2), float("nan"))
    total, parts = ICLoss(lambda_dist=0.0, lambda_supcr=0.0)(
        PREDICTIONS, nan, _embeddings()
    )
    assert total.item() == 0.0
    for k in ("mse", "dist", "supcr"):
        for kind in ("weighted", "unweighted"):
            value = parts[f"norm_{kind}_{k}"]
            assert type(value) is int and value == 0
    total, parts = ICLoss(lambda_dist=0.0, lambda_supcr=0.0)(
        TARGETS.clone(), TARGETS, _embeddings()
    )
    assert total.item() == 0.0
    assert type(parts["norm_weighted_mse"]) is int
    dist, supcr = parts["dist_loss"].item(), parts["supcr_loss"].item()
    assert parts["norm_unweighted_mse"].item() == 0.0
    assert parts["norm_unweighted_dist"].item() == pytest.approx(dist / (dist + supcr))
    assert parts["norm_unweighted_supcr"].item() == pytest.approx(
        supcr / (dist + supcr)
    )


def test_std_init_sets_log_sigma_and_default_task_weights() -> None:
    """log_sigma = [log 2, log 2] for init_sigma 2; task weights default to ones(2)."""
    loss = ICLossStd(lambda_dist=0.1, lambda_supcr=0.2, init_sigma=2.0)
    torch.testing.assert_close(loss.log_sigma.data, torch.full((2,), math.log(2.0)))
    assert torch.equal(loss.task_weights, torch.ones(2))
    assert (loss.lambda_dist, loss.lambda_supcr, loss.lambda_reg, loss.eps) == (
        0.1,
        0.2,
        0.01,
        1e-6,
    )
    weights = torch.tensor([1.0, 3.0])
    assert ICLossStd(0.0, 0.0, task_weights=weights).task_weights is weights


def test_std_forward_cannot_call_its_own_supcr() -> None:
    """Finding: ICLossStd.forward always raises TypeError.

    It calls ``self.supcr_fn(z_P, z_I, targets)`` (isomorphic_cell_loss.py:181) but
    WeightedSupCRCell.forward takes (perturbed_embeddings, labels) only, so the class
    cannot produce a loss; its one caller is commented out (hetero_cell_pma.py:1068,
    retired 2026-10-06 to the graveyard).
    Pinned until the call matches the two-argument SupCR or the class is retired.
    """
    z = _embeddings()
    with pytest.raises(
        TypeError,
        match=re.escape(
            "WeightedSupCRCell.forward() takes 3 positional arguments but 4 were given"
        ),
    ):
        ICLossStd(lambda_dist=0.0, lambda_supcr=0.5)(PREDICTIONS, TARGETS, z, z)


def test_std_heteroscedastic_closed_form() -> None:
    """Sigma = 2, task weights [1, 3], lambda_reg 0.01, base [0.766667, 1.2].

    weighted_t = w_t / (2 sigma^2 + 1e-6) * base_t + log(sigma + 1e-6):
      fitness 0.766667 / 8 + 0.693147 = 0.788981, gi 3 * 1.2 / 8 + 0.693147 = 1.143148;
    reg = 0.01 * 2 / (4 + 1e-6) = 0.005; total = 1.937129. The stub received
    (z_P, z_I, targets) exactly; prediction std is the unbiased std of each column
    ([1, 0, 2] -> 1, [0, 1, 2] -> 1).
    """
    loss, stub = _std_loss(init_sigma=2.0, task_weights=torch.tensor([1.0, 3.0]))
    z_p, z_i = _embeddings(), _embeddings() + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        total, parts = loss(PREDICTIONS, TARGETS, z_p, z_i)
    log2 = math.log(2.0 + 1e-6)
    fit = (2.0 / 3.0 + 0.1) / (8 + 1e-6) + log2
    gi = 3 * 1.2 / (8 + 1e-6) + log2
    reg = 0.01 * 2 / (4 + 1e-6)
    assert total.item() == pytest.approx(fit + gi + reg, abs=1e-6)
    assert [c is a for c, a in zip(stub.calls[0], (z_p, z_i, TARGETS))] == [True] * 3
    expected = {
        "sigma_fitness": 2.0,
        "sigma_gi": 2.0,
        "mse_loss": MSE,
        # the same unweighted distribution term ICLoss reports for these inputs
        "dist_loss": ICLoss(0.0, 0.0)(PREDICTIONS, TARGETS, z_p)[1]["dist_loss"].item(),
        "supcr_loss": 0.3,
        "fitness_base_loss": 2.0 / 3.0 + 0.1,
        "gi_base_loss": 1.2,
        "fitness_weighted_loss": fit,
        "gi_weighted_loss": gi,
        "reg_term": reg,
        "weighted_mse": MSE,
        "weighted_dist": 0.0,
        "weighted_supcr": 0.15,
        "fitness_pred_std": 1.0,
        "gi_pred_std": 1.0,
        "total_loss": fit + gi + reg,
    }
    assert sorted(parts) == sorted(expected)
    assert parts == pytest.approx(expected, abs=1e-6)


def test_std_prediction_std_skips_nan_targets_and_reports_zero_for_empty() -> None:
    """Targets NaN at (2, 0) and in all of column 1: fitness std over rows 0, 1 of
    predictions[:, 0] = std([1, 0]) = sqrt(0.5); the gi column has no valid target
    and reports 0.0 (line 231).
    """
    loss, _ = _std_loss()
    targets = TARGETS.clone()
    targets[2, 0] = float("nan")
    targets[:, 1] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, parts = loss(PREDICTIONS, targets, _embeddings(), _embeddings())
    assert parts["fitness_pred_std"] == pytest.approx(math.sqrt(0.5), abs=1e-6)
    assert parts["gi_pred_std"] == 0.0


def test_std_loss_sends_no_gradient_to_the_predictions() -> None:
    """Finding: ICLossStd trains only its sigmas, never the model.

    ``base_task_losses = torch.tensor([...])`` (isomorphic_cell_loss.py:190) copies the
    per-task losses into a new leaf, cutting the graph: after backward the predictions
    have ``grad is None`` and only log_sigma has a gradient, d total / d log sigma_t =
    -w_t base_t / sigma^2 + 1 - 2 lambda_reg / sigma^2 = [0.803333, 0.095]. Pinned
    until the base losses are stacked with torch.stack.
    """
    loss, _ = _std_loss(init_sigma=2.0, task_weights=torch.tensor([1.0, 3.0]))
    predictions = PREDICTIONS.clone().requires_grad_(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        total, _ = loss(predictions, TARGETS, _embeddings(), _embeddings())
    total.backward()
    assert predictions.grad is None
    assert loss.log_sigma.grad is not None
    fit_base, gi_base = 2.0 / 3.0 + 0.1, 1.2
    expected = torch.tensor(
        [-fit_base / 4 + 1 - 0.02 / 4, -3 * gi_base / 4 + 1 - 0.02 / 4]
    )
    torch.testing.assert_close(loss.log_sigma.grad, expected, atol=1e-5, rtol=0)
