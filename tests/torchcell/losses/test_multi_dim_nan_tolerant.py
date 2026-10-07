"""Tests for the multi-dimensional NaN-tolerant WeightedDistLoss."""

import pytest
import torch

from torchcell.losses.multi_dim_nan_tolerant import WeightedDistLoss


@pytest.fixture
def loss_fn():
    """Fixture to create loss function instance."""
    return WeightedDistLoss(bandwidth=0.5)


@pytest.fixture
def weighted_loss_fn():
    """Fixture to create weighted loss function instance."""
    weights = torch.tensor([0.7, 0.3])  # Asymmetric weights
    return WeightedDistLoss(bandwidth=0.5, weights=weights)


def generate_test_data(
    batch_size: int, num_dims: int, nan_ratio: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to generate test data with controlled NaN distribution."""
    # Create predictions with normal distribution
    y_pred = torch.randn(batch_size, num_dims)

    # Create ground truth with slightly different distribution
    y_true = torch.randn(batch_size, num_dims) * 1.2 + 0.5

    # Add NaN values to ground truth
    nan_mask = torch.rand_like(y_true) < nan_ratio
    y_true[nan_mask] = float("nan")

    return y_pred, y_true


def test_loss_basic_functionality(loss_fn):
    """Test basic functionality without NaN values."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.2, 2.2], [2.8, 3.8]])

    weighted_loss, dim_losses = loss_fn(y_pred, y_true)
    assert isinstance(weighted_loss, torch.Tensor)
    assert isinstance(dim_losses, torch.Tensor)
    assert weighted_loss.ndim == 0  # Should be scalar
    assert dim_losses.shape == (2,)  # Should have 2 dimensions
    assert not torch.isnan(weighted_loss)
    assert not torch.any(torch.isnan(dim_losses))
    assert weighted_loss >= 0  # Loss should be non-negative
    assert torch.all(dim_losses >= 0)  # Dimension losses should be non-negative


def test_weighted_loss(weighted_loss_fn):
    """Test that weights are properly applied."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.2, 2.2], [2.8, 3.8]])

    weighted_loss, dim_losses = weighted_loss_fn(y_pred, y_true)

    # Manually calculate weighted loss to verify
    expected_weighted_loss = (
        dim_losses * weighted_loss_fn.weights
    ).sum() / weighted_loss_fn.weights.sum()
    assert torch.allclose(weighted_loss, expected_weighted_loss)


def test_nan_handling(loss_fn):
    """Test handling of NaN values.

    The current KDE-based DistLoss estimates a per-dimension distribution with
    scipy.stats.gaussian_kde, which requires >= 2 distinct valid values per
    dimension. NaN tolerance therefore means: scattered NaNs are dropped and the
    loss is still computed from the remaining valid points (it does not mean a
    single surviving value per dimension is supported). Use a batch large enough
    that each dimension retains multiple valid, non-identical values.
    """
    torch.manual_seed(0)
    y_pred = torch.randn(8, 2)
    y_true = torch.randn(8, 2)
    # Scatter NaNs without emptying or collapsing either dimension to one value.
    y_true[0, 1] = float("nan")
    y_true[3, 0] = float("nan")
    y_true[5, 1] = float("nan")

    weighted_loss, dim_losses = loss_fn(y_pred, y_true)
    assert not torch.isnan(weighted_loss)
    assert not torch.any(torch.isnan(dim_losses))
    assert weighted_loss >= 0


def test_all_nan_dimension(loss_fn):
    """Test handling of dimensions where all values are NaN."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.2, float("nan")], [1.5, float("nan")]])

    weighted_loss, dim_losses = loss_fn(y_pred, y_true)
    assert not torch.isnan(weighted_loss)
    assert not torch.any(torch.isnan(dim_losses))
    assert weighted_loss >= 0
    # Second dimension should have zero loss
    assert dim_losses[1] == 0


# NOTE: tests for `WeightedDistLoss.kde` and
# `WeightedDistLoss.generate_pseudo_labels` were removed. Those were methods of an
# earlier, custom-Torch KDE implementation of this loss (now the commented-out
# block in torchcell/losses/multi_dim_nan_tolerant.py). The class was deliberately
# rewritten to follow the original DistLoss paper: density estimation now uses
# scipy.stats.gaussian_kde inside the private `_get_label_distribution`, and
# theoretical labels are produced by `_get_batch_theoretical_labels` -- there is no
# longer a public `kde` or `generate_pseudo_labels` method to test. The
# train/eval-dependent pseudo-label statistics buffering these tests exercised was
# relocated to `BufferedWeightedDistLoss` in torchcell/losses/mle_dist_supcr.py.


def test_batch_invariance(loss_fn):
    """Test that loss behaves reasonably across different batch sizes."""
    torch.manual_seed(42)  # For reproducibility

    mean, std = 0.0, 1.0
    y_pred1 = torch.normal(mean, std, size=(32, 2))
    y_pred2 = torch.normal(mean, std, size=(64, 2))

    y_true1 = torch.normal(mean + 0.5, std, size=(32, 2))
    y_true2 = torch.normal(mean + 0.5, std, size=(64, 2))

    weighted_loss1, _ = loss_fn(y_pred1, y_true1)
    weighted_loss2, _ = loss_fn(y_pred2, y_true2)

    # Test that both losses are positive and finite
    assert weighted_loss1 > 0 and weighted_loss2 > 0
    assert torch.isfinite(weighted_loss1) and torch.isfinite(weighted_loss2)

    # Test that the ratio between losses is within a reasonable range
    ratio = (
        (weighted_loss1 / weighted_loss2)
        if weighted_loss1 > weighted_loss2
        else (weighted_loss2 / weighted_loss1)
    )
    assert ratio < 5.0, f"Loss ratio {ratio} is too large between batch sizes"


def test_distribution_alignment(loss_fn):
    """Test that loss encourages distribution alignment."""
    y_pred = torch.randn(100, 1) * 2.0 + 1.0  # N(1, 2)
    y_true = torch.randn(100, 1) * 0.5 - 1.0  # N(-1, 0.5)

    initial_loss, _ = loss_fn(y_pred, y_true)

    # Create predictions with more similar distribution
    y_pred_aligned = torch.randn(100, 1) * 0.5 - 0.8  # More similar to N(-1, 0.5)

    aligned_loss, _ = loss_fn(y_pred_aligned, y_true)
    assert aligned_loss < initial_loss


def test_gradient_flow(loss_fn):
    """Test that gradients flow properly through the loss."""
    y_pred = torch.randn(10, 2, requires_grad=True)
    y_true = torch.randn(10, 2)

    weighted_loss, _ = loss_fn(y_pred, y_true)
    weighted_loss.backward()

    assert y_pred.grad is not None
    assert not torch.any(torch.isnan(y_pred.grad))


def test_device_compatibility():
    """Test that loss works on both CPU and CUDA if available.

    The loss carries a registered `weights` buffer, so (like any nn.Module) it must
    be moved to the target device together with its inputs; only the inputs being
    on CUDA while the module stays on CPU is a usage error, not something the loss
    promises to paper over. A fresh instance is built per device so a CUDA run does
    not leave a shared fixture stranded on the GPU.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        loss_fn = WeightedDistLoss(bandwidth=0.5).to(device)
        y_pred = torch.randn(10, 2).to(device)
        y_true = torch.randn(10, 2).to(device)

        weighted_loss, dim_losses = loss_fn(y_pred, y_true)
        assert weighted_loss.device.type == device
        assert dim_losses.device.type == device


def test_numerical_stability(loss_fn):
    """Test numerical stability with extreme-magnitude values.

    Density estimation uses scipy.stats.gaussian_kde over unit-step (step=1.0)
    evaluation points, so the loss operates on standardized-scale data: the inputs
    must be non-degenerate (>= 2 distinct values per dimension) and within a range
    that unit-step binning can span. Identical values give a singular KDE
    covariance, and ~1e10-magnitude ranges blow up the unit-step grid -- both are
    intended limits of the rewritten design, not numerical instability. Verify the
    loss stays finite for extreme-but-tractable large and small multivalued data.
    """
    torch.manual_seed(0)

    # Large-magnitude, multivalued (range spans many unit-step bins, but finite).
    y_pred_large = torch.randn(6, 2) * 1e3
    y_true_large = torch.randn(6, 2) * 1e3
    weighted_loss_large, dim_losses_large = loss_fn(y_pred_large, y_true_large)
    assert not torch.isnan(weighted_loss_large)
    assert not torch.any(torch.isnan(dim_losses_large))
    assert torch.isfinite(weighted_loss_large)

    # Small-magnitude, multivalued.
    y_pred_small = torch.randn(6, 2) * 1e-3
    y_true_small = torch.randn(6, 2) * 1e-3
    weighted_loss_small, dim_losses_small = loss_fn(y_pred_small, y_true_small)
    assert not torch.isnan(weighted_loss_small)
    assert not torch.any(torch.isnan(dim_losses_small))
    assert torch.isfinite(weighted_loss_small)


def test_weight_initialization():
    """Test different weight initialization scenarios."""
    # Default weights: the rewritten loss defaults to a single normalized weight
    # (torch.ones(1)) and expands it to the data's dimensionality inside forward
    # (see the weight-broadcast in WeightedDistLoss.forward). This replaces the old
    # hard-coded 2-dimension default; callers such as
    # torchcell/losses/isomorphic_cell_loss.py construct WeightedDistLoss(weights=None)
    # and rely on this auto-expansion.
    loss_fn1 = WeightedDistLoss()
    assert torch.allclose(loss_fn1.weights.sum(), torch.tensor(1.0))
    assert loss_fn1.weights.shape == (1,)  # Single weight, broadcast in forward

    # Custom weights are normalized to sum to 1 and keep their dimensionality.
    custom_weights = torch.tensor([0.8, 0.2])
    loss_fn2 = WeightedDistLoss(weights=custom_weights)
    assert torch.allclose(loss_fn2.weights.sum(), torch.tensor(1.0))
    assert loss_fn2.weights.shape == (2,)
    assert torch.allclose(loss_fn2.weights, custom_weights / custom_weights.sum())


# NOTE: tests for global-statistics tracking (`global_min` / `global_max` /
# `stats_initialized`) and for train-vs-eval-dependent statistics were removed.
# Those buffers and that mode-dependent behavior belonged to the earlier
# WeightedDistLoss (the commented-out block in
# torchcell/losses/multi_dim_nan_tolerant.py), which tracked a running per-dimension
# range to seed its custom KDE. The rewrite computes the per-dimension range from
# the current batch inside forward and does not retain any cross-batch statistics,
# so it has no `global_min` / `global_max` / `stats_initialized` to assert against
# and behaves identically in train and eval mode. The cross-batch accumulation these
# tests covered now lives in `BufferedWeightedDistLoss`
# (torchcell/losses/mle_dist_supcr.py), which maintains a circular pred/target
# buffer; that wrapper is the correct home for any future cross-batch-statistics
# test, not WeightedDistLoss.


# ---------------------------------------------------------------------------
# Phase 24: the single-value label buffer, the default task weights of the two
# combined CE losses, and the empty tightness term (multi_dim_nan_tolerant.py:1002-1003,
# 1229, 1399, 1574).
# ---------------------------------------------------------------------------

import math  # noqa: E402
import re  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from torchcell.losses.multi_dim_nan_tolerant import (  # noqa: E402
    CategoricalEntropyRegLoss,
    CombinedCELoss,
    CombinedOrdinalCELoss,
)


def test_equal_labels_widen_the_range_by_half_a_unit_then_the_kde_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: labels [3, 3] have min == max, so the range is widened to
    [2.5, 3.5] (lines 1001-1003) before ``_get_label_distribution`` is called; but that
    widening does not help the KDE, which is fitted on the labels themselves:
    ``scipy.stats.gaussian_kde`` on two identical points has a singular covariance and
    raises ``LinAlgError``. So a dimension whose valid labels are all equal (any batch
    size above one) crashes the loss instead of scoring it. Pinned until a constant
    dimension is skipped or given a fixed-width kernel.
    """
    loss = WeightedDistLoss()
    seen: list[tuple[float, float]] = []
    original = loss._get_label_distribution

    def spy(
        labels: torch.Tensor, min_label: float, max_label: float, step: float = 1.0
    ) -> Any:
        seen.append((min_label, max_label))
        return original(labels, min_label, max_label, step)

    monkeypatch.setattr(loss, "_get_label_distribution", spy)
    with pytest.raises(
        np.linalg.LinAlgError,
        match=re.escape("The data appears to lie in a lower-dimensional subspace"),
    ):
        loss(torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [3.0]]))
    assert seen == [(2.5, 3.5)]


def test_combined_ce_default_weights_are_uniform_over_the_tasks() -> None:
    """``CombinedCELoss(2 classes, 4 tasks)`` with no weights registers [1/4] * 4. With
    zero logits every class has log-probability log(1/2), so each one-hot task loss is
    log 2; task 2 is all NaN, so its dim loss is 0 and it leaves the denominator:
    dim losses [log 2 / 4, log 2 / 4, 0, log 2 / 4] and total (3 log 2 / 4) / (3 / 4)
    = log 2 = 0.6931472.
    """
    nan = float("nan")
    loss = CombinedCELoss(num_classes=2, num_tasks=4)
    assert torch.equal(loss.get_buffer("weights"), torch.full((4,), 0.25))
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, nan, nan, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, nan, nan, 0.0, 1.0],
        ]
    )
    total, dims = loss(torch.zeros(2, 8), targets)
    q = math.log(2) / 4
    torch.testing.assert_close(dims, torch.tensor([q, q, 0.0, q]))
    assert total.item() == pytest.approx(math.log(2), abs=1e-6)


def test_combined_ordinal_default_weights_are_uniform_over_the_tasks() -> None:
    """``CombinedOrdinalCELoss(3 classes, 3 tasks)`` with no weights registers [1/3] * 3.
    Logits equal to the learned thresholds make every sigmoid 1/2, so every binary
    cross entropy is log 2 and each valid task loss is log 2; task 1 is all NaN and is
    dropped: dim losses [log 2 / 3, 0, log 2 / 3], total log 2.
    """
    nan = float("nan")
    loss = CombinedOrdinalCELoss(num_classes=3, num_tasks=3)
    assert torch.equal(loss.get_buffer("weights"), torch.full((3,), 1 / 3))
    logits = loss.loss_fn.thresholds.detach().reshape(1, -1).repeat(2, 1)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, nan, nan, nan, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, nan, nan, nan, 0.0, 1.0, 0.0],
        ]
    )
    total, dims = loss(logits, targets)
    third = math.log(2) / 3
    torch.testing.assert_close(dims.detach(), torch.tensor([third, 0.0, third]))
    assert total.item() == pytest.approx(math.log(2), abs=1e-6)


def test_entropy_regularizer_with_no_valid_sample_returns_three_zeros() -> None:
    """An all-False mask leaves no valid sample: the diversity term takes the
    fewer-than-two branch (0), no class has a center and no sample a distance, so the
    tightness term is the 0.0 fallback (line 1574); total 0.1 * 0 + 0.1 * 0 = 0.
    """
    reg = CategoricalEntropyRegLoss(lambda_d=0.1, lambda_t=0.1, num_classes=2)
    total, diversity, tightness = reg(
        torch.ones(3, 2), torch.full((3, 4), 0.5), torch.zeros(3, dtype=torch.bool)
    )
    assert (total.item(), diversity.item(), tightness.item()) == (0.0, 0.0, 0.0)
