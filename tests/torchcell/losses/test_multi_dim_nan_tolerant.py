from typing import Tuple

import numpy as np
import pytest
import torch

from torchcell.losses.multi_dim_nan_tolerant import WeightedDistLoss


@pytest.fixture
def loss_fn():
    """Fixture to create loss function instance."""
    return WeightedDistLoss(num_bins=50, bandwidth=0.5)


@pytest.fixture
def weighted_loss_fn():
    """Fixture to create weighted loss function instance."""
    weights = torch.tensor([0.7, 0.3])  # Asymmetric weights
    return WeightedDistLoss(num_bins=50, bandwidth=0.5, weights=weights)


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
    """Test handling of NaN values."""
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.2, float("nan")], [float("nan"), 3.8]])

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


def test_kde(loss_fn):
    """Test KDE functionality."""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    eval_points = torch.linspace(0, 5, 10)

    density = loss_fn.kde(x, eval_points)
    assert len(density) == len(eval_points)
    assert torch.all(density >= 0)
    assert torch.allclose(density.sum(), torch.tensor(1.0), atol=1e-6)


def test_pseudo_label_generation(loss_fn):
    """Test pseudo-label generation."""
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
    batch_size = 5

    pseudo_labels = loss_fn.generate_pseudo_labels(y_true, batch_size)
    assert len(pseudo_labels) == batch_size
    assert torch.all(pseudo_labels >= y_true.min())
    assert torch.all(pseudo_labels <= y_true.max())


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


def test_device_compatibility(loss_fn):
    """Test that loss works on both CPU and CUDA if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        y_pred = torch.randn(10, 2).to(device)
        y_true = torch.randn(10, 2).to(device)

        weighted_loss, dim_losses = loss_fn(y_pred, y_true)
        assert weighted_loss.device.type == device
        assert dim_losses.device.type == device


def test_numerical_stability(loss_fn):
    """Test numerical stability with extreme values."""
    # Test with very large values
    y_pred_large = torch.tensor([[1e10, 1e10], [1e10, 1e10]])
    y_true_large = torch.tensor([[1e10, 1e10], [1e10, 1e10]])
    weighted_loss_large, dim_losses_large = loss_fn(y_pred_large, y_true_large)
    assert not torch.isnan(weighted_loss_large)
    assert not torch.any(torch.isnan(dim_losses_large))

    # Test with very small values
    y_pred_small = torch.tensor([[1e-10, 1e-10], [1e-10, 1e-10]])
    y_true_small = torch.tensor([[1e-10, 1e-10], [1e-10, 1e-10]])
    weighted_loss_small, dim_losses_small = loss_fn(y_pred_small, y_true_small)
    assert not torch.isnan(weighted_loss_small)
    assert not torch.any(torch.isnan(dim_losses_small))


def test_weight_initialization():
    """Test different weight initialization scenarios."""
    # Default weights
    loss_fn1 = WeightedDistLoss()
    assert torch.allclose(loss_fn1.weights.sum(), torch.tensor(1.0))
    assert loss_fn1.weights.shape == (2,)  # Default 2 dimensions

    # Custom weights
    custom_weights = torch.tensor([0.8, 0.2])
    loss_fn2 = WeightedDistLoss(weights=custom_weights)
    assert torch.allclose(loss_fn2.weights.sum(), torch.tensor(1.0))
    assert torch.allclose(loss_fn2.weights, custom_weights / custom_weights.sum())
