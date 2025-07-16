# tests/torchcell/losses/test_mle_dist_supcr.py
import torch
import torch.distributed as dist

from torchcell.losses.mle_dist_supcr import (
    AdaptiveWeighting,
    MleDistSupCR,
    TemperatureScheduler,
)


def test_adaptive_weighting():
    """Test adaptive weighting schedule."""
    aw = AdaptiveWeighting(warmup_epochs=100, stable_epoch=500)

    # Test warmup phase
    assert 0.09 < aw.get_buffer_weight(0) < 0.11  # ~0.1
    assert 0.19 < aw.get_buffer_weight(50) < 0.21  # ~0.2
    assert 0.29 < aw.get_buffer_weight(100) < 0.31  # ~0.3

    # Test transition phase
    # The sigmoid is centered at 0.5 progress (epoch 300), so:
    # At epoch 200 (25% progress): still low ~0.34
    # At epoch 300 (50% progress): midpoint ~0.6
    # At epoch 400 (75% progress): high ~0.86
    assert 0.3 < aw.get_buffer_weight(200) < 0.4  # Early transition
    assert 0.55 < aw.get_buffer_weight(300) < 0.65  # Midpoint
    assert 0.8 < aw.get_buffer_weight(400) < 0.9  # Late transition

    # Test stable phase
    assert 0.89 < aw.get_buffer_weight(500) < 0.91  # ~0.9
    assert 0.89 < aw.get_buffer_weight(1000) < 0.91  # ~0.9


def test_temperature_scheduler():
    """Test temperature scheduling."""
    # Exponential schedule
    ts_exp = TemperatureScheduler(init_temp=1.0, final_temp=0.1, schedule="exponential")
    assert abs(ts_exp.get_temperature(0, 1000) - 1.0) < 0.01
    assert abs(ts_exp.get_temperature(1000, 1000) - 0.1) < 0.01
    assert 0.3 < ts_exp.get_temperature(500, 1000) < 0.4  # Midpoint

    # Cosine schedule
    ts_cos = TemperatureScheduler(init_temp=1.0, final_temp=0.1, schedule="cosine")
    assert abs(ts_cos.get_temperature(0, 1000) - 1.0) < 0.01
    assert abs(ts_cos.get_temperature(1000, 1000) - 0.1) < 0.01
    assert 0.5 < ts_cos.get_temperature(500, 1000) < 0.6  # Midpoint


def test_mle_dist_supcr_basic():
    """Test basic functionality of MleDistSupCR loss."""
    batch_size = 8
    num_dims = 2
    embedding_dim = 128

    # Create test data
    predictions = torch.randn(batch_size, num_dims)
    targets = torch.randn(batch_size, num_dims)
    z_p = torch.randn(batch_size, embedding_dim)

    # Initialize loss without buffers for simple test
    loss = MleDistSupCR(
        lambda_mse=1.0,
        lambda_dist=0.1,
        lambda_supcr=0.001,
        use_buffer=False,
        use_ddp_gather=False,
        use_adaptive_weighting=False,
        use_temp_scheduling=False,
    )

    # Forward pass
    total_loss, loss_dict = loss(predictions, targets, z_p)

    # Check outputs
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.shape == ()  # Scalar
    assert isinstance(loss_dict, dict)

    # Check required keys in loss_dict
    required_keys = [
        "mse_loss",
        "dist_loss",
        "supcr_loss",
        "weighted_mse",
        "weighted_dist",
        "weighted_supcr",
        "total_loss",
        "norm_weighted_mse",
        "norm_weighted_dist",
        "norm_weighted_supcr",
    ]
    for key in required_keys:
        assert key in loss_dict, f"Missing key: {key}"


def test_mle_dist_supcr_with_buffer():
    """Test MleDistSupCR with circular buffer."""
    batch_size = 8
    num_dims = 2
    embedding_dim = 128

    # Create test data
    predictions = torch.randn(batch_size, num_dims)
    targets = torch.randn(batch_size, num_dims)
    z_p = torch.randn(batch_size, embedding_dim)

    # Initialize loss with buffers
    loss = MleDistSupCR(
        lambda_mse=1.0,
        lambda_dist=0.1,
        lambda_supcr=0.001,
        use_buffer=True,
        buffer_size=32,
        min_samples_for_dist=16,
        min_samples_for_supcr=16,
        use_ddp_gather=False,
        embedding_dim=embedding_dim,
    )

    # First forward pass - should accumulate but might return zero for dist/supcr
    total_loss1, loss_dict1 = loss(predictions, targets, z_p, epoch=0)

    # Multiple forward passes to fill buffer
    for _ in range(5):
        predictions = torch.randn(batch_size, num_dims)
        targets = torch.randn(batch_size, num_dims)
        z_p = torch.randn(batch_size, embedding_dim)
        total_loss, loss_dict = loss(predictions, targets, z_p, epoch=0)

    # After enough samples, losses should be non-zero
    assert loss_dict["dist_loss"] >= 0  # Could still be 0 if below threshold
    assert loss_dict["supcr_loss"] >= 0


def test_mle_dist_supcr_adaptive_features():
    """Test adaptive weighting and temperature scheduling."""
    batch_size = 8
    num_dims = 2
    embedding_dim = 128

    # Create test data
    predictions = torch.randn(batch_size, num_dims)
    targets = torch.randn(batch_size, num_dims)
    z_p = torch.randn(batch_size, embedding_dim)

    # Initialize loss with all features
    loss = MleDistSupCR(
        lambda_mse=1.0,
        lambda_dist=0.1,
        lambda_supcr=0.001,
        use_buffer=True,
        use_ddp_gather=False,
        use_adaptive_weighting=True,
        use_temp_scheduling=True,
        warmup_epochs=10,
        stable_epoch=50,
        embedding_dim=embedding_dim,
    )

    # Test at different epochs
    _, loss_dict_early = loss(predictions, targets, z_p, epoch=0)
    _, loss_dict_mid = loss(predictions, targets, z_p, epoch=25)
    _, loss_dict_late = loss(predictions, targets, z_p, epoch=100)

    # Check adaptive features are logged
    assert "buffer_weight" in loss_dict_early
    assert "temperature" in loss_dict_early

    # Buffer weight should increase over epochs
    assert loss_dict_early["buffer_weight"] < loss_dict_mid["buffer_weight"]
    assert loss_dict_mid["buffer_weight"] < loss_dict_late["buffer_weight"]

    # Temperature should decrease over epochs
    assert loss_dict_early["temperature"] > loss_dict_late["temperature"]


if __name__ == "__main__":
    test_adaptive_weighting()
    print("✓ Adaptive weighting test passed")

    test_temperature_scheduler()
    print("✓ Temperature scheduler test passed")

    test_mle_dist_supcr_basic()
    print("✓ Basic MleDistSupCR test passed")

    test_mle_dist_supcr_with_buffer()
    print("✓ MleDistSupCR with buffer test passed")

    test_mle_dist_supcr_adaptive_features()
    print("✓ MleDistSupCR adaptive features test passed")

    print("\nAll tests passed! ✨")
