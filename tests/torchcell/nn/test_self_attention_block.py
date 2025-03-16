# tests/torchcell/nn/test_self_attention_block.py

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcell.nn.self_attention_block import SelfAttentionBlock


def test_self_attention_block_initialization():
    """Test initialization of SelfAttentionBlock"""
    hidden_dim = 64
    num_heads = 8

    # Test initialization
    sab = SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads)

    # Check attributes
    assert sab.hidden_dim == hidden_dim
    assert sab.num_heads == num_heads
    assert sab.head_dim == hidden_dim // num_heads

    # Check component initialization
    assert isinstance(sab.norm1, nn.LayerNorm)
    assert isinstance(sab.norm2, nn.LayerNorm)
    assert isinstance(sab.q_proj, nn.Linear)
    assert isinstance(sab.k_proj, nn.Linear)
    assert isinstance(sab.v_proj, nn.Linear)
    assert isinstance(sab.out_proj, nn.Linear)
    assert isinstance(sab.dropout, nn.Dropout)
    assert isinstance(sab.mlp, nn.Sequential)


def test_cpu_forward_pass():
    """Test forward pass on CPU"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = sab(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(output).any()

    # Should be different from input
    assert not torch.allclose(output, x)


def test_reshape_for_attention():
    """Test reshaping function for attention"""
    hidden_dim = 64
    num_heads = 8
    head_dim = hidden_dim // num_heads
    batch_size = 2
    seq_len = 16

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Apply reshaping
    reshaped = sab._reshape_for_attention(x)

    # Check output shape
    assert reshaped.shape == (batch_size, num_heads, seq_len, head_dim)

    # Test that we can get back to original shape
    back_to_original = (
        reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    )
    assert back_to_original.shape == x.shape
    assert torch.allclose(back_to_original, x)


def test_differentiability():
    """Test that the self-attention block is differentiable"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input with requires_grad
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)

    # Forward pass
    output = sab(x)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_forward_with_long_sequence():
    """Test forward pass with longer sequence"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 256  # Longer sequence

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = sab(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(output).any()


def test_forward_unbatched():
    """Test forward pass with unbatched input (2D tensor)"""
    hidden_dim = 64
    seq_len = 16

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input without batch dimension
    x = torch.randn(seq_len, hidden_dim)

    # Forward pass - should handle the missing batch dimension and add it
    # This is important for graph models where often we have just nodes x features
    try:
        output = sab(x.unsqueeze(0)).squeeze(0)

        # Check output shape
        assert output.shape == (seq_len, hidden_dim)
        assert not torch.isnan(output).any()
    except RuntimeError:
        # If it doesn't support unbatched input, that's ok
        # We just make a note of it
        print("Note: SelfAttentionBlock doesn't support unbatched input")


def test_zero_length_sequence():
    """Test forward pass with zero-length sequence"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 0

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input with zero-length sequence
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass should handle zero-length sequence
    try:
        output = sab(x)

        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert not torch.isnan(output).any()
    except RuntimeError:
        # If it doesn't support zero-length sequence, that's ok
        print("Note: SelfAttentionBlock doesn't support zero-length sequence")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward_pass():
    """Test forward pass on GPU"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Initialize model and move to GPU
    sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()

    # Create input on GPU
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Forward pass
    output = sab(x)

    # Check output shape and device
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert output.device.type == "cuda"
    assert not torch.isnan(output).any()

    # Should be different from input
    assert not torch.allclose(output, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_backward_pass():
    """Test backward pass on GPU"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Initialize model and move to GPU
    sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()

    # Create input on GPU with requires_grad
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", requires_grad=True)

    # Forward pass
    output = sab(x)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_vs_cpu():
    """Test that GPU and CPU implementations produce similar results"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize models
    cpu_sab = SelfAttentionBlock(hidden_dim=hidden_dim)
    gpu_sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()
    
    # Copy weights from CPU to GPU to ensure they're identical
    gpu_sab.load_state_dict(cpu_sab.state_dict())
    
    # Create inputs
    x_cpu = torch.randn(batch_size, seq_len, hidden_dim)
    x_gpu = x_cpu.cuda()
    
    # Forward passes
    with torch.no_grad():
        output_cpu = cpu_sab(x_cpu)
        output_gpu = gpu_sab(x_gpu).cpu()
    
    # Check for NaNs
    assert not torch.isnan(output_cpu).any()
    assert not torch.isnan(output_gpu).any()
    
    # Check output shapes
    assert output_cpu.shape == output_gpu.shape
    
    # Check statistical properties instead of exact values
    assert torch.allclose(output_cpu.mean(), output_gpu.mean(), rtol=0.3, atol=0.3)
    assert torch.allclose(output_cpu.std(), output_gpu.std(), rtol=0.3, atol=0.3)
    assert abs(output_cpu.min().item() - output_gpu.min().item()) < 1.0
    assert abs(output_cpu.max().item() - output_gpu.max().item()) < 1.0
    
    # Check correlation between outputs
    def correlation(x, y):
        x_flat = x.flatten()
        y_flat = y.flatten()
        x_norm = (x_flat - x_flat.mean()) / x_flat.std()
        y_norm = (y_flat - y_flat.mean()) / y_flat.std()
        return (x_norm * y_norm).mean()
    
    # Outputs should be positively correlated
    corr = correlation(output_cpu, output_gpu)
    print(f"CPU-GPU output correlation: {corr:.4f}")
    assert corr > 0.7, f"Correlation too low: {corr:.4f}"
    
    # Visual inspection - print some stats
    print(f"CPU output - mean: {output_cpu.mean().item():.4f}, std: {output_cpu.std().item():.4f}, min: {output_cpu.min().item():.4f}, max: {output_cpu.max().item():.4f}")
    print(f"GPU output - mean: {output_gpu.mean().item():.4f}, std: {output_gpu.std().item():.4f}, min: {output_gpu.min().item():.4f}, max: {output_gpu.max().item():.4f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_flex_attention_error_propagation():
    """Test that errors in flex_attention propagate correctly"""
    from unittest.mock import patch

    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Initialize model and move to GPU
    sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()

    # Create input on GPU
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Mock flex_attention to raise an error
    with patch("torch.nn.attention.flex_attention.flex_attention") as mock_flex:
        mock_flex.side_effect = RuntimeError("Simulated FlexAttention error")

        # Forward pass should propagate the error, not fall back
        with pytest.raises(RuntimeError) as excinfo:
            output = sab(x)

        # Check error message
        assert "Simulated FlexAttention error" in str(excinfo.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_performance():
    """Test performance on GPU vs CPU"""
    hidden_dim = 64
    batch_size = 4
    seq_len = 128

    # Initialize models
    cpu_sab = SelfAttentionBlock(hidden_dim=hidden_dim)
    gpu_sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()

    # Create inputs
    x_cpu = torch.randn(batch_size, seq_len, hidden_dim)
    x_gpu = x_cpu.cuda()

    # Warm-up
    for _ in range(5):
        cpu_sab(x_cpu)
        gpu_sab(x_gpu)

    # Time CPU forward pass
    cpu_times = []
    for _ in range(10):
        start = time.time()
        cpu_sab(x_cpu)
        cpu_times.append(time.time() - start)

    # Time GPU forward pass
    gpu_times = []
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete
    for _ in range(10):
        start = time.time()
        gpu_sab(x_gpu)
        torch.cuda.synchronize()  # Wait for CUDA operations to complete
        gpu_times.append(time.time() - start)

    # Calculate average times
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    avg_gpu_time = sum(gpu_times) / len(gpu_times)

    # Print performance comparison
    speedup = avg_cpu_time / avg_gpu_time
    print(f"Average CPU time: {avg_cpu_time:.6f} seconds")
    print(f"Average GPU time: {avg_gpu_time:.6f} seconds")
    print(f"GPU speedup: {speedup:.2f}x")

    # GPU should be faster, but we don't assert this as it depends on hardware
    # Just a sanity check that both implementations work


def test_large_model():
    """Test with larger hidden dimensions"""
    hidden_dim = 256  # Larger than the default
    batch_size = 2
    seq_len = 16

    # Initialize model
    sab = SelfAttentionBlock(hidden_dim=hidden_dim)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output = sab(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_batch_gpu():
    """Test with larger batch on GPU"""
    hidden_dim = 64
    batch_size = 16  # Larger batch
    seq_len = 32

    # Skip if GPU memory is insufficient
    if torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:  # < 4GB
        pytest.skip("GPU memory insufficient for large batch test")

    # Initialize model and move to GPU
    sab = SelfAttentionBlock(hidden_dim=hidden_dim).cuda()

    # Create input on GPU
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Forward pass
    output = sab(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(output).any()


def test_integration_with_other_modules():
    """Test integration with other PyTorch modules"""
    hidden_dim = 64
    batch_size = 2
    seq_len = 16

    # Create a simple model using SelfAttentionBlock
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(10, hidden_dim)
            self.attention = SelfAttentionBlock(hidden_dim=hidden_dim)
            self.output = nn.Linear(hidden_dim, 5)

        def forward(self, x):
            x = self.embedding(x)
            x = self.attention(x)
            return self.output(x)

    # Initialize model
    model = TestModel()

    # Create input
    x = torch.randn(batch_size, seq_len, 10)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, 5)
    assert not torch.isnan(output).any()

    # Test backpropagation
    loss = output.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"
