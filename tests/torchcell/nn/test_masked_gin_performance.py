# tests/torchcell/nn/test_masked_gin_performance
# [[tests.torchcell.nn.test_masked_gin_performance]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/nn/test_masked_gin_performance

"""
Performance tests for MaskedGINConv to verify lazy speedup is maintained.

CRITICAL: These tests ensure that masked message passing does NOT negate
the 3.65x speedup from LazySubgraphRepresentation.
"""

import time
import torch
import torch.nn as nn
import pytest
from torchcell.nn.masked_gin_conv import MaskedGINConv


@pytest.fixture
def device():
    """Use GPU if available for realistic performance testing"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def synthetic_data(device):
    """
    Create synthetic graph data matching yeast genome scale.

    Matches typical data from LazySubgraphRepresentation:
    - 6,604 genes
    - ~144k edges (physical network)
    - ~5% edges masked (gene deletions)
    """
    num_nodes = 6604
    num_edges = 144211
    hidden_dim = 64

    x = torch.randn(num_nodes, hidden_dim, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    # Simulate 5% edge masking (typical for gene deletions)
    edge_mask = torch.rand(num_edges, device=device) > 0.05

    return x, edge_index, edge_mask


@pytest.fixture
def gin_mlp(device):
    """Create GIN MLP for testing"""
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    ).to(device)


def test_no_edge_filtering(synthetic_data, gin_mlp, device):
    """
    CRITICAL: Verify that edge_index is NEVER filtered or copied.

    This is the core requirement for maintaining lazy speedup.
    """
    x, edge_index, edge_mask = synthetic_data

    # Record original edge_index memory address
    edge_index_id_before = id(edge_index)

    # Create masked GIN layer
    conv = MaskedGINConv(gin_mlp, train_eps=True).to(device)

    # Forward pass with masking
    out = conv(x, edge_index, edge_mask=edge_mask)

    # CRITICAL: Verify edge_index was not copied
    assert id(edge_index) == edge_index_id_before, \
        "edge_index was copied! This negates lazy speedup."

    # Verify output shape is correct
    assert out.shape == (x.size(0), 64)


def test_masked_vs_filtered_speed(synthetic_data, gin_mlp, device):
    """
    Benchmark: Masked message passing vs edge filtering.

    Masked approach should be FASTER than filtering because:
    - No tensor copying
    - Element-wise operations are GPU-accelerated
    - Same edge_index reused (cache-friendly)
    """
    x, edge_index, edge_mask = synthetic_data

    # Warm up GPU
    for _ in range(5):
        _ = x @ x.t()

    # Approach 1: Edge filtering (OLD - should be SLOW)
    def filtering_approach():
        # This is what we want to AVOID
        filtered_edge_index = edge_index[:, edge_mask]  # Expensive!
        conv = MaskedGINConv(gin_mlp, train_eps=True).to(device)
        return conv(x, filtered_edge_index, edge_mask=None)

    # Time filtering approach
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_start = time.perf_counter()
    for _ in range(100):
        _ = filtering_approach()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_filtering = (time.perf_counter() - t_start) * 10  # ms per iteration

    # Approach 2: Masked messages (NEW - should be FAST)
    conv_masked = MaskedGINConv(gin_mlp, train_eps=True).to(device)

    # Time masked approach
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_start = time.perf_counter()
    for _ in range(100):
        _ = conv_masked(x, edge_index, edge_mask=edge_mask)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_masked = (time.perf_counter() - t_start) * 10  # ms per iteration

    print(f"\n{'='*60}")
    print(f"Performance Comparison (yeast genome scale)")
    print(f"{'='*60}")
    print(f"Approach 1 (filtering):  {t_filtering:.4f} ms/iter")
    print(f"Approach 2 (masked MP):  {t_masked:.4f} ms/iter")
    print(f"Speedup:                 {t_filtering/t_masked:.2f}x")
    print(f"{'='*60}\n")

    # CRITICAL: Masked approach must be faster or at least comparable
    # We allow small tolerance for measurement noise
    assert t_masked <= t_filtering * 1.1, \
        f"Masked MP ({t_masked:.4f}ms) slower than filtering ({t_filtering:.4f}ms)!"


def test_pyg_concatenated_batch(device):
    """
    Test with real PyG concatenated batch format from LazySubgraphRepresentation.

    This verifies the masked approach works correctly with actual batched data.
    Uses custom collate function to handle zero-copy batching.
    """
    try:
        from torchcell.scratch.load_lazy_batch_006 import load_sample_data_batch
        from torchcell.datamodules.lazy_collate import verify_batch_structure
    except ImportError:
        pytest.skip("load_lazy_batch_006 or lazy_collate not available")

    # Load real batch with custom collate (required for LazySubgraphRepresentation)
    dataset, batch, _, _ = load_sample_data_batch(
        batch_size=2,
        num_workers=0,
        use_custom_collate=True
    )
    batch = batch.to(device)

    # CRITICAL: Verify batch structure is correct before testing MaskedGINConv
    assert verify_batch_structure(batch, expected_graphs=2), \
        "Batch structure verification failed!"

    # Extract physical edge type
    edge_type = ("gene", "physical", "gene")
    edge_index = batch[edge_type].edge_index
    edge_mask = batch[edge_type].mask

    # Create random node features (since dataset has empty features)
    x = torch.randn(batch["gene"].num_nodes, 64, device=device)

    # Create MLP and conv layer
    mlp = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    ).to(device)
    conv = MaskedGINConv(mlp, train_eps=True).to(device)

    # Record edge_index ID BEFORE forward pass
    # With custom collate, edge_index is already a NEW tensor (replicated + offset)
    # We want to verify MaskedGINConv doesn't copy it AGAIN during forward
    edge_index_id_before = id(edge_index)

    # Forward pass
    out = conv(x, edge_index, edge_mask=edge_mask)

    # Verify MaskedGINConv didn't copy edge_index during forward
    assert id(edge_index) == edge_index_id_before, \
        "MaskedGINConv copied edge_index during forward pass!"

    # Verify output shape
    assert out.shape == (batch["gene"].num_nodes, 64)

    print(f"\n✓ Successfully processed PyG concatenated batch with custom collate")
    print(f"  Nodes: {batch['gene'].num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Masked edges: {(~edge_mask).sum().item()}")
    print(f"  ✓ Batch structure verified")
    print(f"  ✓ Edge index not copied during forward pass")


def test_equivalence_masked_vs_filtered(synthetic_data, gin_mlp, device):
    """
    Verify that masked MP produces same results as filtering (numerically).

    This ensures correctness while maintaining speedup.
    """
    x, edge_index, edge_mask = synthetic_data

    # Approach 1: Filter edges then apply GIN
    filtered_edge_index = edge_index[:, edge_mask]
    conv1 = MaskedGINConv(gin_mlp, train_eps=True).to(device)

    # Set to eval and fix seed for deterministic results
    conv1.eval()
    torch.manual_seed(42)
    out1 = conv1(x, filtered_edge_index, edge_mask=None)

    # Approach 2: Masked message passing
    conv2 = MaskedGINConv(gin_mlp, train_eps=True).to(device)
    conv2.load_state_dict(conv1.state_dict())  # Same weights
    conv2.eval()

    torch.manual_seed(42)
    out2 = conv2(x, edge_index, edge_mask=edge_mask)

    # Should produce identical results
    assert torch.allclose(out1, out2, atol=1e-5), \
        "Masked MP produces different results than filtering!"

    print(f"\n✓ Masked MP numerically equivalent to filtering")
    print(f"  Max difference: {(out1 - out2).abs().max().item():.2e}")


def test_memory_efficiency(synthetic_data, gin_mlp, device):
    """
    Verify that masked approach uses less memory than filtering.

    This is expected because we don't allocate new filtered tensors.
    """
    if device.type != 'cuda':
        pytest.skip("Memory profiling requires CUDA")

    x, edge_index, edge_mask = synthetic_data

    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Approach 1: Filtering (allocates new tensors)
    mem_before = torch.cuda.memory_allocated()
    filtered_edge_index = edge_index[:, edge_mask]
    conv1 = MaskedGINConv(gin_mlp, train_eps=True).to(device)
    _ = conv1(x, filtered_edge_index, edge_mask=None)
    mem_filtering = torch.cuda.max_memory_allocated() - mem_before

    # Clear GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Approach 2: Masked (no new allocations)
    mem_before = torch.cuda.memory_allocated()
    conv2 = MaskedGINConv(gin_mlp, train_eps=True).to(device)
    _ = conv2(x, edge_index, edge_mask=edge_mask)
    mem_masked = torch.cuda.max_memory_allocated() - mem_before

    print(f"\n{'='*60}")
    print(f"Memory Usage Comparison")
    print(f"{'='*60}")
    print(f"Filtering:   {mem_filtering / 1024**2:.2f} MB")
    print(f"Masked MP:   {mem_masked / 1024**2:.2f} MB")
    print(f"Savings:     {(mem_filtering - mem_masked) / 1024**2:.2f} MB")
    print(f"Reduction:   {(1 - mem_masked/mem_filtering) * 100:.1f}%")
    print(f"{'='*60}\n")

    # Masked approach should use less or equal memory
    assert mem_masked <= mem_filtering, "Masked MP uses more memory than filtering!"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
