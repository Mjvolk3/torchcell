#!/usr/bin/env python
"""
Test script to verify that the vectorized GPU mask generation produces
identical results to the original loop-based implementation.

This ensures correctness of the optimization.
"""

import torch
import time
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from torchcell.models.gpu_edge_mask_generator import GPUEdgeMaskGenerator
from torchcell.datasets import Neo4jCellDataset
from torch_geometric.data import HeteroData


def create_test_cell_graph():
    """Create a simple test cell graph for verification."""
    cell_graph = HeteroData()

    # Create a simple gene-gene graph
    num_genes = 100
    cell_graph["gene"].num_nodes = num_genes

    # Create some edges
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]
    ], dtype=torch.long)

    cell_graph[("gene", "gigi", "gene")].edge_index = edge_index

    return cell_graph


def test_vectorized_vs_original(num_tests=10):
    """Compare vectorized and original implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create test graph
    cell_graph = create_test_cell_graph()

    # Initialize generator
    generator = GPUEdgeMaskGenerator(cell_graph, device)

    # Test with different batch sizes and perturbation patterns
    batch_sizes = [1, 4, 8, 16, 32]

    for batch_size in batch_sizes:
        print(f"\nTesting batch_size={batch_size}")

        for test_idx in range(num_tests):
            # Generate random perturbations (3 genes per sample typically)
            batch_pert_indices = []
            for _ in range(batch_size):
                num_pert = torch.randint(1, 5, (1,)).item()  # 1-4 perturbations
                pert_indices = torch.randperm(100)[:num_pert].to(device)
                batch_pert_indices.append(pert_indices)

            # Time original implementation
            start = time.time()
            masks_original = generator.generate_batch_masks(
                batch_pert_indices, batch_size
            )
            time_original = time.time() - start

            # Time vectorized implementation
            start = time.time()
            masks_vectorized = generator.generate_batch_masks_vectorized(
                batch_pert_indices, batch_size
            )
            time_vectorized = time.time() - start

            # Compare results
            all_match = True
            for edge_type in masks_original.keys():
                if not torch.allclose(masks_original[edge_type], masks_vectorized[edge_type]):
                    all_match = False
                    print(f"  ❌ Test {test_idx}: Mismatch for edge_type {edge_type}")
                    # Show differences
                    diff = (masks_original[edge_type] != masks_vectorized[edge_type]).sum()
                    total = masks_original[edge_type].numel()
                    print(f"     Differences: {diff}/{total} positions")
                    break

            if all_match:
                speedup = time_original / time_vectorized if time_vectorized > 0 else float('inf')
                print(f"  ✅ Test {test_idx}: Masks match! Speedup: {speedup:.2f}x "
                      f"(orig: {time_original*1000:.3f}ms, vec: {time_vectorized*1000:.3f}ms)")


def benchmark_performance():
    """Benchmark the performance improvement."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a more realistic graph (similar to actual yeast data)
    cell_graph = HeteroData()
    num_genes = 6607  # Actual yeast gene count
    cell_graph["gene"].num_nodes = num_genes

    # Create multiple edge types with realistic edge counts
    edge_types = [
        ("gene", "gigi", "gene"),
        ("gene", "regulatory", "gene"),
        ("gene", "physical", "gene"),
    ]

    for edge_type in edge_types:
        # Create random edges (simplified for testing)
        num_edges = torch.randint(50000, 200000, (1,)).item()
        src = torch.randint(0, num_genes, (num_edges,))
        dst = torch.randint(0, num_genes, (num_edges,))
        cell_graph[edge_type].edge_index = torch.stack([src, dst])

    # Initialize generator
    generator = GPUEdgeMaskGenerator(cell_graph, device)

    # Test with production batch size
    batch_size = 24  # From config
    num_iterations = 100

    print(f"\nBenchmarking with realistic graph:")
    print(f"  Genes: {num_genes}")
    print(f"  Edge types: {len(edge_types)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")

    # Generate test perturbations (3 genes per sample as in real data)
    test_batches = []
    for _ in range(num_iterations):
        batch_pert_indices = []
        for _ in range(batch_size):
            pert_indices = torch.randperm(num_genes)[:3].to(device)
            batch_pert_indices.append(pert_indices)
        test_batches.append(batch_pert_indices)

    # Warm up GPU
    for i in range(5):
        _ = generator.generate_batch_masks_vectorized(test_batches[0], batch_size)

    # Benchmark original
    torch.cuda.synchronize()
    start = time.time()
    for batch_pert_indices in test_batches:
        _ = generator.generate_batch_masks(batch_pert_indices, batch_size)
    torch.cuda.synchronize()
    time_original = time.time() - start

    # Benchmark vectorized
    torch.cuda.synchronize()
    start = time.time()
    for batch_pert_indices in test_batches:
        _ = generator.generate_batch_masks_vectorized(batch_pert_indices, batch_size)
    torch.cuda.synchronize()
    time_vectorized = time.time() - start

    print(f"\nResults:")
    print(f"  Original: {time_original:.3f}s total, {time_original/num_iterations*1000:.3f}ms per batch")
    print(f"  Vectorized: {time_vectorized:.3f}s total, {time_vectorized/num_iterations*1000:.3f}ms per batch")
    print(f"  Speedup: {time_original/time_vectorized:.2f}x")

    # Calculate expected impact on training
    ms_saved_per_batch = (time_original - time_vectorized) / num_iterations * 1000
    print(f"\nExpected training impact:")
    print(f"  Time saved per batch: {ms_saved_per_batch:.1f}ms")
    print(f"  At 0.42 it/s, batch takes: {1/0.42*1000:.0f}ms")
    print(f"  New expected speed: {1/(1/0.42 - ms_saved_per_batch/1000):.2f} it/s")


if __name__ == "__main__":
    print("="*60)
    print("Testing Vectorized GPU Mask Generation")
    print("="*60)

    # First verify correctness
    print("\n1. Correctness Test:")
    test_vectorized_vs_original()

    # Then benchmark performance
    print("\n2. Performance Benchmark:")
    benchmark_performance()

    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)