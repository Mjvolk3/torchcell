#!/usr/bin/env python3
"""
Benchmark graph processors to verify whether graph processing is the bottleneck.

Compares:
1. SubgraphRepresentation (current implementation with Phase 1 optimization)
2. Perturbation (simpler processor used by Dango model)

This test measures data loading with GPU transfer to simulate real training.
"""

import time
import torch
import os
import os.path as osp
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.data.graph_processor import SubgraphRepresentation, Perturbation
from torchcell.graph import SCerevisiaeGraph
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph.graph import build_gene_multigraph

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def create_dataset(graph_processor, dataset_name):
    """Create a Neo4jCellDataset with the specified graph processor."""

    # Load the genome
    genome = SCerevisiaeGraph()

    # Build gene multigraph
    gene_multigraph = build_gene_multigraph(
        genome=genome,
        graphs=["physical", "regulatory"]
    )

    # Load metabolism model
    metabolism = YeastGEM()
    incidence_graphs = {
        "metabolism_bipartite": metabolism.bipartite
    }

    # Node embeddings
    node_embeddings = {"learnable": 64}

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Create dataset root
    dataset_root = osp.join(
        DATA_ROOT,
        f"data/torchcell/experiments/006-kuzmin-tmi/benchmark_{dataset_name}"
    )

    # Create dataset
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
        transform=None,
    )

    return dataset


def benchmark_graph_processor(processor_name, processor, dataset_name, num_samples=10000, batch_size=32):
    """Benchmark a graph processor on data loading."""

    print(f"\n{'='*80}")
    print(f"Benchmarking: {processor_name}")
    print(f"{'='*80}")

    # Create dataset with specified processor
    print("Creating dataset...")
    dataset = create_dataset(processor, dataset_name)
    print(f"Dataset length: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU only.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"Using device: {device}")

    # Warmup
    print("Warming up...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        batch = batch.to(device)

    # Benchmark
    print(f"Processing {num_samples} samples with batch_size={batch_size}...")

    start_time = time.time()
    samples_processed = 0

    for batch in dataloader:
        # Transfer to GPU like real training
        batch = batch.to(device)

        samples_processed += batch.num_graphs

        if samples_processed >= num_samples:
            break

    elapsed = time.time() - start_time

    print(f"\nResults:")
    print(f"  Samples processed: {samples_processed}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per sample: {elapsed/samples_processed*1000:.2f}ms")
    print(f"  Throughput: {samples_processed/elapsed:.1f} samples/sec")

    return elapsed, samples_processed


def main():
    print("="*80)
    print("Graph Processor Benchmark")
    print("="*80)
    print("\nObjective: Verify whether graph processing is the training bottleneck")
    print("\nComparing:")
    print("  1. SubgraphRepresentation (HeteroCell model - with Phase 1 optimization)")
    print("  2. Perturbation (Dango model)")
    print("\nConfiguration:")
    print("  - Graphs: physical, regulatory")
    print("  - Incidence graphs: metabolism_bipartite")
    print("  - Batch size: 32")
    print("  - Sample count: 10,000")
    print("  - GPU transfer: Yes")

    # Test 1: SubgraphRepresentation
    subgraph_time, subgraph_samples = benchmark_graph_processor(
        "SubgraphRepresentation (optimized)",
        SubgraphRepresentation(),
        "subgraph_repr",
        num_samples=10000,
        batch_size=32
    )

    # Test 2: Perturbation
    pert_time, pert_samples = benchmark_graph_processor(
        "Perturbation",
        Perturbation(),
        "perturbation",
        num_samples=10000,
        batch_size=32
    )

    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"\nSubgraphRepresentation: {subgraph_time:.2f}s ({subgraph_samples} samples)")
    print(f"Perturbation:           {pert_time:.2f}s ({pert_samples} samples)")
    print(f"\nRelative Performance:")

    if subgraph_time > pert_time * 1.1:  # 10% threshold for significance
        slowdown = subgraph_time / pert_time
        print(f"  SubgraphRepresentation is {slowdown:.2f}x SLOWER than Perturbation")
        print(f"\n  → Graph processing IS a bottleneck")
        print(f"  → Recommendation: Continue with Phase 2-5 optimizations")
        print(f"  → Clear cache and re-profile to see training benefit")
    elif pert_time > subgraph_time * 1.1:
        speedup = pert_time / subgraph_time
        print(f"  SubgraphRepresentation is {speedup:.2f}x FASTER than Perturbation")
        print(f"\n  → Graph processing is NOT the bottleneck")
        print(f"  → Recommendation: Pivot to optimizing GNN model forward pass")
        print(f"  → Target: torchcell/models/hetero_cell_bipartite_dango_gi.py")
    else:
        print(f"  SubgraphRepresentation and Perturbation have similar performance")
        print(f"  → Difference is within 10% margin")
        print(f"\n  → Graph processing is NOT the bottleneck")
        print(f"  → Recommendation: Pivot to optimizing GNN model forward pass")
        print(f"  → Target: torchcell/models/hetero_cell_bipartite_dango_gi.py")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
