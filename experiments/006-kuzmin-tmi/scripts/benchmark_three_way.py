#!/usr/bin/env python3
"""
Three-way benchmark: Perturbation vs LazySubgraphRepresentation vs SubgraphRepresentation.

Compares the simplest processor (Perturbation) against the optimized (Lazy) and baseline (Subgraph).
"""

import time
import torch
import os
import os.path as osp
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from torchcell.data import (
    Neo4jCellDataset,
    MeanExperimentDeduplicator,
    GenotypeAggregator,
)
from torchcell.data.graph_processor import (
    Perturbation,
    SubgraphRepresentation,
    LazySubgraphRepresentation,
)
from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph.graph import build_gene_multigraph
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.profiling.timing import print_timing_summary

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def create_dataset(graph_processor, processor_name):
    """Create dataset for specified processor."""

    # Initialize genome
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()

    # Initialize graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Perturbation processor doesn't use graphs, but we include them for other processors
    if processor_name == "Perturbation":
        # Minimal configuration for Perturbation (no edges needed)
        gene_multigraph = None
        incidence_graphs = None
        node_embeddings = None
    else:
        # Full HeteroCell configuration for Subgraph and Lazy
        graph_names = [
            "physical",
            "regulatory",
            "tflink",
            "string12_0_neighborhood",
            "string12_0_fusion",
            "string12_0_cooccurence",
            "string12_0_coexpression",
            "string12_0_experimental",
            "string12_0_database",
        ]

        # Build gene multigraph
        gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

        # Load metabolism model
        yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
        incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

        # Node embeddings
        node_embeddings = NodeEmbeddingBuilder.build(
            embedding_names=["learnable"],
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        )

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Dataset root
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
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


def benchmark_processor(
    processor_name, processor, num_samples=10000, batch_size=32, num_workers=2
):
    """Benchmark a graph processor on data loading."""

    print(f"\n{'='*80}")
    print(f"Benchmarking: {processor_name}")
    print(f"{'='*80}")

    # Create dataset
    print("Creating dataset...")
    dataset = create_dataset(processor, processor_name)
    print(f"Dataset length: {len(dataset)}")

    # Pre-build cache if this is LazySubgraphRepresentation (one-time cost)
    if isinstance(processor, LazySubgraphRepresentation):
        print("Pre-building incidence cache (one-time cost)...")
        cache_stats = processor.build_cache(dataset.cell_graph)
        print(f"  Cache build time: {cache_stats['total_time_ms']:.2f}ms")
        print(f"  Edge types cached: {cache_stats['num_edge_types']}")
        print(f"  Total edges: {cache_stats['total_edges']:,}")
        print("  → This cost is amortized over entire training run")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Check GPU availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running on CPU only.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
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
    print("=" * 80)
    print("Three-Way Graph Processor Benchmark")
    print("=" * 80)
    print("\nObjective: Compare Perturbation (simplest) vs LazySubgraphRepresentation")
    print("           (optimized) vs SubgraphRepresentation (baseline)")
    print("\nProcessors:")
    print("  1. Perturbation (Dango)")
    print("     - Simplest: only node masks, no edge processing")
    print("     - Theoretical minimum for perturbation-only tasks")
    print("  2. LazySubgraphRepresentation (Phase 3+4 optimized)")
    print("     - Zero-copy: full graphs + boolean masks")
    print("     - Handles edges but avoids tensor allocation")
    print("  3. SubgraphRepresentation (baseline)")
    print("     - Expensive: filters and relabels all tensors")
    print("     - Original implementation before optimization")
    print("\nConfiguration:")
    print("  - Perturbation: No graphs, no embeddings")
    print("  - Lazy + Subgraph: Full HeteroCell (9 edge types + metabolism)")
    print("  - Batch size: 32")
    print("  - Sample count: 1,000")
    print("  - Num workers: 0 (for timing collection)")
    print("  - GPU transfer: Yes")

    # Test 1: Perturbation (theoretical minimum)
    pert_time, pert_samples = benchmark_processor(
        "Perturbation (Simplest - No Edges)",
        Perturbation(),
        num_samples=1000,
        batch_size=32,
        num_workers=0,  # Required for timing collection
    )

    # Test 2: LazySubgraphRepresentation (optimized)
    lazy_time, lazy_samples = benchmark_processor(
        "LazySubgraphRepresentation (Optimized Zero-Copy)",
        LazySubgraphRepresentation(),
        num_samples=1000,
        batch_size=32,
        num_workers=0,  # Required for timing collection
    )

    # Test 3: SubgraphRepresentation (baseline)
    subgraph_time, subgraph_samples = benchmark_processor(
        "SubgraphRepresentation (Baseline)",
        SubgraphRepresentation(),
        num_samples=1000,
        batch_size=32,
        num_workers=0,  # Required for timing collection
    )

    # Summary
    print(f"\n{'='*80}")
    print("THREE-WAY BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"\n1. Perturbation (Simplest):              {pert_time:.2f}s ({pert_samples} samples)")
    print(
        f"2. LazySubgraphRepresentation (Optimized): {lazy_time:.2f}s ({lazy_samples} samples)"
    )
    print(
        f"3. SubgraphRepresentation (Baseline):      {subgraph_time:.2f}s ({subgraph_samples} samples)"
    )

    print(f"\nPer-sample time:")
    print(f"  Perturbation:              {pert_time/pert_samples*1000:.2f}ms/sample")
    print(
        f"  LazySubgraphRepresentation: {lazy_time/lazy_samples*1000:.2f}ms/sample"
    )
    print(
        f"  SubgraphRepresentation:     {subgraph_time/subgraph_samples*1000:.2f}ms/sample"
    )

    # Speedup comparisons
    print(f"\nSpeedup Analysis:")
    print(f"\nLazy vs Subgraph (optimization achievement):")
    speedup_lazy_vs_subgraph = subgraph_time / lazy_time
    reduction = (1 - lazy_time / subgraph_time) * 100
    print(
        f"  LazySubgraphRepresentation is {speedup_lazy_vs_subgraph:.2f}x FASTER than baseline"
    )
    print(f"  Time reduction: {reduction:.1f}%")

    print(f"\nPerturbation vs Lazy (remaining gap):")
    gap_pert_vs_lazy = lazy_time / pert_time
    overhead_pct = ((lazy_time / pert_time) - 1) * 100
    print(
        f"  LazySubgraphRepresentation is {gap_pert_vs_lazy:.2f}x SLOWER than Perturbation"
    )
    print(f"  Graph processing overhead: {overhead_pct:.1f}%")

    print(f"\nPerturbation vs Subgraph (total opportunity):")
    gap_pert_vs_subgraph = subgraph_time / pert_time
    print(
        f"  SubgraphRepresentation is {gap_pert_vs_subgraph:.2f}x SLOWER than Perturbation"
    )

    # Overhead breakdown
    pert_ms = pert_time / pert_samples * 1000
    lazy_ms = lazy_time / lazy_samples * 1000
    subgraph_ms = subgraph_time / subgraph_samples * 1000

    print(f"\nOverhead Breakdown (per sample):")
    print(f"  Perturbation baseline:     {pert_ms:.2f}ms (node masks + phenotype only)")
    print(
        f"  Lazy graph overhead:        {lazy_ms - pert_ms:.2f}ms (edge masks + zero-copy)"
    )
    print(
        f"  Subgraph graph overhead:    {subgraph_ms - pert_ms:.2f}ms (edge filtering + relabeling)"
    )

    print(f"\nKey Findings:")
    print(
        f"  ✓ Optimization reduced graph overhead by {(1 - (lazy_ms-pert_ms)/(subgraph_ms-pert_ms))*100:.1f}%"
    )
    print(
        f"  ✓ Lazy is now only {lazy_ms/pert_ms:.1f}x slower than theoretical minimum"
    )
    print(
        f"  → Remaining overhead is from necessary edge mask computation + zero-copy references"
    )

    print(f"{'='*80}\n")

    # Print timing breakdown if profiling is enabled
    print_timing_summary(
        title="Perturbation - Method Timing", filter_class="Perturbation"
    )
    print_timing_summary(
        title="LazySubgraphRepresentation - Method Timing",
        filter_class="LazySubgraphRepresentation",
    )
    print_timing_summary(
        title="SubgraphRepresentation - Method Timing",
        filter_class="SubgraphRepresentation",
    )


if __name__ == "__main__":
    main()
