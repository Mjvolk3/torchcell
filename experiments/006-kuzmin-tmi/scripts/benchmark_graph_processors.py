#!/usr/bin/env python3
"""
Benchmark graph processors to verify whether graph processing is the bottleneck.

Compares:
1. SubgraphRepresentation (HeteroCell model - with Phase 1 optimization)
   - Uses production graphs from hetero_cell_bipartite_dango_gi_cabbi_056.yaml
2. Perturbation (Dango model)
   - Uses production graphs from dango_kuzmin2018_tmi_string12_0.yaml

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
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph.graph import build_gene_multigraph
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.datamodules import CellDataModule
from torchcell.profiling.timing import print_timing_summary

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def create_dataset_hetero(graph_processor):
    """Create dataset for HeteroCell model (SubgraphRepresentation)."""

    # Initialize genome - following hetero_cell_bipartite_dango_gi.py
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

    # Graph names from hetero_cell_bipartite_dango_gi_cabbi_056.yaml
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
    gene_multigraph = build_gene_multigraph(
        graph=graph,
        graph_names=graph_names
    )

    # Load metabolism model - HeteroCell uses bipartite
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {
        "metabolism_bipartite": yeast_gem.bipartite_graph
    }

    # Node embeddings - HeteroCell uses learnable embeddings
    # Following the exact pattern from hetero_cell_bipartite_dango_gi.py
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

    # Use same dataset root as production
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset - using genome.gene_set (not graph.gene_set)
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


def create_dataset_dango(graph_processor):
    """Create dataset for Dango model (Perturbation)."""

    # Initialize genome - following dango.py
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

    # Graph names from dango_kuzmin2018_tmi_string12_0.yaml
    graph_names = [
        "string12_0_neighborhood",
        "string12_0_fusion",
        "string12_0_cooccurence",
        "string12_0_coexpression",
        "string12_0_experimental",
        "string12_0_database",
    ]

    # Build gene multigraph
    gene_multigraph = build_gene_multigraph(
        graph=graph,
        graph_names=graph_names
    )

    # Dango doesn't use incidence graphs
    incidence_graphs = None

    # Dango doesn't use node embeddings
    node_embeddings = None

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    # Use same dataset root as production
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset - using genome.gene_set
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


def benchmark_graph_processor(processor_name, processor, dataset_creator, num_samples=10000, batch_size=32, num_workers=2):
    """Benchmark a graph processor on data loading."""

    print(f"\n{'='*80}")
    print(f"Benchmarking: {processor_name}")
    print(f"{'='*80}")

    # Create dataset with specified processor
    print("Creating dataset...")
    dataset = dataset_creator(processor)
    print(f"Dataset length: {len(dataset)}")

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
    print("     - Graphs: physical, regulatory, tflink, string12_0 (all 6 channels)")
    print("     - Incidence graphs: metabolism_bipartite")
    print("     - Node embeddings: learnable (64 channels)")
    print("  2. Perturbation (Dango model)")
    print("     - Graphs: string12_0 (all 6 channels)")
    print("     - No incidence graphs")
    print("     - No node embeddings")
    print("\nConfiguration:")
    print("  - Batch size: 32")
    print("  - Sample count: 1,000")
    print("  - Num workers: 2")
    print("  - GPU transfer: Yes")

    # Test 1: SubgraphRepresentation with HeteroCell configuration
    # TEMPORARY: num_workers=0 to test timing collection
    subgraph_time, subgraph_samples = benchmark_graph_processor(
        "SubgraphRepresentation (HeteroCell)",
        SubgraphRepresentation(),
        create_dataset_hetero,
        num_samples=1000,
        batch_size=32,
        num_workers=0  # Changed from 2 to 0 for timing test
    )

    # Test 2: Perturbation with Dango configuration
    pert_time, pert_samples = benchmark_graph_processor(
        "Perturbation (Dango)",
        Perturbation(),
        create_dataset_dango,
        num_samples=1000,
        batch_size=32,
        num_workers=2
    )

    # Summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"\nSubgraphRepresentation (HeteroCell): {subgraph_time:.2f}s ({subgraph_samples} samples)")
    print(f"Perturbation (Dango):                {pert_time:.2f}s ({pert_samples} samples)")
    print(f"\nPer-sample time:")
    print(f"  SubgraphRepresentation: {subgraph_time/subgraph_samples*1000:.2f}ms/sample")
    print(f"  Perturbation:           {pert_time/pert_samples*1000:.2f}ms/sample")
    print(f"\nRelative Performance:")

    if subgraph_time > pert_time * 1.1:  # 10% threshold for significance
        slowdown = subgraph_time / pert_time
        print(f"  SubgraphRepresentation is {slowdown:.2f}x SLOWER than Perturbation")
        print(f"\n  ✓ Graph processing IS a bottleneck")
    elif pert_time > subgraph_time * 1.1:
        speedup = pert_time / subgraph_time
        print(f"  SubgraphRepresentation is {speedup:.2f}x FASTER than Perturbation")
        print(f"\n  ✓ Graph processing is NOT the bottleneck")
    else:
        print(f"  SubgraphRepresentation and Perturbation have similar performance")
        print(f"  → Difference is within 10% margin")
        print(f"\n  ✓ Graph processing is NOT the bottleneck")

    print(f"{'='*80}\n")

    # Print timing breakdown if profiling is enabled
    print_timing_summary("SubgraphRepresentation Timing Profile")


if __name__ == "__main__":
    main()
