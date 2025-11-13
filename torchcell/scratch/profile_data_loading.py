#!/usr/bin/env python3
# torchcell/scratch/profile_data_loading.py
# [[torchcell.scratch.profile_data_loading]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/scratch/profile_data_loading

"""
Profile full data loading pipeline to identify remaining bottlenecks.

After optimizing graph processor to 3.76ms (8% of time), we need to find
where the remaining 40.88ms (92% of time) is being spent.
"""

import os
import os.path as osp
import random
import time
import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.loader import DataLoader
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import LazySubgraphRepresentation
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        print(f"  {self.name}: {self.elapsed_ms:.2f}ms")


def profile_dataset_creation():
    """Profile dataset creation and initialization."""
    print("\n" + "=" * 80)
    print("DATASET CREATION PROFILING")
    print("=" * 80 + "\n")

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with TimingContext("Total dataset creation"):
        with TimingContext("1. Initialize genome"):
            genome = SCerevisiaeGenome(
                genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
                go_root=osp.join(DATA_ROOT, "data/go"),
                overwrite=False,
            )
            genome.drop_empty_go()

        with TimingContext("2. Initialize graph"):
            graph = SCerevisiaeGraph(
                sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
                string_root=osp.join(DATA_ROOT, "data/string"),
                tflink_root=osp.join(DATA_ROOT, "data/tflink"),
                genome=genome,
            )

        with TimingContext("3. Build gene multigraph"):
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
            gene_multigraph = build_gene_multigraph(
                graph=graph, graph_names=graph_names
            )

        with TimingContext("4. Load metabolism model"):
            yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
            incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

        with TimingContext("5. Build node embeddings"):
            node_embeddings = NodeEmbeddingBuilder.build(
                embedding_names=["learnable"],
                data_root=DATA_ROOT,
                genome=genome,
                graph=graph,
            )

        with TimingContext("6. Create dataset"):
            with open(
                osp.join(
                    EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"
                ),
                "r",
            ) as f:
                query = f.read()

            dataset_root = osp.join(
                DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
            )

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
                graph_processor=LazySubgraphRepresentation(),
                transform=None,
            )

    print(f"\nDataset length: {len(dataset)}")
    return dataset


def profile_single_sample_loading(dataset):
    """Profile loading a single sample."""
    print("\n" + "=" * 80)
    print("SINGLE SAMPLE LOADING PROFILING")
    print("=" * 80 + "\n")

    # Profile __getitem__
    print("Loading sample 0 (cold)...")
    with TimingContext("  Cold load (first access)"):
        sample = dataset[0]

    print("\nLoading sample 0 (warm - should be cached)...")
    with TimingContext("  Warm load (cached)"):
        sample = dataset[0]

    print("\nSample structure:")
    print(f"  Node types: {sample.node_types}")
    print(f"  Edge types: {sample.edge_types}")
    if "gene" in sample.node_types:
        print(f"  Gene nodes: {sample['gene'].num_nodes}")
        if hasattr(sample["gene"], "x"):
            print(f"  Gene features: {sample['gene'].x.shape}")
    print()

    return sample


def profile_batch_loading(dataset, batch_size=32, num_samples=100):
    """Profile batch loading through DataLoader."""
    print("=" * 80)
    print("BATCH LOADING PROFILING")
    print("=" * 80 + "\n")

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num samples: {num_samples}")
    print(f"  Num workers: 0 (single process for accurate timing)")
    print()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single process for timing
        pin_memory=True,
    )

    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CUDA not available)")

    print()

    # Warmup
    print("Warmup (3 batches)...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        _ = batch.to(device)
    print()

    # Profile batch loading
    print(f"Profiling {num_samples} samples...")
    print()

    samples_processed = 0
    batch_create_times = []
    transfer_times = []
    total_times = []

    dataloader_iter = iter(dataloader)

    while samples_processed < num_samples:
        # Time getting batch from DataLoader (includes all __getitem__ calls and collation)
        batch_start = time.time()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break
        batch_create_time = (time.time() - batch_start) * 1000

        # Time GPU transfer
        transfer_start = time.time()
        batch_gpu = batch.to(device)
        transfer_time = (time.time() - transfer_start) * 1000

        total_time = batch_create_time + transfer_time

        batch_create_times.append(batch_create_time)
        transfer_times.append(transfer_time)
        total_times.append(total_time)

        samples_processed += batch.num_graphs

    # Statistics
    mean_batch_create = np.mean(batch_create_times)
    std_batch_create = np.std(batch_create_times)
    mean_transfer_time = np.mean(transfer_times)
    std_transfer_time = np.std(transfer_times)
    mean_total = np.mean(total_times)
    std_total = np.std(total_times)
    mean_per_sample = mean_total / batch_size

    print(f"Results:")
    print(f"  Samples processed: {samples_processed}")
    print(f"  Batches processed: {len(batch_create_times)}")
    print()
    print(f"Batch-level timing:")
    print(f"  Mean batch creation (CPU): {mean_batch_create:.2f}ms ± {std_batch_create:.2f}ms")
    print(
        f"  Mean GPU transfer:         {mean_transfer_time:.2f}ms ± {std_transfer_time:.2f}ms"
    )
    print(f"  Mean total time:           {mean_total:.2f}ms ± {std_total:.2f}ms")
    print()
    print(f"Per-sample estimates:")
    print(f"  Batch creation per sample: {mean_batch_create / batch_size:.2f}ms")
    print(f"  GPU transfer per sample:   {mean_transfer_time / batch_size:.2f}ms")
    print(f"  Total time per sample:     {mean_per_sample:.2f}ms")
    print()

    return {
        "mean_batch_create": mean_batch_create,
        "mean_transfer_time": mean_transfer_time,
        "mean_total": mean_total,
        "mean_per_sample": mean_per_sample,
    }


def main():
    print("=" * 80)
    print("DATA LOADING PIPELINE PROFILING")
    print("=" * 80)
    print()
    print("Objective: Identify where the remaining 40.88ms is spent")
    print("           (after optimizing graph processor to 3.76ms)")
    print()
    print("Components to profile:")
    print("  1. Dataset creation and initialization")
    print("  2. Single sample loading (__getitem__)")
    print("  3. Batch loading (DataLoader + GPU transfer)")
    print()

    # Profile dataset creation (one-time cost)
    dataset = profile_dataset_creation()

    # Profile single sample loading
    sample = profile_single_sample_loading(dataset)

    # Profile batch loading
    batch_stats = profile_batch_loading(dataset, batch_size=32, num_samples=1000)

    # Summary
    print("=" * 80)
    print("PROFILING SUMMARY")
    print("=" * 80)
    print()
    print("Key findings:")
    print(f"  1. Total per-sample time: {batch_stats['mean_per_sample']:.2f}ms")
    print(
        f"  2. Batch creation (CPU): {batch_stats['mean_batch_create'] / 32:.2f}ms per sample"
    )
    print(
        f"  3. GPU transfer: {batch_stats['mean_transfer_time'] / 32:.2f}ms per sample"
    )
    print()

    cpu_time = batch_stats['mean_batch_create'] / 32
    gpu_time = batch_stats['mean_transfer_time'] / 32
    graph_proc_time = 3.76  # From benchmark
    remaining_cpu = cpu_time - graph_proc_time

    print("Breakdown per sample:")
    print(f"  Total time:           {batch_stats['mean_per_sample']:.2f}ms")
    print(f"    ├─ Batch creation:  {cpu_time:.2f}ms (CPU work)")
    print(f"    │   ├─ Graph proc:  {graph_proc_time:.2f}ms (known from benchmark)")
    print(f"    │   └─ Other CPU:   {remaining_cpu:.2f}ms (UNKNOWN - to investigate)")
    print(f"    └─ GPU transfer:    {gpu_time:.2f}ms")
    print()

    if remaining_cpu > 0:
        print(f"⚠ Found {remaining_cpu:.2f}ms of unaccounted CPU work!")
        print()
        print("Candidates for this remaining time:")
        print("  - Dataset.__getitem__ overhead")
        print("  - Node embedding retrieval/processing")
        print("  - Phenotype COO tensor creation")
        print("  - HeteroData construction")
        print("  - DataLoader collation")
        print()

    print("Next steps:")
    print("  1. Review timing breakdown from Neo4jCellDataset.get() method")
    print("  2. Profile node embedding operations if needed")
    print("  3. Profile phenotype processing if needed")
    print("  4. Instrument HeteroData collation if needed")
    print()

    # Print detailed timing summaries
    from torchcell.profiling.timing import print_timing_summary

    print_timing_summary(
        title="Neo4jCellDataset - Method Timing",
        filter_class="Neo4jCellDataset"
    )
    print_timing_summary(
        title="LazySubgraphRepresentation - Method Timing",
        filter_class="LazySubgraphRepresentation"
    )


if __name__ == "__main__":
    main()
