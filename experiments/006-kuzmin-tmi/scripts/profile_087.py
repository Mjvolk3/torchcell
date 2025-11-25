# experiments/006-kuzmin-tmi/scripts/profile_087.py
"""
Experiment 087: Comprehensive Dataloader Profiling for All Graph Processing Methods.

Tests 5 different data loading approaches:
1. DANGO (Perturbation processor - minimal graph data)
2. Lazy Hetero (LazySubgraphRepresentation - full graph + masks)
3. NeighborSubgraph 1-hop
4. NeighborSubgraph 2-hop
5. NeighborSubgraph 3-hop

Measures iterations/sec, memory usage, and graph sizes for direct comparison.
"""

import os
import os.path as osp
import time
import argparse
import psutil
from dotenv import load_dotenv
from tqdm import tqdm

import torch
from torchcell.graph import SCerevisiaeGraph
from torchcell.datamodules import CellDataModule
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset
from torchcell.data.graph_processor import (
    NeighborSubgraphRepresentation,
    LazySubgraphRepresentation,
    Perturbation,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Profile different graph processing methods for experiment 087"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dango", "lazy", "neighbor"],
        default="lazy",
        help="Graph processing method: dango (Perturbation), lazy (LazySubgraph), neighbor (NeighborSubgraph)",
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="Number of hops for neighbor method (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,  # Will be set based on method
        help="Batch size for profiling (default: 64 for dango, 28 for others)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Number of samples in subset (default: 10000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of profiling steps (default: 100)",
    )
    args = parser.parse_args()

    # Load environment
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

    # Configuration from args
    METHOD = args.method
    NUM_HOPS = args.num_hops
    SUBSET_SIZE = args.subset_size
    MAX_STEPS = args.max_steps
    NUM_WORKERS = args.num_workers

    # Set default batch size based on method (matching 086 experiments)
    if args.batch_size is None:
        BATCH_SIZE = 64 if METHOD == "dango" else 28
    else:
        BATCH_SIZE = args.batch_size

    # Determine processor name for display
    if METHOD == "dango":
        processor_display = "Perturbation (DANGO)"
    elif METHOD == "lazy":
        processor_display = "LazySubgraphRepresentation"
    else:  # neighbor
        processor_display = f"NeighborSubgraphRepresentation ({NUM_HOPS}-hop)"

    print("=" * 80)
    print("EXPERIMENT 087: COMPREHENSIVE DATALOADER PROFILING")
    print("=" * 80)
    print(f"Method: {METHOD}")
    print(f"Graph Processor: {processor_display}")
    print("=" * 80)
    print()

    print(f"Configuration:")
    print(f"  method: {METHOD}")
    if METHOD == "neighbor":
        print(f"  num_hops: {NUM_HOPS}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  num_workers: {NUM_WORKERS}")
    print(f"  subset_size: {SUBSET_SIZE}")
    print(f"  max_steps: {MAX_STEPS}")
    print()

    # Initialize genome
    print("Initializing genome...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()

    # Initialize graph
    print("Initializing graph...")
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph with all 9 edge types (same as experiment 086)
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
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Load metabolism
    print("Loading metabolism...")
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    incidence_graphs = {"metabolism_bipartite": yeast_gem.bipartite_graph}

    # Load query
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()

    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Create dataset with appropriate graph processor
    print(f"Creating dataset with {processor_display}...")
    if METHOD == "dango":
        graph_processor = Perturbation()
    elif METHOD == "lazy":
        graph_processor = LazySubgraphRepresentation()
    else:  # neighbor
        graph_processor = NeighborSubgraphRepresentation(num_hops=NUM_HOPS)

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=gene_multigraph,
        incidence_graphs=incidence_graphs,
        node_embeddings=None,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
        transform=None,
    )

    # Normalization transform (same as 086)
    norm_configs = {"gene_interaction": {"strategy": "standard"}}
    normalizer = COOLabelNormalizationTransform(dataset, norm_configs)
    dataset.transform = normalizer

    print(f"Dataset length: {len(dataset)}")
    print()

    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=8,
        random_seed=42,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch=False,
    )
    cell_data_module.setup()

    # Subset Module (same as 086)
    print(f"Creating {SUBSET_SIZE}-sample subset...")
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch=False,
        seed=42,
        follow_batch=["perturbation_indices"] if METHOD != "dango" else [],
        gene_subsets={"metabolism": yeast_gem.gene_set},
    )
    perturbation_subset_data_module.setup()

    # Get dataloader
    train_loader = perturbation_subset_data_module.train_dataloader()

    print()
    print("=" * 80)
    print("STARTING PROFILING")
    print("=" * 80)
    print()

    # Warmup (skip first few batches)
    print("Warming up (10 batches)...")
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        _ = batch  # Just iterate

    print("Warmup complete. Starting profiling...")
    print()

    # Profile for MAX_STEPS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process = psutil.Process()

    times = []
    cpu_mem_usage = []
    gpu_mem_usage = []
    num_nodes_list = []
    num_edges_list = []
    step_count = 0

    for batch in tqdm(train_loader, total=MAX_STEPS, desc="Profiling"):
        start_time = time.time()

        # Track CPU memory before GPU transfer
        cpu_mem_mb = process.memory_info().rss / 1024 / 1024

        # Move batch to GPU (same as 086 dataloader profiling)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            batch = batch.to(device)
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            gpu_mem_usage.append(gpu_mem_mb)

        cpu_mem_usage.append(cpu_mem_mb)

        # Track subgraph size - different for dango vs hetero methods
        if METHOD == "dango":
            # DANGO: Full graph (6607 nodes), no edges in batch
            num_nodes_list.append(6607)
            num_edges_list.append(0)  # DANGO doesn't include graph edges in batch
        else:
            # Hetero methods: Track actual graph size from batch
            if hasattr(batch, "__getitem__") and "gene" in batch.node_types:
                # Track gene nodes
                if hasattr(batch["gene"], "num_nodes"):
                    num_nodes = batch["gene"].num_nodes
                    if isinstance(num_nodes, torch.Tensor):
                        if num_nodes.numel() > 1:
                            num_nodes = num_nodes.sum().item()
                        else:
                            num_nodes = num_nodes.item()
                    num_nodes_list.append(num_nodes)

                # Track total edges across all edge types
                total_edges = 0
                for edge_type in batch.edge_types:
                    if hasattr(batch[edge_type], "num_edges"):
                        num_edges = batch[edge_type].num_edges
                        if isinstance(num_edges, torch.Tensor):
                            if num_edges.numel() > 1:
                                num_edges = num_edges.sum().item()
                            else:
                                num_edges = num_edges.item()
                        total_edges += num_edges
                num_edges_list.append(total_edges)

        end_time = time.time()
        times.append(end_time - start_time)

        step_count += 1
        if step_count >= MAX_STEPS:
            break

    # Calculate statistics
    avg_time = sum(times) / len(times)
    avg_its = 1.0 / avg_time if avg_time > 0 else 0
    avg_cpu_mem = (
        sum(cpu_mem_usage) / len(cpu_mem_usage) if cpu_mem_usage else 0
    )
    avg_gpu_mem = (
        sum(gpu_mem_usage) / len(gpu_mem_usage) if gpu_mem_usage else 0
    )
    avg_num_nodes = (
        sum(num_nodes_list) / len(num_nodes_list) if num_nodes_list else 0
    )
    avg_num_edges = (
        sum(num_edges_list) / len(num_edges_list) if num_edges_list else 0
    )

    print()
    print("=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)
    print()
    print(f"Steps profiled: {len(times)}")
    print(f"Average time per batch: {avg_time:.4f} sec")
    print(f"Average iterations/sec: {avg_its:.3f} it/s")
    print()

    # Memory metrics
    print("=" * 80)
    print("MEMORY USAGE")
    print("=" * 80)
    print()
    print(f"Average CPU memory: {avg_cpu_mem:.1f} MB")
    if gpu_mem_usage:
        print(f"Average GPU memory per batch: {avg_gpu_mem:.1f} MB")
        # Estimate max batch size (assuming 80GB GPU, reserve 20GB for model/overhead)
        available_gpu_mem = 60 * 1024  # 60 GB in MB
        if avg_gpu_mem > 0:
            estimated_max_batch = int(
                (available_gpu_mem / avg_gpu_mem) * BATCH_SIZE
            )
            print(
                f"Estimated max batch size (60GB available): {estimated_max_batch}"
            )
        print()

    # Subgraph size metrics
    print("=" * 80)
    print("SUBGRAPH SIZE")
    print("=" * 80)
    print()
    print(f"Average nodes per batch: {avg_num_nodes:.1f}")
    print(f"Average edges per batch: {avg_num_edges:.1f}")
    if BATCH_SIZE > 0:
        print(f"Average nodes per sample: {avg_num_nodes/BATCH_SIZE:.1f}")
        print(f"Average edges per sample: {avg_num_edges/BATCH_SIZE:.1f}")
    print()

    # Comparison with baselines
    print("=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)
    print()
    print(f"  Current ({processor_display}): {avg_its:.3f} it/s")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Method: {METHOD}")
    print(f"Processor: {processor_display}")
    print(f"Iterations/sec: {avg_its:.3f}")
    print(f"GPU memory/batch: {avg_gpu_mem:.1f} MB")
    print(f"Nodes/sample: {avg_num_nodes/BATCH_SIZE:.1f}")
    print(f"Edges/sample: {avg_num_edges/BATCH_SIZE:.1f}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
