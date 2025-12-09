# experiments/006-kuzmin-tmi/scripts/profile_neighbor_subgraph_087a.py
"""
Experiment 087a: Profile NeighborSubgraphRepresentation dataloader speed.

Measures iterations/sec for data loading only (no model).
Compares with Experiment 086a results to determine if neighbor sampling is faster.
Tracks memory usage and subgraph sizes to estimate optimal batch sizes.
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
from torchcell.data.graph_processor import NeighborSubgraphRepresentation, LazySubgraphRepresentation
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.graph import build_gene_multigraph
from torchcell.transforms.coo_regression_to_classification import COOLabelNormalizationTransform


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Profile NeighborSubgraphRepresentation with different hop counts"
    )
    parser.add_argument(
        "--num-hops",
        type=int,
        default=2,
        help="Number of hops for neighborhood sampling (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=28,
        help="Batch size for profiling (default: 28)",
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

    print("="*80)
    print("EXPERIMENT 087a: NEIGHBOR SUBGRAPH DATALOADER PROFILING")
    print("="*80)
    print(f"Comparing with Experiment 086a (LazySubgraphRepresentation)")
    print(f"  086a Lazy Hetero: 1.31 it/s")
    print(f"  086a DANGO:      73.82 it/s")
    print("="*80)
    print()

    # Configuration from args
    NUM_HOPS = args.num_hops
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    SUBSET_SIZE = args.subset_size
    MAX_STEPS = args.max_steps

    print(f"Configuration:")
    print(f"  Graph Processor: NeighborSubgraphRepresentation")
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
    if NUM_HOPS == 0:
        print("Creating dataset with LazySubgraphRepresentation (full graph + masks)...")
        graph_processor = LazySubgraphRepresentation()
        processor_name = "LazySubgraph"
    else:
        print(f"Creating dataset with {NUM_HOPS}-hop NeighborSubgraphRepresentation...")
        graph_processor = NeighborSubgraphRepresentation(num_hops=NUM_HOPS)
        processor_name = f"NeighborSubgraph ({NUM_HOPS}-hop)"

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

    # Normalization transform (same as 086a)
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

    # Subset Module (same as 086a)
    print(f"Creating {SUBSET_SIZE}-sample subset...")
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=SUBSET_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch=False,
        seed=42,
        follow_batch=["perturbation_indices"],
        gene_subsets={"metabolism": yeast_gem.gene_set},
    )
    perturbation_subset_data_module.setup()

    # Get dataloader
    train_loader = perturbation_subset_data_module.train_dataloader()

    print()
    print("="*80)
    print("STARTING PROFILING")
    print("="*80)
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

        # Move batch to GPU (same as 086a dataloader profiling)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            batch = batch.to(device)
            gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            gpu_mem_usage.append(gpu_mem_mb)

        cpu_mem_usage.append(cpu_mem_mb)

        # Track subgraph size (for gene nodes)
        if hasattr(batch["gene"], "num_nodes"):
            # Convert to int to avoid tensor formatting issues
            num_nodes = batch["gene"].num_nodes
            if isinstance(num_nodes, torch.Tensor):
                num_nodes = num_nodes.item()
            num_nodes_list.append(num_nodes)

        # Track total edges across all edge types
        total_edges = 0
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "num_edges"):
                num_edges = batch[edge_type].num_edges
                if isinstance(num_edges, torch.Tensor):
                    # For batched data, num_edges might be a tensor - sum it
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
    avg_cpu_mem = sum(cpu_mem_usage) / len(cpu_mem_usage) if cpu_mem_usage else 0
    avg_gpu_mem = sum(gpu_mem_usage) / len(gpu_mem_usage) if gpu_mem_usage else 0
    avg_num_nodes = sum(num_nodes_list) / len(num_nodes_list) if num_nodes_list else 0
    avg_num_edges = sum(num_edges_list) / len(num_edges_list) if num_edges_list else 0

    print()
    print("="*80)
    print("PROFILING RESULTS")
    print("="*80)
    print()
    print(f"Steps profiled: {len(times)}")
    print(f"Average time per batch: {avg_time:.4f} sec")
    print(f"Average iterations/sec: {avg_its:.3f} it/s")
    print()

    # Memory metrics
    print("="*80)
    print("MEMORY USAGE")
    print("="*80)
    print()
    print(f"Average CPU memory: {avg_cpu_mem:.1f} MB")
    if gpu_mem_usage:
        print(f"Average GPU memory per batch: {avg_gpu_mem:.1f} MB")
        # Estimate max batch size (assuming 80GB GPU, reserve 20GB for model/overhead)
        available_gpu_mem = 60 * 1024  # 60 GB in MB
        if avg_gpu_mem > 0:
            estimated_max_batch = int((available_gpu_mem / avg_gpu_mem) * BATCH_SIZE)
            print(f"Estimated max batch size (60GB available): {estimated_max_batch}")
        print()

    # Subgraph size metrics
    print("="*80)
    print("SUBGRAPH SIZE")
    print("="*80)
    print()
    print(f"Average nodes per batch: {avg_num_nodes:.1f}")
    print(f"Average edges per batch: {avg_num_edges:.1f}")
    if BATCH_SIZE > 0:
        print(f"Average nodes per sample: {avg_num_nodes/BATCH_SIZE:.1f}")
        print(f"Average edges per sample: {avg_num_edges/BATCH_SIZE:.1f}")
    print()

    # Comparison with reference (DANGO from 086a)
    dango_its = 73.82

    print("="*80)
    print("COMPARISON WITH BASELINES")
    print("="*80)
    print()
    print(f"  Current ({processor_name}): {avg_its:.3f} it/s")
    print(f"  DANGO (086a):              {dango_its:.3f} it/s")
    print()

    slowdown_vs_dango = dango_its / avg_its if avg_its > 0 else 0
    print(f"Slowdown vs DANGO: {slowdown_vs_dango:.2f}×")

    # Only show speedup vs lazy if this isn't the lazy run
    if NUM_HOPS > 0:
        lazy_its = 1.31  # Reference from 086a
        speedup_vs_lazy = avg_its / lazy_its if lazy_its > 0 else 0
        print(f"Speedup vs LazySubgraph (086a): {speedup_vs_lazy:.2f}×")
    print()

    # Interpretation
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    if speedup_vs_lazy > 5.0:
        print("✅ SIGNIFICANT IMPROVEMENT!")
        print(f"  NeighborSubgraph is {speedup_vs_lazy:.1f}× faster than LazySubgraph")
        print("  This graph processor drastically reduces data pipeline overhead.")
    elif speedup_vs_lazy > 2.0:
        print("✅ GOOD IMPROVEMENT")
        print(f"  NeighborSubgraph is {speedup_vs_lazy:.1f}× faster than LazySubgraph")
        print("  Significant reduction in data processing time.")
    elif speedup_vs_lazy > 1.2:
        print("⚠ MODERATE IMPROVEMENT")
        print(f"  NeighborSubgraph is {speedup_vs_lazy:.1f}× faster than LazySubgraph")
        print("  Some improvement, but not dramatic.")
    else:
        print("❌ NO SIGNIFICANT IMPROVEMENT")
        print(f"  NeighborSubgraph is only {speedup_vs_lazy:.2f}× faster")
        print("  Neighbor sampling doesn't help much for this workload.")

    print()

    if slowdown_vs_dango < 10:
        print("✅ Close to DANGO performance!")
        print(f"  Only {slowdown_vs_dango:.1f}× slower than DANGO")
    else:
        print(f"⚠ Still {slowdown_vs_dango:.1f}× slower than DANGO")
        print("  Further optimization needed to match DANGO speed.")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
