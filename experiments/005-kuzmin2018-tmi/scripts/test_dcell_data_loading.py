# experiments/005-kuzmin2018-tmi/scripts/test_dcell_data_loading
# [[experiments.005-kuzmin2018-tmi.scripts.test_dcell_data_loading]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/test_dcell_data_loading
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_test_dcell_data_loading.py


import os
import os.path as osp
import hydra
import torch
import time
import logging
import psutil
import numpy as np
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

import hypernetx

# Torchcell imports
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data.graph_processor import Perturbation, DCellGraphProcessor
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.datamodules import CellDataModule
from torchcell.data import Neo4jCellDataset
from torchcell.graph import (
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
    filter_by_contained_genes,
)
from torch_geometric.loader import DataLoader


log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    ram_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
    
    # Add GPU memory tracking if GPU is available
    gpu_usage = None
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        
    return ram_usage, gpu_usage


def print_timing(
    step_name, start_time, end_time=None, memory_before=None, memory_after=None
):
    """Print timing and memory usage for a step"""
    if end_time is None:
        end_time = time.time()

    duration = end_time - start_time

    memory_str = ""
    if memory_before is not None and memory_after is not None:
        # Unpack RAM and GPU memory
        ram_before, gpu_before = memory_before if isinstance(memory_before, tuple) else (memory_before, None)
        ram_after, gpu_after = memory_after if isinstance(memory_after, tuple) else (memory_after, None)
        
        # Calculate RAM difference
        ram_diff = ram_after - ram_before
        memory_str = f", RAM: {ram_before:.2f}MB → {ram_after:.2f}MB (Δ: {ram_diff:+.2f}MB)"
        
        # Add GPU memory info if available
        if gpu_before is not None and gpu_after is not None:
            gpu_diff = gpu_after - gpu_before
            memory_str += f", GPU: {gpu_before:.2f}MB → {gpu_after:.2f}MB (Δ: {gpu_diff:+.2f}MB)"

    print(f"⏱️  {step_name}: {duration:.2f}s{memory_str}")
    return end_time


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig) -> None:
    # Check if all imports were successful
    """
    Test DCell data loading performance.
    This script loads the DCell datasets and measures loading times,
    without performing any model training.
    """
    print("\n" + "=" * 80)
    print("📊 DCELL DATA LOADING PERFORMANCE TEST")
    print("=" * 80 + "\n")

    # Convert config to container for easier access
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Determine device based on accelerator config
    if config["trainer"]["accelerator"] == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device")

    # Start timing the overall process
    overall_start = time.time()
    memory_start = get_memory_usage()

    print("Configuration:")
    print(
        f"- Perturbation subset size: {config['data_module']['perturbation_subset_size']}"
    )
    print(f"- Batch size: {config['data_module']['batch_size']}")
    print(f"- Num workers: {config['data_module']['num_workers']}")
    print(f"- Pin memory: {config['data_module']['pin_memory']}")
    print(f"- Prefetch: {config['data_module']['prefetch']}")
    print("")

    # Step 1: Load genome
    print("Step 1: Loading genome...")
    step_start = time.time()
    mem_before = get_memory_usage()

    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()

    mem_after = get_memory_usage()
    step_end = print_timing(
        "Genome loading", step_start, memory_before=mem_before, memory_after=mem_after
    )

    # Step 2: Create graph
    print("\nStep 2: Creating SCerevisiaeGraph...")
    step_start = time.time()
    mem_before = get_memory_usage()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    mem_after = get_memory_usage()
    step_end = print_timing(
        "Graph creation", step_start, memory_before=mem_before, memory_after=mem_after
    )

    # Step 3: Apply GO filters
    print("\nStep 3: Processing and filtering GO graph...")
    step_start = time.time()
    mem_before = get_memory_usage()

    G_go = graph.G_go.copy()
    print(f"Original GO graph: {G_go.number_of_nodes()} nodes")

    # Apply DCell-specific GO graph filters
    date_filter = config["model"].get("go_date_filter", None)
    if date_filter:
        G_go = filter_by_date(G_go, date_filter)
        print(f"After date filter ({date_filter}): {G_go.number_of_nodes()} nodes")

    G_go = filter_go_IGI(G_go)
    print(f"After IGI filter: {G_go.number_of_nodes()} nodes")

    # Apply redundant terms filter for more efficient network
    if config["model"].get("filter_redundant_terms", True):
        g_before = G_go.number_of_nodes()
        G_go = filter_redundant_terms(G_go)
        print(
            f"After redundant terms filter: {G_go.number_of_nodes()} nodes (removed {g_before - G_go.number_of_nodes()})"
        )

    # Build gene to GO term mapping
    gene_to_terms = {}
    for term in G_go.nodes():
        for gene in G_go.nodes[term].get("genes", []):
            if gene not in gene_to_terms:
                gene_to_terms[gene] = set()
            gene_to_terms[gene].add(term)

    print(f"Number of genes in GO mapping: {len(gene_to_terms)}")

    mem_after = get_memory_usage()
    step_end = print_timing(
        "GO graph processing",
        step_start,
        memory_before=mem_before,
        memory_after=mem_after,
    )

    # Step 4: Create Neo4jCellDataset
    print("\nStep 4: Creating Neo4jCellDataset...")
    step_start = time.time()
    mem_before = get_memory_usage()

    try:
        # Create dataset
        # Read the query file
        with open(
            osp.join(EXPERIMENT_ROOT, "005-kuzmin2018-tmi/queries/001_small_build.cql"),
            "r",
        ) as f:
            query = f.read()

        dataset_root = osp.join(
            DATA_ROOT, "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build"
        )

        # Initialize the graph processor once and reuse it
        graph_processor = DCellGraphProcessor()
        
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=set(genome.gene_set),
            graphs=None,  # For DCell, we don't use gene multigraph
            incidence_graphs={"gene_ontology": G_go},  # Pass the GO graph we processed
            node_embeddings=config["cell_dataset"].get("node_embeddings"),
            graph_processor=graph_processor,
        )
        
        # Move the cell graph to the correct device
        if hasattr(dataset, "cell_graph"):
            dataset.cell_graph = dataset.cell_graph.to(device)

        print(f"Dataset size: {len(dataset)} samples")

        mem_after = get_memory_usage()
        step_end = print_timing(
            "Neo4jCellDataset creation",
            step_start,
            memory_before=mem_before,
            memory_after=mem_after,
        )

        # Step 5: Create CellDataModule
        print("\nStep 5: Creating CellDataModule...")
        step_start = time.time()
        mem_before = get_memory_usage()

        # Determine cache dir
        if config.get("cell_dataset", {}).get("incidence_graphs") == "go":
            cache_dir_suffix = "go"
        else:
            cache_dir_suffix = "graph"

        cache_dir = osp.join(
            DATA_ROOT, "data/torchcell/cache", f"kuzmin2018_tmi_{cache_dir_suffix}"
        )

        # Create dummy index files to avoid using the dataset indices, which might have overlaps
        os.makedirs(cache_dir, exist_ok=True)

        # Use random_seed to force random sampling without using dataset indices
        random_seed = 12345

        # Define the follow_batch parameter
        follow_batch = ["perturbation_indices"]
        if hasattr(graph_processor, "process_batch"):
            follow_batch.append("mutant_state")

        dm = CellDataModule(
            dataset=dataset,
            cache_dir=cache_dir,
            split_indices=[
                "phenotype_label_index",
                "perturbation_count_index",
            ],  # Same as in dcell.py
            random_seed=random_seed,
            batch_size=config["data_module"]["batch_size"],
            num_workers=config["data_module"]["num_workers"],
            pin_memory=config["data_module"]["pin_memory"],
            prefetch=config["data_module"]["prefetch"],
            follow_batch=follow_batch,
        )

        mem_after = get_memory_usage()
        step_end = print_timing(
            "CellDataModule creation",
            step_start,
            memory_before=mem_before,
            memory_after=mem_after,
        )

        # Step 6: Create PerturbationSubsetDataModule
        print("\nStep 6: Creating PerturbationSubsetDataModule...")
        step_start = time.time()
        mem_before = get_memory_usage()

        subset_size = config["data_module"].get("perturbation_subset_size", 100)
        follow_batch = config["data_module"].get("follow_batch", ["x", "x_pert"])

        subset_dm = PerturbationSubsetDataModule(
            cell_data_module=dm,
            size=subset_size,
            batch_size=config["data_module"]["batch_size"],
            num_workers=config["data_module"]["num_workers"],
            pin_memory=config["data_module"]["pin_memory"],
            prefetch=config["data_module"]["prefetch"],
            follow_batch=follow_batch,
        )

        mem_after = get_memory_usage()
        step_end = print_timing(
            "PerturbationSubsetDataModule creation",
            step_start,
            memory_before=mem_before,
            memory_after=mem_after,
        )

        # Step 7: Skip creating Graph Processor since it's already created above
        print("\nStep 7: DCellGraphProcessor already created in Step 5")

        # Step 8: Process a batch
        print("\nStep 8: Setting up and processing first batch...")
        step_start = time.time()
        mem_before = get_memory_usage()

        # Setup data module
        subset_dm.setup()

        # Get train loader
        train_loader = subset_dm.train_dataloader()
        print(f"Train loader length: {len(train_loader)} batches")

        # Process first batch
        print("\nProcessing first batch...")
        first_batch_start = time.time()
        first_batch = next(iter(train_loader))
        first_batch_end = time.time()
        print(f"First batch retrieval time: {first_batch_end - first_batch_start:.2f}s")

        # DCellGraphProcessor doesn't have build_cell_graph and process_batch methods
        # The processing is already done by the dataset during loading
        # Just examine the batch structure
        print("\nExamining batch structure...")
        process_start = time.time()

        # Display batch structure
        if hasattr(first_batch, "node_types"):
            print(f"Batch node types: {first_batch.node_types}")

            if "gene_ontology" in first_batch.node_types:
                go_nodes = first_batch["gene_ontology"]
                print(f"GO nodes in batch: {go_nodes.num_nodes}")

                # Check if mutant_state is present
                if hasattr(go_nodes, "mutant_state"):
                    print(f"Mutant state shape: {go_nodes.mutant_state.shape}")

                # Check if strata information is available
                if hasattr(go_nodes, "strata"):
                    strata = go_nodes.strata
                    print(f"Strata tensor shape: {strata.shape}")

        process_end = time.time()
        print(f"Batch examination time: {process_end - process_start:.2f}s")

        # Print batch details
        print(f"\nBatch details:")
        print(f"- Batch size: {first_batch.num_graphs}")
        if hasattr(first_batch, "node_types"):
            for node_type in first_batch.node_types:
                print(f"- {node_type} nodes: {first_batch[node_type].num_nodes}")

                # Check attributes of each node type
                for attr_name in vars(first_batch[node_type]):
                    if not attr_name.startswith("_") and hasattr(
                        first_batch[node_type], attr_name
                    ):
                        attr = getattr(first_batch[node_type], attr_name)
                        if hasattr(attr, "shape"):
                            print(f"  - {attr_name} shape: {attr.shape}")

                # If strata information is available in gene_ontology
                if node_type == "gene_ontology" and hasattr(
                    first_batch["gene_ontology"], "strata"
                ):
                    strata = first_batch["gene_ontology"].strata
                    num_strata = len(torch.unique(strata))
                    print(f"- Number of strata: {num_strata}")

                    # Count terms per stratum
                    strata_counts = {}
                    for s in strata:
                        stratum = s.item()
                        if stratum not in strata_counts:
                            strata_counts[stratum] = 0
                        strata_counts[stratum] += 1

                    print("- Strata distribution:")
                    for stratum in sorted(strata_counts.keys())[:5]:
                        print(f"  Stratum {stratum}: {strata_counts[stratum]} terms")
                    if len(strata_counts) > 5:
                        print(f"  ... and {len(strata_counts)-5} more strata")

        mem_after = get_memory_usage()
        step_end = print_timing(
            "Batch processing",
            step_start,
            memory_before=mem_before,
            memory_after=mem_after,
        )

        # Step 9: Iterate through all batches to test full pipeline
        print("\nStep 9: Testing full data pipeline...")
        step_start = time.time()
        mem_before = get_memory_usage()

        batch_times = []

        # Start timer before loading first batch
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Record the time it took to load this batch
            end_time = time.time()
            batch_time = end_time - start_time
            batch_times.append(batch_time)

            print(
                f"Batch {batch_idx+1}/{len(train_loader)}, Batch loading time: {batch_time:.4f}s"
            )
        
            # Reset timer for next batch
            start_time = time.time()

        # Calculate statistics
        avg_batch_time = np.mean(batch_times)
        total_dataloader_time = sum(batch_times)

        print(f"\nDataloader statistics:")
        print(f"- Average batch loading time: {avg_batch_time:.4f}s")
        print(f"- Total dataloader time: {total_dataloader_time:.2f}s")
        print(f"- Estimated epoch time: {total_dataloader_time:.2f}s")

        mem_after = get_memory_usage()
        step_end = print_timing(
            "Full data pipeline test",
            step_start,
            memory_before=mem_before,
            memory_after=mem_after,
        )

    except Exception as e:
        print(f"Error during dataset creation: {e}")
        raise

    # Print overall timing
    overall_time = time.time() - overall_start
    memory_end, gpu_memory_end = get_memory_usage()
    memory_start_ram, memory_start_gpu = memory_start if isinstance(memory_start, tuple) else (memory_start, None)
    
    memory_used = memory_end - memory_start_ram
    
    print("\n" + "=" * 80)
    print(f"Total execution time: {overall_time:.2f}s")
    print(f"RAM usage: {memory_used:.2f}MB")
    
    if memory_start_gpu is not None and gpu_memory_end is not None:
        gpu_used = gpu_memory_end - memory_start_gpu
        print(f"GPU memory usage: {gpu_used:.2f}MB")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
