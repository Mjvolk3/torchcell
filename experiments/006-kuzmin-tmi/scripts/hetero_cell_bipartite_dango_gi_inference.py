#!/usr/bin/env python
"""
Inference script for GeneInteractionDango model using InferenceDataset.
Loads a trained checkpoint and generates predictions for triple gene combinations.
Results are written incrementally to a CSV file for real-time monitoring.
"""

import csv
import json
import logging
import os
import os.path as osp
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader, PrefetchLoader
from dotenv import load_dotenv
import gc
import wandb

# Add the current script's directory to Python path for local imports
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from inference_dataset import InferenceDataset
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.isomorphic_cell_loss import ICLoss
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.models.hetero_cell_bipartite_dango_gi import GeneInteractionDango
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.trainers.int_hetero_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
    COOLabelBinningTransform,
    COOInverseCompose,
)
from torch_geometric.transforms import Compose
from torch_geometric.data import HeteroData
from torchcell.data import (
    Neo4jCellDataset,
    MeanExperimentDeduplicator,
    GenotypeAggregator,
)

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


def aggressive_cleanup():
    """Perform aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gene_names_from_lmdb(dataset, idx):
    """Stream gene names from LMDB for a single index."""
    dataset._init_lmdb_read()
    try:
        with dataset.env.begin() as txn:
            value = txn.get(str(idx).encode())
            if value:
                data_list = json.loads(value.decode())
                exp = data_list[0]["experiment"]
                perturbations = exp["genotype"]["perturbations"]
                return [pert["systematic_gene_name"] for pert in perturbations]
    except Exception:
        pass
    finally:
        dataset.close_lmdb()
    return []


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="hetero_cell_bipartite_dango_gi_test",
)
def main(cfg: DictConfig) -> None:
    print("Starting GeneInteractionDango Inference 🔬")
    
    # Initialize wandb with the Hydra config
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True))
    
    # Hardcoded memory optimization config
    mem_config = {
        "stream_gene_names": True,
        "monitor_memory": True,
        "cache_clear_frequency": 10,
        "adaptive_batch_size": True,
        "min_batch_size": 1
    }

    # Setup genome
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(genome_root=genome_root, go_root=go_root, overwrite=True)
    genome.drop_empty_go()

    # Setup graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph using graph names from config
    graph_names = wandb.config.cell_dataset["graphs"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    # Extract the graphs as a dict if gene_multigraph exists
    # This allows InferenceDataset to properly create a GeneMultiGraph with base graph
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None

    # Build node embeddings
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=wandb.config.cell_dataset["node_embeddings"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )

    # Setup incidence graphs
    incidence_graphs = {}
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    if "metabolism_bipartite" in wandb.config.cell_dataset["incidence_graphs"]:
        incidence_graphs["metabolism_bipartite"] = yeast_gem.bipartite_graph

    # Setup graph processor
    graph_processor = SubgraphRepresentation()

    # Create InferenceDataset
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_0"
    )

    # Initialize transforms if configured
    forward_transform = None
    inverse_transform = None

    if wandb.config.transforms.get("use_transforms", False):
        print("\nInitializing transforms from original dataset...")
        transform_config = wandb.config.transforms.get("forward_transform", {})

        # Load the original dataset to get the normalization statistics
        original_dataset_root = osp.join(
            DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
        )

        # Read the query for the original dataset
        with open(
            osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
        ) as f:
            query = f.read()

        print("Loading original dataset to extract transform statistics...")
        original_dataset = Neo4jCellDataset(
            root=original_dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            incidence_graphs=incidence_graphs,
            node_embeddings=node_embeddings,
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=graph_processor,
            transform=None,  # No transform initially
        )

        transforms_list = []
        norm_transform = None

        # Normalization transform
        if "normalization" in transform_config:
            norm_config = transform_config["normalization"]
            norm_transform = COOLabelNormalizationTransform(
                original_dataset, norm_config
            )
            transforms_list.append(norm_transform)
            print(f"Added normalization transform for: {list(norm_config.keys())}")

            # Print normalization parameters
            for label, stats in norm_transform.stats.items():
                print(f"Normalization parameters for {label}:")
                for key, value in stats.items():
                    if isinstance(value, (int, float)) and key != "strategy":
                        print(f"  {key}: {value:.6f}")
                    else:
                        print(f"  {key}: {value}")

        # Binning transform
        if "binning" in transform_config:
            bin_config = transform_config["binning"]
            bin_transform = COOLabelBinningTransform(
                original_dataset, bin_config, norm_transform
            )
            transforms_list.append(bin_transform)
            print(f"Added binning transform for: {list(bin_config.keys())}")

        if transforms_list:
            forward_transform = Compose(transforms_list)
            inverse_transform = COOInverseCompose(transforms_list)
            print("Transforms initialized successfully")

        # Close the original dataset's LMDB
        original_dataset.close_lmdb()
        print("Closed original dataset")

    # Create InferenceDataset for predictions
    print("\nCreating InferenceDataset...")
    dataset = InferenceDataset(
        root=dataset_root,
        gene_set=genome.gene_set,
        graphs=graphs_dict,  # Pass dict instead of GeneMultiGraph
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        graph_processor=graph_processor,
        transform=forward_transform,  # Apply the same transforms
    )

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader helper function
    def create_inference_dataloader(dataset, batch_size=None):
        """Create a DataLoader for inference that preserves important settings."""
        if batch_size is None:
            batch_size = wandb.config.data_module["batch_size"]
        num_workers = wandb.config.data_module.get("num_workers", 0)
        pin_memory = wandb.config.data_module.get("pin_memory", False)
        prefetch = wandb.config.data_module.get("prefetch", False)
        prefetch_factor = wandb.config.data_module.get("prefetch_factor", None)

        # Use reduced settings for memory optimization
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for inference
            num_workers=0,  # Use 0 workers for better memory control
            persistent_workers=False,
            pin_memory=False,  # Disable pin memory to save RAM
            follow_batch=["x", "x_pert"],  # Important for node features
        )

        return loader

    # Setup device - Force GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {total_memory:.2f} GB")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: Running on CPU - inference will be slow!")
        print("   Consider running on a machine with GPU support.")

    # Initial batch size with adaptive sizing support
    current_batch_size = wandb.config.data_module.get(
        "inference_batch_size", wandb.config.data_module["batch_size"]
    )
    print(f"   Using initial batch size: {current_batch_size}")

    # Initialize model architecture
    gene_encoder_config = dict(wandb.config.model["gene_encoder_config"])
    if any("learnable" in emb for emb in wandb.config.cell_dataset["node_embeddings"]):
        gene_encoder_config.update(
            {
                "embedding_type": "learnable",
                "max_num_nodes": dataset.cell_graph["gene"].num_nodes,
                "learnable_embedding_input_channels": wandb.config.cell_dataset[
                    "learnable_embedding_input_channels"
                ],
            }
        )

    local_predictor_config = dict(wandb.config.model.get("local_predictor_config", {}))

    # Create model
    model = GeneInteractionDango(
        gene_num=wandb.config.model["gene_num"],
        hidden_channels=wandb.config.model["hidden_channels"],
        num_layers=wandb.config.model["num_layers"],
        gene_multigraph=gene_multigraph,
        dropout=wandb.config.model["dropout"],
        norm=wandb.config.model["norm"],
        activation=wandb.config.model["activation"],
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
    ).to(device)

    # Setup loss function
    weights = torch.ones(1).to(device)
    if wandb.config.regression_task["loss"] == "icloss":
        loss_func = ICLoss(
            lambda_dist=wandb.config.regression_task["lambda_dist"],
            lambda_supcr=wandb.config.regression_task["lambda_supcr"],
            weights=weights,
        )
    elif wandb.config.regression_task["loss"] == "logcosh":
        loss_func = LogCoshLoss(reduction="mean")

    # Load checkpoint
    checkpoint_path = wandb.config.model["checkpoint_path"]
    print(f"\nLoading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load the task from checkpoint
    task = RegressionTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=model,
        cell_graph=dataset.cell_graph,
        loss_func=loss_func,
        device=device,
        optimizer_config=wandb.config.regression_task["optimizer"],
        lr_scheduler_config=wandb.config.regression_task["lr_scheduler"],
        batch_size=wandb.config.data_module["batch_size"],
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=wandb.config.regression_task["plot_every_n_epochs"],
        plot_sample_ceiling=wandb.config.regression_task["plot_sample_ceiling"],
        grad_accumulation_schedule=wandb.config.regression_task["grad_accumulation_schedule"],
    )

    print("Successfully loaded model checkpoint")

    # Set model to evaluation mode
    task.eval()
    model.eval()

    # Create output file
    output_file = osp.join(dataset_root, f"inference_predictions_{timestamp()}.csv")
    print(f"\nWriting predictions to: {output_file}")

    # Pre-cache gene names if not streaming
    gene_names_cache = {}
    if not mem_config.get("stream_gene_names", True):
        # Pre-cache all gene names for faster lookup
        print("\nCaching gene names from LMDB...")
        print(
            "Note: This may take a few minutes for large datasets. Set cache_gene_names=False in config to skip."
        )

        dataset._init_lmdb_read()

        try:
            with dataset.env.begin() as txn:
                # Get total count first
                stat = txn.stat()
                total_entries = stat["entries"]

                cursor = txn.cursor()

                # Process in chunks without loading all into memory
                chunk_size = 50000
                processed = 0

                with tqdm(total=total_entries, desc="Loading gene names") as pbar:
                    chunk_data = []

                    for key, value in cursor:
                        chunk_data.append((key, value))

                        if len(chunk_data) >= chunk_size:
                            # Process chunk
                            for k, v in chunk_data:
                                try:
                                    idx = int(k.decode())
                                    data_list = json.loads(v.decode())
                                    exp = data_list[0]["experiment"]
                                    perturbations = exp["genotype"]["perturbations"]
                                    gene_names_cache[idx] = [
                                        pert["systematic_gene_name"]
                                        for pert in perturbations
                                    ]
                                except Exception as e:
                                    continue

                            pbar.update(len(chunk_data))
                            processed += len(chunk_data)
                            chunk_data = []

                    # Process remaining data
                    if chunk_data:
                        for k, v in chunk_data:
                            try:
                                idx = int(k.decode())
                                data_list = json.loads(v.decode())
                                exp = data_list[0]["experiment"]
                                perturbations = exp["genotype"]["perturbations"]
                                gene_names_cache[idx] = [
                                    pert["systematic_gene_name"]
                                    for pert in perturbations
                                ]
                            except Exception:
                                continue
                        pbar.update(len(chunk_data))

        finally:
            dataset.close_lmdb()

        print(f"Cached {len(gene_names_cache)} gene name entries")
    else:
        print(
            "\nStreaming gene names on-demand to save memory."
        )

    # Open CSV file and start inference
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["index", "gene1", "gene2", "gene3", "prediction"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        current_idx = 0
        oom_retries = 0
        max_oom_retries = 3

        # Enable mixed precision for GPU inference
        use_amp = device.type == "cuda"

        while current_idx < len(dataset):
            try:
                # Create dataloader with current batch size
                dataloader = create_inference_dataloader(dataset, current_batch_size)
                
                # Skip to current position
                skip_batches = current_idx // current_batch_size
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(
                        tqdm(dataloader, desc=f"Running inference (batch_size={current_batch_size})")
                    ):
                        # Skip already processed batches
                        if batch_idx < skip_batches:
                            continue
                        # Memory monitoring
                        if mem_config.get("monitor_memory", True) and batch_idx % 50 == 0:
                            alloc, reserved = get_gpu_memory_usage()
                            print(f"\nGPU Memory: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved")
                        
                        # Move batch to device
                        batch = batch.to(device)
                        
                        # Clear GPU cache more frequently
                        if device.type == "cuda" and batch_idx % mem_config.get("cache_clear_frequency", 10) == 0:
                            aggressive_cleanup()

                        # Forward pass through task with optional mixed precision
                        try:
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    predictions, representations = task(batch)
                            else:
                                predictions, representations = task(batch)
                            
                            # Delete representations to save memory
                            del representations

                            # Ensure predictions has correct shape
                            if predictions.dim() == 0:
                                predictions = predictions.unsqueeze(0).unsqueeze(0)
                            elif predictions.dim() == 1:
                                predictions = predictions.unsqueeze(1)

                            # Apply inverse transform if available
                            if inverse_transform is not None:
                                # Create a temp HeteroData object with predictions in COO format
                                temp_data = HeteroData()

                                # Create COO format data for predictions
                                batch_size_actual = predictions.size(0)
                                temp_data["gene"].phenotype_values = predictions.squeeze()
                                temp_data["gene"].phenotype_type_indices = torch.zeros(
                                    batch_size_actual, dtype=torch.long, device=device
                                )
                                temp_data["gene"].phenotype_sample_indices = torch.arange(
                                    batch_size_actual, device=device
                                )
                                temp_data["gene"].phenotype_types = ["gene_interaction"]

                                # Apply the inverse transform
                                inv_data = inverse_transform(temp_data)

                                # Extract the inversed predictions and move to CPU immediately
                                predictions_cpu = inv_data["gene"]["phenotype_values"].cpu()
                                
                                # Clean up temp objects
                                del temp_data, inv_data
                                
                                # Handle tensor shape
                                if predictions_cpu.dim() == 0:
                                    predictions_cpu = predictions_cpu.unsqueeze(0).unsqueeze(0)
                                elif predictions_cpu.dim() == 1:
                                    predictions_cpu = predictions_cpu.unsqueeze(1)
                            else:
                                predictions_cpu = predictions.cpu()

                            # Get batch size
                            batch_size = predictions_cpu.size(0)

                            # Process each prediction in the batch
                            for i in range(batch_size):
                                # Get experiment index
                                exp_idx = current_idx + i

                                # Skip if index is out of bounds
                                if exp_idx >= len(dataset):
                                    break

                                # Get gene names
                                try:
                                    if mem_config.get("stream_gene_names", True):
                                        genes = get_gene_names_from_lmdb(dataset, exp_idx)
                                    else:
                                        genes = gene_names_cache.get(exp_idx, [])
                                    
                                    # Extract the gene names (handle variable number of genes)
                                    row = {
                                        "index": exp_idx,
                                        "gene1": genes[0] if len(genes) > 0 else "",
                                        "gene2": genes[1] if len(genes) > 1 else "",
                                        "gene3": genes[2] if len(genes) > 2 else "",
                                        "prediction": float(predictions_cpu[i].numpy()),
                                    }
                                    writer.writerow(row)
                                except Exception as e:
                                    print(f"\nError processing experiment {exp_idx}: {e}")
                                    continue
                            
                            # Cleanup tensors
                            del predictions
                            if 'predictions_cpu' in locals():
                                del predictions_cpu
                            del batch
                            
                            # Update index
                            current_idx += batch_size
                            
                            # Flush after each batch to ensure data is written
                            csvfile.flush()
                            
                            # Print progress every 100 batches
                            if batch_idx % 100 == 0 and batch_idx > 0:
                                print(
                                    f"\nProcessed {current_idx}/{len(dataset)} experiments ({current_idx/len(dataset)*100:.1f}%)"
                                )
                        
                        except torch.cuda.OutOfMemoryError as e:
                            print(f"\nOOM during forward pass with batch_size={current_batch_size}")
                            aggressive_cleanup()
                            raise e
                
                # Successful completion
                break
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n⚠️  OOM Error with batch_size={current_batch_size}: {e}")
                aggressive_cleanup()
                
                # Adaptive batch size reduction
                if mem_config.get("adaptive_batch_size", True) and oom_retries < max_oom_retries:
                    current_batch_size = max(
                        current_batch_size // 2, 
                        mem_config.get("min_batch_size", 1)
                    )
                    oom_retries += 1
                    print(f"Reducing batch size to {current_batch_size} (retry {oom_retries}/{max_oom_retries})")
                else:
                    print("Max OOM retries reached or adaptive batch sizing disabled")
                    raise e

    print(f"\n✅ Inference complete! Processed {current_idx} experiments")
    print(f"Results saved to: {output_file}")
    
    # Final cleanup
    aggressive_cleanup()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
