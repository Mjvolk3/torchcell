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
from torchcell.data import Neo4jCellDataset, MeanExperimentDeduplicator, GenotypeAggregator

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="hetero_cell_bipartite_dango_gi_test",
)
def main(cfg: DictConfig) -> None:
    print("Starting GeneInteractionDango Inference 🔬")
    
    # Setup genome
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")
    
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=True
    )
    genome.drop_empty_go()
    
    # Setup graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    
    # Build gene multigraph using graph names from config
    graph_names = cfg.cell_dataset["graphs"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    
    # Extract the graphs as a dict if gene_multigraph exists
    # This allows InferenceDataset to properly create a GeneMultiGraph with base graph
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None
    
    # Build node embeddings
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=cfg.cell_dataset["node_embeddings"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )
    
    # Setup incidence graphs
    incidence_graphs = {}
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    if "metabolism_bipartite" in cfg.cell_dataset["incidence_graphs"]:
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
    
    if cfg.transforms.get("use_transforms", False):
        print("\nInitializing transforms from original dataset...")
        transform_config = cfg.transforms.get("forward_transform", {})
        
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
            norm_transform = COOLabelNormalizationTransform(original_dataset, norm_config)
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
            bin_transform = COOLabelBinningTransform(original_dataset, bin_config, norm_transform)
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
    def create_inference_dataloader(dataset, cfg):
        """Create a DataLoader for inference that preserves important settings."""
        batch_size = cfg.data_module["batch_size"]
        num_workers = cfg.data_module["num_workers"]
        pin_memory = cfg.data_module["pin_memory"]
        prefetch = cfg.data_module["prefetch"]
        prefetch_factor = cfg.data_module["prefetch_factor"]
        
        # Create base dataloader with settings matching CellDataModule
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffling for inference
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin_memory,
            follow_batch=["x", "x_pert"],  # Important for node features
            timeout=10800,
            multiprocessing_context=("spawn" if num_workers > 0 else None),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        
        # Optionally wrap with PrefetchLoader for GPU optimization
        if prefetch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return PrefetchLoader(loader, device=device)
        
        return loader
    
    # Setup device - Force GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: Running on CPU - inference will be slow!")
        print("   Consider running on a machine with GPU support.")
    
    # Override batch size for inference if needed
    inference_batch_size = cfg.data_module.get("inference_batch_size", cfg.data_module["batch_size"])
    if inference_batch_size != cfg.data_module["batch_size"]:
        print(f"   Using inference batch size: {inference_batch_size}")
        cfg.data_module["batch_size"] = inference_batch_size
    
    # Create dataloader for inference
    dataloader = create_inference_dataloader(dataset, cfg)
    
    # Initialize model architecture
    gene_encoder_config = dict(cfg.model["gene_encoder_config"])
    if any("learnable" in emb for emb in cfg.cell_dataset["node_embeddings"]):
        gene_encoder_config.update(
            {
                "embedding_type": "learnable",
                "max_num_nodes": dataset.cell_graph["gene"].num_nodes,
                "learnable_embedding_input_channels": cfg.cell_dataset["learnable_embedding_input_channels"],
            }
        )
    
    local_predictor_config = dict(cfg.model.get("local_predictor_config", {}))
    
    # Create model
    model = GeneInteractionDango(
        gene_num=cfg.model["gene_num"],
        hidden_channels=cfg.model["hidden_channels"],
        num_layers=cfg.model["num_layers"],
        gene_multigraph=gene_multigraph,
        dropout=cfg.model["dropout"],
        norm=cfg.model["norm"],
        activation=cfg.model["activation"],
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
    ).to(device)
    
    # Setup loss function
    weights = torch.ones(1).to(device)
    if cfg.regression_task["loss"] == "icloss":
        loss_func = ICLoss(
            lambda_dist=cfg.regression_task["lambda_dist"],
            lambda_supcr=cfg.regression_task["lambda_supcr"],
            weights=weights,
        )
    elif cfg.regression_task["loss"] == "logcosh":
        loss_func = LogCoshLoss(reduction="mean")
    
    # Load checkpoint
    checkpoint_path = cfg.model["checkpoint_path"]
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
        optimizer_config=OmegaConf.to_container(cfg.regression_task["optimizer"]),
        lr_scheduler_config=OmegaConf.to_container(cfg.regression_task["lr_scheduler"]),
        batch_size=cfg.data_module["batch_size"],
        clip_grad_norm=cfg.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=cfg.regression_task["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=cfg.regression_task["plot_every_n_epochs"],
        plot_sample_ceiling=cfg.regression_task["plot_sample_ceiling"],
        grad_accumulation_schedule=cfg.regression_task["grad_accumulation_schedule"],
    )
    
    print("Successfully loaded model checkpoint")
    
    # Set model to evaluation mode
    task.eval()
    model.eval()
    
    # Create output file
    output_file = osp.join(
        dataset_root, f"inference_predictions_{timestamp()}.csv"
    )
    print(f"\nWriting predictions to: {output_file}")
    
    # Get total batches for progress tracking
    total_batches = len(dataloader)
    
    # Option to skip gene name caching for faster inference
    cache_gene_names = cfg.get("cache_gene_names", True)
    gene_names_cache = {}
    
    if cache_gene_names:
        # Pre-cache all gene names for faster lookup
        print("\nCaching gene names from LMDB...")
        print("Note: This may take a few minutes for large datasets. Set cache_gene_names=False in config to skip.")
        
        dataset._init_lmdb_read()
        
        try:
            with dataset.env.begin() as txn:
                # Get total count first
                stat = txn.stat()
                total_entries = stat['entries']
                
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
                                        pert["systematic_gene_name"] for pert in perturbations
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
                                    pert["systematic_gene_name"] for pert in perturbations
                                ]
                            except Exception:
                                continue
                        pbar.update(len(chunk_data))
                        
        finally:
            dataset.close_lmdb()
        
        print(f"Cached {len(gene_names_cache)} gene name entries")
    else:
        print("\nSkipping gene name caching. Output will use indices instead of gene names.")
    
    # Open CSV file and start inference
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['index', 'gene1', 'gene2', 'gene3', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        current_idx = 0
        
        # Enable mixed precision for GPU inference
        use_amp = device.type == "cuda"
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
                # Move batch to device
                batch = batch.to(device)
                
                # Clear GPU cache periodically to prevent memory buildup
                if device.type == "cuda" and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                
                # Forward pass through task with optional mixed precision
                if use_amp:
                    with torch.cuda.amp.autocast():
                        predictions, representations = task(batch)
                else:
                    predictions, representations = task(batch)
                
                # Ensure predictions has correct shape
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0).unsqueeze(0)
                elif predictions.dim() == 1:
                    predictions = predictions.unsqueeze(1)
                
                # Apply inverse transform if available (following RegressionTask logic)
                if inverse_transform is not None:
                    # Create a temp HeteroData object with predictions in COO format
                    temp_data = HeteroData()
                    
                    # Create COO format data for predictions
                    batch_size_actual = predictions.size(0)
                    device = predictions.device
                    temp_data["gene"].phenotype_values = predictions.squeeze()
                    temp_data["gene"].phenotype_type_indices = torch.zeros(batch_size_actual, dtype=torch.long, device=device)
                    temp_data["gene"].phenotype_sample_indices = torch.arange(batch_size_actual, device=device)
                    temp_data["gene"].phenotype_types = ["gene_interaction"]
                    
                    # Apply the inverse transform
                    inv_data = inverse_transform(temp_data)
                    
                    # Extract the inversed predictions
                    inv_gene_int = inv_data["gene"]["phenotype_values"]
                    
                    # Handle tensor shape
                    if isinstance(inv_gene_int, torch.Tensor):
                        if inv_gene_int.dim() == 0:
                            predictions = inv_gene_int.unsqueeze(0).unsqueeze(0)
                        elif inv_gene_int.dim() == 1:
                            predictions = inv_gene_int.unsqueeze(1)
                        else:
                            predictions = inv_gene_int
                
                # Get batch size
                batch_size = predictions.size(0)
                
                # Process each prediction in the batch
                for i in range(batch_size):
                    # Get experiment index
                    exp_idx = current_idx + i
                    
                    # Skip if index is out of bounds
                    if exp_idx >= len(dataset):
                        continue
                    
                    # Get gene names or use indices
                    try:
                        if cache_gene_names:
                            genes = gene_names_cache.get(exp_idx, [])
                            # Extract the gene names (handle variable number of genes)
                            row = {
                                'index': exp_idx,
                                'gene1': genes[0] if len(genes) > 0 else '',
                                'gene2': genes[1] if len(genes) > 1 else '',
                                'gene3': genes[2] if len(genes) > 2 else '',
                                'prediction': float(predictions[i].cpu().numpy())
                            }
                        else:
                            # Use indices when gene names are not cached
                            row = {
                                'index': exp_idx,
                                'gene1': f'idx_{exp_idx}_gene1',
                                'gene2': f'idx_{exp_idx}_gene2', 
                                'gene3': f'idx_{exp_idx}_gene3',
                                'prediction': float(predictions[i].cpu().numpy())
                            }
                        writer.writerow(row)
                    except Exception as e:
                        print(f"\nError processing experiment {exp_idx}: {e}")
                        continue
                
                # Flush after each batch to ensure data is written
                csvfile.flush()
                current_idx += batch_size
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"\nProcessed {current_idx}/{len(dataset)} experiments ({current_idx/len(dataset)*100:.1f}%)")
    
    print(f"\n✅ Inference complete! Processed {current_idx} experiments")
    print(f"Results saved to: {output_file}")
    
    # Dataset LMDB already closed after caching


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()