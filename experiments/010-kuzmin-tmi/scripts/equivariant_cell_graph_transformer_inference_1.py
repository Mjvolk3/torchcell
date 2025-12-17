#!/usr/bin/env python
"""
Inference script for CellGraphTransformer model using InferenceDataset.
Loads a trained checkpoint and generates predictions for triple gene combinations.
Results are streamed to a Parquet file with batched writes for efficient storage.

Key features:
- Streaming Parquet output with 100K record batches
- Dictionary encoding for gene columns (~50x compression vs CSV)
- Handles 275M+ predictions efficiently (~2-4GB vs ~50GB CSV)

Adapted from hetero_cell_bipartite_dango_gi_inference.py for the transformer architecture.
"""

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
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv
import gc
import wandb
import pyarrow as pa
import pyarrow.parquet as pq

# Add the current script's directory to Python path for local imports
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

# Key difference: Use Perturbation processor (not SubgraphRepresentation)
from inference_dataset_1 import InferenceDataset
from torchcell.data.graph_processor import Perturbation
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.trainers.int_transformer_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
    COOInverseCompose,
)
from torch_geometric.transforms import Compose
from torch_geometric.data import Data, HeteroData
from torchcell.data import (
    Neo4jCellDataset,
    MeanExperimentDeduplicator,
    GenotypeAggregator,
)

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Parquet batch size for efficient streaming writes
PARQUET_BATCH_SIZE = 100_000  # Write batches of 100K predictions


class StreamingPredictionParquetWriter:
    """
    Helper class to write predictions to Parquet in batches.

    Adapted from StreamingParquetWriter in generate_triple_combinations_inference_1.py.
    Uses dictionary encoding for gene columns (huge compression with ~2K unique genes)
    and snappy compression for speed.

    For 275M predictions:
    - CSV would be ~50GB
    - Parquet with dict encoding: ~2-4GB (10-25x compression)
    """

    def __init__(self, filename: str, batch_size: int = PARQUET_BATCH_SIZE):
        self.filename = filename
        self.batch_size = batch_size
        self.batch = []
        self.total_written = 0
        self.schema = pa.schema([
            ('index', pa.int64()),
            ('gene1', pa.string()),
            ('gene2', pa.string()),
            ('gene3', pa.string()),
            ('prediction', pa.float32()),  # float32 saves space, sufficient precision
        ])
        self.writer = pq.ParquetWriter(
            filename,
            self.schema,
            compression='snappy',  # Fast compression
            use_dictionary=['gene1', 'gene2', 'gene3'],  # Dict-encode gene columns only
        )

    def write(self, index: int, genes: list, prediction: float):
        """Add a prediction to the batch."""
        self.batch.append((
            index,
            genes[0] if len(genes) > 0 else "",
            genes[1] if len(genes) > 1 else "",
            genes[2] if len(genes) > 2 else "",
            float(prediction)
        ))

        if len(self.batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """Write accumulated batch to Parquet file."""
        if not self.batch:
            return

        table = pa.table({
            'index': [t[0] for t in self.batch],
            'gene1': [t[1] for t in self.batch],
            'gene2': [t[2] for t in self.batch],
            'gene3': [t[3] for t in self.batch],
            'prediction': [t[4] for t in self.batch],
        }, schema=self.schema)

        self.writer.write_table(table)
        self.total_written += len(self.batch)
        self.batch = []

    def close(self):
        """Flush remaining batch and close writer."""
        self._flush_batch()
        self.writer.close()

    def get_total_written(self) -> int:
        """Get total rows written (including current batch)."""
        return self.total_written + len(self.batch)


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


def get_output_filename_from_checkpoint(checkpoint_path: str) -> str:
    """Convert checkpoint path to output filename.

    Replaces path separators with dashes and changes .ckpt to .parquet.

    Example:
        'models/checkpoints/gilahyper-647_.../Pearson=0.4149.ckpt'
        -> 'models-checkpoints-gilahyper-647_...-Pearson=0.4149.parquet'
    """
    filename = checkpoint_path.replace("/", "-").replace("\\", "-")
    filename = filename.replace(".ckpt", ".parquet")
    return filename


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="equivariant_cell_graph_transformer_gh_014_inference_test",
)
def main(cfg: DictConfig) -> None:
    print("Starting CellGraphTransformer Inference 🔬")

    # Initialize wandb with the Hydra config
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True))

    # Memory optimization config
    mem_config = {
        "stream_gene_names": True,
        "monitor_memory": True,
        "cache_clear_frequency": 10,
        "adaptive_batch_size": True,
        "min_batch_size": 1,
    }

    # Setup genome
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")

    genome = SCerevisiaeGenome(genome_root=genome_root, go_root=go_root, overwrite=False)
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

    # Extract the graphs as a dict for InferenceDataset
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None

    # Build node embeddings (may be empty for transformer with learnable embeddings)
    node_embedding_names = wandb.config.cell_dataset.get("node_embeddings", [])
    if node_embedding_names:
        node_embeddings = NodeEmbeddingBuilder.build(
            embedding_names=node_embedding_names,
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        )
    else:
        node_embeddings = None
    print(f"Node embeddings: {list(node_embeddings.keys()) if node_embeddings else 'None (using learnable)'}")

    # Key difference: Use Perturbation processor (not SubgraphRepresentation)
    graph_processor = Perturbation()

    # Dataset path - using inference_1 from 006 (shared processed LMDB)
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/inference_1"
    )

    # Output path - experiment-specific directory for inference results
    output_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_1"
    )

    # Initialize transforms if configured
    forward_transform = None
    inverse_transform = None

    if wandb.config.transforms.get("use_transforms", False):
        print("\nInitializing transforms from original dataset...")
        transform_config = wandb.config.transforms.get("forward_transform", {})

        # Load the original training dataset to get normalization statistics
        original_dataset_root = osp.join(
            DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
        )

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
            node_embeddings=node_embeddings,
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=graph_processor,
            transform=None,
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
        graphs=graphs_dict,
        node_embeddings=node_embeddings,
        graph_processor=graph_processor,
        transform=forward_transform,
    )

    print(f"Dataset size: {len(dataset):,}")

    # Create dataloader helper function
    def create_inference_dataloader(dataset, batch_size=None):
        """Create a DataLoader for inference."""
        if batch_size is None:
            batch_size = wandb.config.data_module["batch_size"]

        # Key difference: follow_batch for Perturbation processor
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 workers for better memory control
            persistent_workers=False,
            pin_memory=False,
            follow_batch=["perturbation_indices"],  # Key for transformer
        )

        return loader

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU Memory: {total_memory:.2f} GB")
        torch.cuda.set_per_process_memory_fraction(0.9)
    else:
        device = torch.device("cpu")
        print("⚠️  WARNING: Running on CPU - inference will be slow!")

    # Initial batch size
    current_batch_size = wandb.config.data_module["batch_size"]
    print(f"   Using initial batch size: {current_batch_size}")

    # Get graph regularization lambda from loss config
    loss_config = wandb.config.regression_task.get("loss", {})
    graph_reg_lambda = 0.0
    if isinstance(loss_config, dict):
        graph_reg_config = loss_config.get("graph_regularization", {})
        if isinstance(graph_reg_config, dict):
            lambda_val = graph_reg_config.get("lambda", 0.0)
            graph_reg_lambda = 0.0 if lambda_val is None else float(lambda_val)

    # Create CellGraphTransformer model
    print(f"\nInstantiating CellGraphTransformer ({timestamp()})")
    model = CellGraphTransformer(
        gene_num=wandb.config["model"]["gene_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        num_transformer_layers=wandb.config["model"]["num_transformer_layers"],
        num_attention_heads=wandb.config["model"]["num_attention_heads"],
        cell_graph=dataset.cell_graph,
        graph_regularization_config=wandb.config["model"]["graph_regularization"],
        perturbation_head_config=wandb.config["model"]["perturbation_head"],
        dropout=wandb.config["model"]["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=wandb.config["model"].get("learnable_embedding"),
    ).to(device)

    # Setup loss function
    loss_config = wandb.config.regression_task.get("loss", "logcosh")

    if isinstance(loss_config, dict):
        loss_type = loss_config.get("type", "logcosh")

        if loss_type == "point_dist_graph_reg":
            loss_func = PointDistGraphReg(
                point_estimator=loss_config.get("point_estimator"),
                distribution_loss=loss_config.get("distribution_loss"),
                graph_regularization=loss_config.get("graph_regularization"),
                buffer=loss_config.get("buffer"),
                ddp=loss_config.get("ddp"),
            )
        elif loss_type == "logcosh":
            loss_func = LogCoshLoss(reduction="mean")
        elif loss_type == "mse":
            loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        if loss_config == "logcosh":
            loss_func = LogCoshLoss(reduction="mean")
        elif loss_config == "mse":
            loss_func = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_config}")

    # Load checkpoint
    checkpoint_path = wandb.config.model["checkpoint_path"]

    if not os.path.isabs(checkpoint_path):
        checkpoint_path = osp.join(DATA_ROOT, checkpoint_path)

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
        plot_edge_recovery_every_n_epochs=wandb.config.regression_task.get(
            "plot_edge_recovery_every_n_epochs", 10
        ),
        plot_transformer_diagnostics_every_n_epochs=wandb.config.regression_task.get(
            "plot_transformer_diagnostics_every_n_epochs", 10
        ),
        grad_accumulation_schedule=wandb.config.regression_task.get(
            "grad_accumulation_schedule"
        ),
        execution_mode="inference",
        strict=False,
    )

    print("Successfully loaded model checkpoint")

    # Set model to evaluation mode
    task.eval()
    model.eval()

    # Create output file in inferred/ directory, named by checkpoint path
    # Expected compression: ~50GB CSV → ~2-4GB Parquet with dictionary encoding
    # Use output_root (010-kuzmin-tmi) instead of dataset_root (006-kuzmin-tmi)
    inferred_dir = osp.join(output_root, "inferred")
    os.makedirs(inferred_dir, exist_ok=True)

    # Use checkpoint path as filename (unique per model)
    output_filename = get_output_filename_from_checkpoint(
        wandb.config.model["checkpoint_path"]
    )
    output_file = osp.join(inferred_dir, output_filename)
    print(f"\nWriting predictions to: {output_file}")
    print(f"Using streaming Parquet with batch_size={PARQUET_BATCH_SIZE:,}")

    # Initialize streaming parquet writer
    parquet_writer = StreamingPredictionParquetWriter(output_file, batch_size=PARQUET_BATCH_SIZE)

    try:
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
                        tqdm(
                            dataloader,
                            desc=f"Running inference (batch_size={current_batch_size})",
                        )
                    ):
                        # Skip already processed batches
                        if batch_idx < skip_batches:
                            continue

                        # Memory monitoring
                        if (
                            mem_config.get("monitor_memory", True)
                            and batch_idx % 50 == 0
                        ):
                            alloc, reserved = get_gpu_memory_usage()
                            print(
                                f"\nGPU Memory: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved"
                            )

                        # Move batch to device
                        batch = batch.to(device)

                        # Clear GPU cache periodically
                        if (
                            device.type == "cuda"
                            and batch_idx % mem_config.get("cache_clear_frequency", 10)
                            == 0
                        ):
                            aggressive_cleanup()

                        # Forward pass
                        try:
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    predictions, representations = task(batch)
                            else:
                                predictions, representations = task(batch)

                            del representations

                            # Ensure predictions has correct shape
                            if predictions.dim() == 0:
                                predictions = predictions.unsqueeze(0).unsqueeze(0)
                            elif predictions.dim() == 1:
                                predictions = predictions.unsqueeze(1)

                            # Apply inverse transform if available
                            if inverse_transform is not None:
                                # Create HeteroData with "gene" node store (matches training format)
                                temp_data = HeteroData()
                                batch_size_actual = predictions.size(0)
                                temp_data["gene"].phenotype_values = predictions.squeeze()
                                temp_data["gene"].phenotype_type_indices = torch.zeros(
                                    batch_size_actual, dtype=torch.long, device=device
                                )
                                temp_data["gene"].phenotype_sample_indices = torch.arange(
                                    batch_size_actual, device=device
                                )
                                temp_data["gene"].phenotype_types = ["gene_interaction"]

                                inv_data = inverse_transform(temp_data)
                                predictions_cpu = inv_data["gene"].phenotype_values.cpu()

                                del temp_data, inv_data

                                if predictions_cpu.dim() == 0:
                                    predictions_cpu = (
                                        predictions_cpu.unsqueeze(0).unsqueeze(0)
                                    )
                                elif predictions_cpu.dim() == 1:
                                    predictions_cpu = predictions_cpu.unsqueeze(1)
                            else:
                                predictions_cpu = predictions.cpu()

                            batch_size = predictions_cpu.size(0)

                            # Process each prediction
                            for i in range(batch_size):
                                exp_idx = current_idx + i

                                if exp_idx >= len(dataset):
                                    break

                                if mem_config.get("stream_gene_names", True):
                                    genes = get_gene_names_from_lmdb(dataset, exp_idx)
                                else:
                                    genes = []

                                # Write to streaming parquet (batched internally)
                                parquet_writer.write(
                                    index=exp_idx,
                                    genes=genes,
                                    prediction=float(predictions_cpu[i].numpy()),
                                )

                            del predictions
                            if "predictions_cpu" in locals():
                                del predictions_cpu
                            del batch

                            current_idx += batch_size

                            # Progress reporting (parquet flush is handled internally by batch size)
                            if batch_idx % 100 == 0 and batch_idx > 0:
                                print(
                                    f"\nProcessed {current_idx:,}/{len(dataset):,} experiments "
                                    f"({current_idx/len(dataset)*100:.1f}%) "
                                    f"[{parquet_writer.total_written:,} written to parquet]"
                                )

                        except torch.cuda.OutOfMemoryError as e:
                            print(
                                f"\nOOM during forward pass with batch_size={current_batch_size}"
                            )
                            aggressive_cleanup()
                            raise e

                # Successful completion
                break

            except torch.cuda.OutOfMemoryError as e:
                print(f"\n⚠️  OOM Error with batch_size={current_batch_size}: {e}")
                aggressive_cleanup()

                if (
                    mem_config.get("adaptive_batch_size", True)
                    and oom_retries < max_oom_retries
                ):
                    current_batch_size = max(
                        current_batch_size // 2, mem_config.get("min_batch_size", 1)
                    )
                    oom_retries += 1
                    print(
                        f"Reducing batch size to {current_batch_size} (retry {oom_retries}/{max_oom_retries})"
                    )
                else:
                    print("Max OOM retries reached or adaptive batch sizing disabled")
                    raise e

    finally:
        # Always close parquet writer to flush remaining batch and finalize file
        parquet_writer.close()

    print(f"\n✅ Inference complete! Processed {current_idx:,} experiments")
    print(f"Results saved to: {output_file}")
    print(f"Total predictions written: {parquet_writer.total_written:,}")

    aggressive_cleanup()
    wandb.finish()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
