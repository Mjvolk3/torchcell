# experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_3.py
# [[experiments.010-kuzmin-tmi.scripts.equivariant_cell_graph_transformer_inference_3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/equivariant_cell_graph_transformer_inference_3.py
"""
Inference script for CellGraphTransformer model using InferenceDataset.
Loads a trained checkpoint and generates predictions for triple gene combinations.
Results are streamed to a Parquet file with batched writes for efficient storage.

INFERENCE_3 VERSION:
- Uses triples from inference_3 directory (relaxed thresholds)
- Thresholds: max(SMF) > 1.04, max(DMF) > 1.08
- Baselines: all(SMF) > 0.90, all(DMF) > 0.90
- Designed for JT test validation (0.04 gap for ~96% power at n=8)

Multi-GPU support:
- Launch with torchrun --nproc_per_node=4 for 4-GPU inference
- Each rank processes a contiguous 1/N block of the dataset
- Each rank writes its own parquet shard
- Rank 0 merges all shards into final output file
- Also works with single GPU (no torchrun needed)

Key features:
- Streaming Parquet output with 100K record batches
- Dictionary encoding for gene columns (~50x compression vs CSV)
- Handles large prediction sets efficiently
"""

import json
import logging
import os
import os.path as osp
import sys

import gc
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm

# Add the current script's directory to Python path for local imports
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

# Key difference: Use Perturbation processor (not SubgraphRepresentation)
from inference_dataset_2 import InferenceDataset
from torchcell.data import (
    GenotypeAggregator,
    MeanExperimentDeduplicator,
    Neo4jCellDataset,
)
from torchcell.data.graph_processor import Perturbation
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.models.equivariant_cell_graph_transformer import CellGraphTransformer
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.trainers.int_transformer_cell import RegressionTask
from torchcell.transforms.coo_regression_to_classification import (
    COOInverseCompose,
    COOLabelNormalizationTransform,
)

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")

# Parquet batch size for efficient streaming writes
PARQUET_BATCH_SIZE = 100_000  # Write batches of 100K predictions


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed():
    """Detect and initialize distributed environment from torchrun.

    Returns (rank, world_size, local_rank, is_distributed).
    When not launched via torchrun, returns single-process defaults.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0, False

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def log_print(msg, rank=0, is_main=True):
    """Print only on rank 0."""
    if is_main:
        print(msg)


# ---------------------------------------------------------------------------
# Parquet writer (unchanged)
# ---------------------------------------------------------------------------


class StreamingPredictionParquetWriter:
    """
    Helper class to write predictions to Parquet in batches.

    Uses dictionary encoding for gene columns (huge compression with ~2K unique genes)
    and snappy compression for speed.
    """

    def __init__(self, filename: str, batch_size: int = PARQUET_BATCH_SIZE):
        self.filename = filename
        self.batch_size = batch_size
        self.batch = []
        self.total_written = 0
        self.schema = pa.schema(
            [
                ("index", pa.int64()),
                ("gene1", pa.string()),
                ("gene2", pa.string()),
                ("gene3", pa.string()),
                ("prediction", pa.float32()),
            ]
        )
        self.writer = pq.ParquetWriter(
            filename,
            self.schema,
            compression="snappy",
            use_dictionary=["gene1", "gene2", "gene3"],
        )

    def write(self, index: int, genes: list, prediction: float):
        """Add a prediction to the batch."""
        self.batch.append(
            (
                index,
                genes[0] if len(genes) > 0 else "",
                genes[1] if len(genes) > 1 else "",
                genes[2] if len(genes) > 2 else "",
                float(prediction),
            )
        )
        if len(self.batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        if not self.batch:
            return
        table = pa.table(
            {
                "index": [t[0] for t in self.batch],
                "gene1": [t[1] for t in self.batch],
                "gene2": [t[2] for t in self.batch],
                "gene3": [t[3] for t in self.batch],
                "prediction": [t[4] for t in self.batch],
            },
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.total_written += len(self.batch)
        self.batch = []

    def close(self):
        self._flush_batch()
        self.writer.close()

    def get_total_written(self) -> int:
        return self.total_written + len(self.batch)


# ---------------------------------------------------------------------------
# Shard merge
# ---------------------------------------------------------------------------


def merge_parquet_shards(inferred_dir: str, base_filename: str, world_size: int):
    """Merge per-rank parquet shards into a single sorted file.

    Reads all rank shards, concatenates, sorts by index, and writes the final
    merged file. Removes shard files after successful merge.
    """
    shard_files = []
    for r in range(world_size):
        shard = osp.join(inferred_dir, f"{base_filename}.rank{r}")
        shard_files.append(shard)

    print(f"Merging {world_size} shards...")
    tables = [pq.read_table(f) for f in shard_files]
    merged = pa.concat_tables(tables)

    # Cast string columns to large_string (int64 offsets) to avoid the 2GB
    # offset overflow that occurs when take() consolidates chunked arrays
    # with ~465M rows of gene names.
    for i, field in enumerate(merged.schema):
        if field.type == pa.string():
            merged = merged.set_column(
                i, field.name, merged.column(field.name).cast(pa.large_string())
            )

    sorted_indices = pa.compute.sort_indices(
        merged, sort_keys=[("index", "ascending")]
    )
    merged = merged.take(sorted_indices)

    final_path = osp.join(inferred_dir, base_filename)
    pq.write_table(
        merged,
        final_path,
        compression="snappy",
        use_dictionary=["gene1", "gene2", "gene3"],
    )
    print(f"Merged {len(merged):,} predictions -> {final_path}")

    # Clean up shards
    for f in shard_files:
        os.remove(f)
        print(f"  Removed shard: {f}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved
    return 0, 0


def aggressive_cleanup():
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
    """Convert checkpoint path to output filename."""
    filename = checkpoint_path.replace("/", "-").replace("\\", "-")
    filename = filename.replace(".ckpt", ".parquet")
    return filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="equivariant_cell_graph_transformer_cabbi_002_inference_m02",
)
def main(cfg: DictConfig) -> None:
    # --- Distributed setup ---
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = rank == 0

    log_print("Starting CellGraphTransformer Inference 3", is_main=is_main)
    log_print("=" * 80, is_main=is_main)
    log_print(
        "INFERENCE_3: Relaxed thresholds for JT test validation", is_main=is_main
    )
    log_print("Thresholds: max(SMF) > 1.04, max(DMF) > 1.08", is_main=is_main)
    log_print("Baselines: all(SMF) > 0.90, all(DMF) > 0.90", is_main=is_main)
    if is_distributed:
        log_print(
            f"Distributed: {world_size} GPUs (rank {rank})", is_main=is_main
        )
    log_print("=" * 80, is_main=is_main)

    # Only rank 0 initializes wandb
    if is_main:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True))
        config = wandb.config
    else:
        # Other ranks read config directly from Hydra
        config = OmegaConf.to_container(cfg, resolve=True)

    # --- Helper to access config consistently ---
    def cfg_get(keys, default=None):
        """Access nested config keys, works with both wandb.config and dict."""
        obj = config
        for k in keys.split("."):
            if isinstance(obj, dict):
                obj = obj.get(k, default)
            else:
                obj = getattr(obj, k, default)
            if obj is default:
                return default
        return obj

    # Memory optimization config
    mem_config = {
        "stream_gene_names": True,
        "monitor_memory": True,
        "cache_clear_frequency": 10,
        "adaptive_batch_size": True,
        "min_batch_size": 1,
    }

    # --- Device ---
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        log_print(f"GPU {rank}: {gpu_name} ({total_memory:.2f} GB)", is_main=is_main)

    # --- Setup genome ---
    genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
    go_root = osp.join(DATA_ROOT, "data/go")
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()

    # --- Setup graph ---
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    graph_names = cfg_get("cell_dataset.graphs")
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    graphs_dict = dict(gene_multigraph.graphs) if gene_multigraph else None

    # --- Node embeddings ---
    node_embedding_names = cfg_get("cell_dataset.node_embeddings", [])
    if node_embedding_names:
        node_embeddings = NodeEmbeddingBuilder.build(
            embedding_names=node_embedding_names,
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        )
    else:
        node_embeddings = None
    log_print(
        f"Node embeddings: {list(node_embeddings.keys()) if node_embeddings else 'None (using learnable)'}",
        is_main=is_main,
    )

    graph_processor = Perturbation()

    # INFERENCE_3: Dataset path
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_3"
    )
    output_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_3"
    )

    # --- Transforms ---
    forward_transform = None
    inverse_transform = None

    if cfg_get("transforms.use_transforms", False):
        log_print(
            "\nInitializing transforms from original dataset...", is_main=is_main
        )
        transform_config = cfg_get("transforms.forward_transform", {})

        original_dataset_root = osp.join(
            DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
        )
        with open(
            osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi/queries/001_small_build.cql"), "r"
        ) as f:
            query = f.read()

        log_print(
            "Loading original dataset to extract transform statistics...",
            is_main=is_main,
        )
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
        if "normalization" in transform_config:
            norm_config = transform_config["normalization"]
            norm_transform = COOLabelNormalizationTransform(
                original_dataset, norm_config
            )
            transforms_list.append(norm_transform)
            log_print(
                f"Added normalization transform for: {list(norm_config.keys())}",
                is_main=is_main,
            )
            for label, stats in norm_transform.stats.items():
                log_print(f"Normalization parameters for {label}:", is_main=is_main)
                for key, value in stats.items():
                    if isinstance(value, (int, float)) and key != "strategy":
                        log_print(f"  {key}: {value:.6f}", is_main=is_main)
                    else:
                        log_print(f"  {key}: {value}", is_main=is_main)

        if transforms_list:
            forward_transform = Compose(transforms_list)
            inverse_transform = COOInverseCompose(transforms_list)
            log_print("Transforms initialized successfully", is_main=is_main)

        original_dataset.close_lmdb()
        log_print("Closed original dataset", is_main=is_main)

    # --- Create InferenceDataset ---
    log_print("\nCreating InferenceDataset...", is_main=is_main)
    dataset = InferenceDataset(
        root=dataset_root,
        gene_set=genome.gene_set,
        graphs=graphs_dict,
        node_embeddings=node_embeddings,
        graph_processor=graph_processor,
        transform=forward_transform,
    )
    total_size = len(dataset)
    log_print(f"Dataset size: {total_size:,}", is_main=is_main)

    # --- Partition dataset across ranks ---
    per_rank = (total_size + world_size - 1) // world_size
    start_idx = rank * per_rank
    end_idx = min(start_idx + per_rank, total_size)
    rank_indices = list(range(start_idx, end_idx))
    rank_dataset = Subset(dataset, rank_indices)

    log_print(
        f"Rank {rank}: processing indices [{start_idx:,}, {end_idx:,}) "
        f"({len(rank_indices):,} samples)",
        is_main=True,  # All ranks print their range
    )

    # --- DataLoader ---
    current_batch_size = cfg_get("data_module.batch_size", 4096)
    log_print(f"   Using batch size: {current_batch_size}", is_main=is_main)

    def create_inference_dataloader(ds, batch_size):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            follow_batch=["perturbation_indices"],
        )

    # --- Graph regularization lambda ---
    loss_config_raw = cfg_get("regression_task.loss", {})
    graph_reg_lambda = 0.0
    if isinstance(loss_config_raw, dict):
        graph_reg_config = loss_config_raw.get("graph_regularization", {})
        if isinstance(graph_reg_config, dict):
            lambda_val = graph_reg_config.get("lambda", 0.0)
            graph_reg_lambda = 0.0 if lambda_val is None else float(lambda_val)

    # --- Create model ---
    log_print(
        f"\nInstantiating CellGraphTransformer ({timestamp()})", is_main=is_main
    )
    model_cfg = cfg_get("model")
    model = CellGraphTransformer(
        gene_num=model_cfg["gene_num"],
        hidden_channels=model_cfg["hidden_channels"],
        num_transformer_layers=model_cfg["num_transformer_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        cell_graph=dataset.cell_graph,
        graph_regularization_config=model_cfg["graph_regularization"],
        perturbation_head_config=model_cfg["perturbation_head"],
        dropout=model_cfg["dropout"],
        graph_reg_lambda=graph_reg_lambda,
        node_embeddings=node_embeddings,
        learnable_embedding_config=model_cfg.get("learnable_embedding"),
    ).to(device)

    # --- Loss function ---
    loss_config = cfg_get("regression_task.loss", "logcosh")
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

    # --- Load checkpoint ---
    checkpoint_path = model_cfg["checkpoint_path"]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = osp.join(DATA_ROOT, checkpoint_path)
    log_print(f"\nLoading checkpoint from: {checkpoint_path}", is_main=is_main)

    reg_cfg = cfg_get("regression_task")
    task = RegressionTask.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=model,
        cell_graph=dataset.cell_graph,
        loss_func=loss_func,
        device=device,
        optimizer_config=reg_cfg["optimizer"],
        lr_scheduler_config=reg_cfg["lr_scheduler"],
        batch_size=cfg_get("data_module.batch_size"),
        clip_grad_norm=reg_cfg["clip_grad_norm"],
        clip_grad_norm_max_norm=reg_cfg["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=reg_cfg["plot_every_n_epochs"],
        plot_sample_ceiling=reg_cfg["plot_sample_ceiling"],
        plot_edge_recovery_every_n_epochs=reg_cfg.get(
            "plot_edge_recovery_every_n_epochs", 10
        ),
        plot_transformer_diagnostics_every_n_epochs=reg_cfg.get(
            "plot_transformer_diagnostics_every_n_epochs", 10
        ),
        grad_accumulation_schedule=reg_cfg.get("grad_accumulation_schedule"),
        execution_mode="inference",
        strict=False,
    )
    log_print("Successfully loaded model checkpoint", is_main=is_main)

    task.eval()
    model.eval()

    # --- Output file ---
    inferred_dir = osp.join(output_root, "inferred")
    os.makedirs(inferred_dir, exist_ok=True)

    output_filename = get_output_filename_from_checkpoint(model_cfg["checkpoint_path"])

    # Each rank writes to its own shard (or single file if not distributed)
    if is_distributed:
        shard_file = osp.join(inferred_dir, f"{output_filename}.rank{rank}")
    else:
        shard_file = osp.join(inferred_dir, output_filename)

    log_print(f"\nWriting predictions to: {shard_file}", is_main=True)

    parquet_writer = StreamingPredictionParquetWriter(
        shard_file, batch_size=PARQUET_BATCH_SIZE
    )

    # --- Inference loop ---
    use_amp = device.type == "cuda"
    local_idx = 0  # counts samples processed by this rank
    oom_retries = 0
    max_oom_retries = 3

    try:
        while local_idx < len(rank_dataset):
            try:
                dataloader = create_inference_dataloader(
                    rank_dataset, current_batch_size
                )
                skip_batches = local_idx // current_batch_size

                with torch.no_grad():
                    pbar = tqdm(
                        dataloader,
                        desc=f"[rank {rank}] inference (bs={current_batch_size})",
                        disable=(not is_main and not is_distributed),
                    )
                    for batch_idx, batch in enumerate(pbar):
                        if batch_idx < skip_batches:
                            continue

                        # Memory monitoring
                        if mem_config["monitor_memory"] and batch_idx % 50 == 0:
                            alloc, reserved = get_gpu_memory_usage()
                            if is_main:
                                print(
                                    f"\nGPU Memory: {alloc:.2f}GB allocated, "
                                    f"{reserved:.2f}GB reserved"
                                )

                        batch = batch.to(device)

                        if (
                            device.type == "cuda"
                            and batch_idx % mem_config["cache_clear_frequency"] == 0
                        ):
                            aggressive_cleanup()

                        # Forward pass
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                predictions, representations = task(batch)
                        else:
                            predictions, representations = task(batch)

                        del representations

                        if predictions.dim() == 0:
                            predictions = predictions.unsqueeze(0).unsqueeze(0)
                        elif predictions.dim() == 1:
                            predictions = predictions.unsqueeze(1)

                        # Inverse transform
                        if inverse_transform is not None:
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

                        for i in range(batch_size):
                            # Map local position back to global dataset index
                            global_idx = start_idx + local_idx + i
                            if global_idx >= total_size:
                                break

                            if mem_config["stream_gene_names"]:
                                genes = get_gene_names_from_lmdb(dataset, global_idx)
                            else:
                                genes = []

                            parquet_writer.write(
                                index=global_idx,
                                genes=genes,
                                prediction=float(predictions_cpu[i].numpy()),
                            )

                        del predictions
                        if "predictions_cpu" in locals():
                            del predictions_cpu
                        del batch

                        local_idx += batch_size

                        if batch_idx % 100 == 0 and batch_idx > 0 and is_main:
                            global_processed = local_idx  # rank 0 progress
                            print(
                                f"\n[rank {rank}] Processed {local_idx:,}/{len(rank_dataset):,} "
                                f"({local_idx/len(rank_dataset)*100:.1f}%) "
                                f"[{parquet_writer.total_written:,} written to parquet]"
                            )

                # Successful completion of this rank's work
                break

            except torch.cuda.OutOfMemoryError as e:
                print(
                    f"\n[rank {rank}] OOM with batch_size={current_batch_size}: {e}"
                )
                aggressive_cleanup()
                if mem_config["adaptive_batch_size"] and oom_retries < max_oom_retries:
                    current_batch_size = max(
                        current_batch_size // 2, mem_config["min_batch_size"]
                    )
                    oom_retries += 1
                    print(
                        f"[rank {rank}] Reducing batch size to {current_batch_size} "
                        f"(retry {oom_retries}/{max_oom_retries})"
                    )
                else:
                    raise e

    finally:
        parquet_writer.close()

    print(
        f"[rank {rank}] Inference complete! "
        f"Processed {local_idx:,} experiments, "
        f"written {parquet_writer.total_written:,} predictions"
    )

    # --- Merge shards on rank 0 ---
    if is_distributed:
        dist.barrier()  # Wait for all ranks to finish writing
        if is_main:
            merge_parquet_shards(inferred_dir, output_filename, world_size)

    log_print(
        f"\nFinal output: {osp.join(inferred_dir, output_filename)}", is_main=is_main
    )

    aggressive_cleanup()
    if is_main:
        wandb.finish()
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    import sys

    if "--merge-only" in sys.argv:
        # Re-run just the shard merge (no GPUs needed).
        # Usage: python script.py --merge-only
        import glob as globmod

        from dotenv import load_dotenv

        load_dotenv()
        data_root = os.environ["DATA_ROOT"]
        inferred_dir = osp.join(
            data_root,
            "data/torchcell/experiments/010-kuzmin-tmi/inference_3/inferred",
        )
        shard_files = sorted(globmod.glob(osp.join(inferred_dir, "*.rank0")))
        for shard0 in shard_files:
            base = shard0.removesuffix(".rank0")
            base_filename = osp.basename(base)
            world_size = len(globmod.glob(f"{base}.rank*"))
            print(f"Merging {base_filename} ({world_size} shards)")
            merge_parquet_shards(inferred_dir, base_filename, world_size)
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        main()
