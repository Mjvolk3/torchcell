# experiments/006-kuzmin-tmi/scripts/cell_graph_transformer
# [[experiments.006-kuzmin-tmi.scripts.cell_graph_transformer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/cell_graph_transformer
# Test file: experiments/006-kuzmin-tmi/scripts/test_cell_graph_transformer.py

import hashlib
import json
import logging
import os
import os.path as osp
import uuid
import warnings
import hydra

# Suppress SWIG-related deprecation warnings from frozen importlib
warnings.filterwarnings(
    "ignore", message="builtin type SwigPy", category=DeprecationWarning
)
import torch.nn as nn
import lightning as L
import torch
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.transforms.coo_regression_to_classification import (
    COOLabelNormalizationTransform,
    COOLabelBinningTransform,
    COOInverseCompose,
)
from torch_geometric.transforms import Compose
from torchcell.data.graph_processor import Perturbation
from torchcell.trainers.int_hetero_cell import RegressionTask
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb
import socket
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.cell_graph_transformer import CellGraphTransformer
from torchcell.graph.graph import build_gene_multigraph
from torchcell.datamodules import CellDataModule
from torchcell.data import Neo4jCellDataset
import torch.distributed as dist
from torchcell.timestamp import timestamp
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
WANDB_MODE = os.getenv("WANDB_MODE")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")


def get_slurm_nodes() -> int:
    try:
        print("SLURM_JOB_NUM_NODES:", os.environ.get("SLURM_JOB_NUM_NODES"))
        print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
        print("SLURM_NPROCS:", os.environ.get("SLURM_NPROCS"))
        if "SLURM_NNODES" in os.environ:
            return int(os.environ["SLURM_NNODES"])
        elif "SLURM_JOB_NUM_NODES" in os.environ:
            return int(os.environ["SLURM_JOB_NUM_NODES"])
        else:
            return 1
    except (TypeError, ValueError) as e:
        print(f"Error getting node count: {e}")
        return 1


def get_num_devices() -> int:
    if wandb.config.trainer["devices"] != "auto":
        return wandb.config.trainer["devices"]
    slurm_devices = os.environ.get("SLURM_GPUS_ON_NODE")
    if slurm_devices is not None:
        return int(slurm_devices)
    num_devices = torch.cuda.device_count()
    return num_devices if num_devices > 0 else 1


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="cell_graph_transformer",
)
def main(cfg: DictConfig) -> None:
    print("Starting Cell Graph Transformer Training 🚀")
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Set distributed timeout to 2 hours if using DDP
    if dist.is_available() and dist.is_initialized():
        pass
    else:
        os.environ["TORCH_DISTRIBUTED_DEFAULT_TIMEOUT"] = "7200"  # 2 hours in seconds

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print("wandb_cfg", wandb_cfg)

    # Check if using optuna sweeper
    is_optuna = (
        cfg.get("hydra", {})
        .get("sweeper", {})
        .get("_target_", "")
        .endswith("optuna.sweeper.OptunaSweeper")
    )

    # Get SLURM job IDs
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")

    # Determine job ID
    if slurm_array_job_id and slurm_array_task_id and is_optuna:
        job_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
    elif slurm_array_job_id:
        job_id = slurm_array_job_id
    elif slurm_job_id:
        job_id = slurm_job_id
    else:
        job_id = str(uuid.uuid4())

    hostname = socket.gethostname()
    hostname_job_id = f"{hostname}-{job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_job_id}_{hashed_cfg}"

    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    run = wandb.init(
        mode=WANDB_MODE,
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
        name=f"run_{group}",
    )

    wandb_logger = WandbLogger(
        project=wandb_cfg["wandb"]["project"],
        log_model=True,
        save_dir=experiment_dir,
        name=f"run_{group}",
    )

    # Log num_workers configuration for resource tracking
    num_workers = wandb.config.data_module["num_workers"]
    print(f"Using num_workers: {num_workers}")
    wandb.log({"config/num_workers": num_workers})

    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
        go_root = osp.join(DATA_ROOT, f"data/go/go_{rank}")
    else:
        genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
        go_root = osp.join(DATA_ROOT, "data/go")
        rank = 0

    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Build gene multigraph for dataset using graph names from config
    graph_names = wandb.config.cell_dataset["graphs"]
    gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)

    print(EXPERIMENT_ROOT)
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/006-kuzmin-tmi/001-small-build"
    )

    # Initialize transforms based on configuration
    forward_transform = None
    inverse_transform = None

    if wandb.config.transforms.get("use_transforms", False):
        print("\nInitializing transforms from configuration...")
        transform_config = wandb.config.transforms.get("forward_transform", {})

        transforms_list = []
        norm_transform = None

        # First create the dataset without transforms to get label statistics
        # Using Perturbation processor for transformer (only stores perturbation indices)
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            node_embeddings={},
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=Perturbation(),
            transform=None,
        )

        # Normalization transform
        if "normalization" in transform_config:
            norm_config = transform_config["normalization"]
            norm_transform = COOLabelNormalizationTransform(dataset, norm_config)
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
                dataset, bin_config, norm_transform
            )
            transforms_list.append(bin_transform)
            print(f"Added binning transform for: {list(bin_config.keys())}")

            # Print binning info
            for label in bin_config:
                bin_info = bin_transform.get_bin_info(label)
                print(f"Binning parameters for {label}:")
                print(f"  strategy: {bin_info['strategy']}")
                print(f"  num_bins: {len(bin_info['bin_edges']) - 1}")
                print(f"  bin_edges: {bin_info['bin_edges']}")

        if transforms_list:
            forward_transform = Compose(transforms_list)
            inverse_transform = COOInverseCompose(transforms_list)
            print("Transforms initialized successfully")

            # Log transform parameters to wandb
            if "normalization" in transform_config:
                for label, stats in norm_transform.stats.items():
                    for key, value in stats.items():
                        if isinstance(value, (int, float)) and key != "strategy":
                            wandb.log({f"normalization/{label}/{key}": value})

            # Set the forward transform on the dataset
            dataset.transform = forward_transform
    else:
        # Create dataset without transforms if not using transforms
        # Using Perturbation processor for transformer (only stores perturbation indices)
        dataset = Neo4jCellDataset(
            root=dataset_root,
            query=query,
            gene_set=genome.gene_set,
            graphs=gene_multigraph,
            node_embeddings={},
            converter=None,
            deduplicator=MeanExperimentDeduplicator,
            aggregator=GenotypeAggregator,
            graph_processor=Perturbation(),
            transform=None,
        )
    print(f"Dataset Length: {len(dataset)}")

    seed = 42

    # For Perturbation processor, need to track perturbation_indices for batch assignment
    follow_batch = ["perturbation_indices"]

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=seed,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
        prefetch=wandb.config.data_module["prefetch"],
        prefetch_factor=wandb.config.data_module["prefetch_factor"],
        persistent_workers=wandb.config.data_module["persistent_workers"],
        follow_batch=follow_batch,
        val_batch_size=wandb.config.data_module.get("val_batch_size"),
    )
    data_module.setup()

    if wandb.config.data_module["is_perturbation_subset"]:
        data_module = PerturbationSubsetDataModule(
            cell_data_module=data_module,
            size=int(wandb.config.data_module["perturbation_subset_size"]),
            batch_size=wandb.config.data_module["batch_size"],
            num_workers=wandb.config.data_module["num_workers"],
            pin_memory=wandb.config.data_module["pin_memory"],
            prefetch=wandb.config.data_module["prefetch"],
            seed=seed,
            prefetch_factor=wandb.config.data_module["prefetch_factor"],
            persistent_workers=wandb.config.data_module["persistent_workers"],
            follow_batch=follow_batch,
            val_batch_size=wandb.config.data_module.get("val_batch_size"),
        )
        data_module.setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)
    devices = get_num_devices()

    print(f"Instantiating CellGraphTransformer ({timestamp()})")

    # Instantiate CellGraphTransformer model
    model = CellGraphTransformer(
        gene_num=wandb.config["model"]["gene_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        num_transformer_layers=wandb.config["model"]["num_transformer_layers"],
        num_attention_heads=wandb.config["model"]["num_attention_heads"],
        cell_graph=dataset.cell_graph,
        graph_regularization_config=wandb.config["model"]["graph_regularization"],
        perturbation_head_config=wandb.config["model"]["perturbation_head"],
        dropout=wandb.config["model"]["dropout"],
        adaptive_loss_weighting=wandb.config["model"].get(
            "adaptive_loss_weighting", False
        ),
        graph_reg_scale=wandb.config["model"].get("graph_reg_scale", 0.001),
    ).to(device)

    # Log parameter counts
    param_counts = model.num_parameters
    print("Parameter counts:", param_counts)
    wandb.log(
        {
            "model/params_gene_embedding": param_counts.get("gene_embedding", 0),
            "model/params_cls_token": param_counts.get("cls_token", 0),
            "model/params_transformer_layers": param_counts.get(
                "transformer_layers", 0
            ),
            "model/params_perturbation_head": param_counts.get(
                "perturbation_head", 0
            ),
            "model/params_total": param_counts.get("total", 0),
        }
    )

    # Loss function - support both nested dict and simple string configs
    loss_config = wandb.config.regression_task.get("loss", "logcosh")

    # Check if loss is a dictionary (new structure) or string (legacy)
    if isinstance(loss_config, dict):
        loss_type = loss_config.get("type", "logcosh")

        if loss_type == "point_dist_graph_reg":
            # Extract nested configs using consistent access pattern
            point_estimator = loss_config.get("point_estimator")
            distribution_loss = loss_config.get("distribution_loss")
            graph_regularization = loss_config.get("graph_regularization")
            buffer = loss_config.get("buffer")
            ddp = loss_config.get("ddp")

            loss_func = PointDistGraphReg(
                point_estimator=point_estimator,
                distribution_loss=distribution_loss,
                graph_regularization=graph_regularization,
                buffer=buffer,
                ddp=ddp,
            )
            print(f"Using PointDistGraphReg loss:")
            print(f"  Point: {point_estimator.get('type') if point_estimator else 'N/A'}")
            print(f"  Dist: {distribution_loss.get('type') if distribution_loss else 'disabled'}")
            print(f"  Graph reg λ: {graph_regularization.get('lambda') if graph_regularization else 'disabled'}")
        elif loss_type == "logcosh":
            loss_func = LogCoshLoss(reduction="mean")
            print("Using LogCosh loss")
        elif loss_type == "mse":
            loss_func = nn.MSELoss()
            print("Using MSE loss")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        # Legacy string-based config (backward compatibility)
        if loss_config == "logcosh":
            loss_func = LogCoshLoss(reduction="mean")
            print("Using LogCosh loss")
        elif loss_config == "mse":
            loss_func = nn.MSELoss()
            print("Using MSE loss")
        else:
            raise ValueError(f"Unknown loss type: {loss_config}")

    print(f"Creating RegressionTask ({timestamp()})")
    checkpoint_path = wandb.config["model"].get("checkpoint_path")
    execution_mode = wandb.config["regression_task"].get("execution_mode", "training")
    print(f"Execution mode: {execution_mode}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        task = RegressionTask.load_from_checkpoint(
            checkpoint_path,
            map_location=device,
            model=model,
            cell_graph=dataset.cell_graph,
            loss_func=loss_func,
            device=device,
            optimizer_config=wandb_cfg["regression_task"]["optimizer"],
            lr_scheduler_config=wandb_cfg["regression_task"]["lr_scheduler"],
            batch_size=wandb_cfg["data_module"]["batch_size"],
            clip_grad_norm=wandb_cfg["regression_task"]["clip_grad_norm"],
            clip_grad_norm_max_norm=wandb_cfg["regression_task"][
                "clip_grad_norm_max_norm"
            ],
            inverse_transform=inverse_transform,
            plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
            plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
            grad_accumulation_schedule=wandb.config["regression_task"][
                "grad_accumulation_schedule"
            ],
            execution_mode=execution_mode,
        )
        print("Successfully loaded model weights from checkpoint")
    else:
        task = RegressionTask(
            model=model,
            cell_graph=dataset.cell_graph,
            optimizer_config=wandb_cfg["regression_task"]["optimizer"],
            lr_scheduler_config=wandb_cfg["regression_task"]["lr_scheduler"],
            batch_size=wandb_cfg["data_module"]["batch_size"],
            clip_grad_norm=wandb_cfg["regression_task"]["clip_grad_norm"],
            clip_grad_norm_max_norm=wandb_cfg["regression_task"][
                "clip_grad_norm_max_norm"
            ],
            plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
            loss_func=loss_func,
            grad_accumulation_schedule=wandb.config["regression_task"][
                "grad_accumulation_schedule"
            ],
            device=device,
            inverse_transform=inverse_transform,
            plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
            execution_mode=execution_mode,
        )

    # Try to compile the model for better performance (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        compile_mode = wandb.config.get("compile_mode", "default")
        if compile_mode is not None:
            try:
                print(
                    f"Attempting torch.compile optimization (PyTorch {torch.__version__})..."
                )
                print(f"  Mode: {compile_mode}, Dynamic: True")

                import torch._inductor.config as inductor_config

                inductor_config.precompilation_timeout_seconds = 2 * 60 * 60  # 2 hours
                print(f"  Set precompilation timeout to 2 hours")

                if compile_mode == "max-autotune":
                    inductor_config.max_autotune_subproc_result_timeout_seconds = 300.0
                    inductor_config.max_autotune_subproc_graceful_timeout_seconds = 10.0
                    inductor_config.max_autotune_subproc_terminate_timeout_seconds = (
                        20.0
                    )
                    print(f"  Set autotune subprocess timeout to 5 minutes")

                task.model = torch.compile(task.model, mode=compile_mode, dynamic=True)
                print(
                    f"✓ Successfully compiled model with torch.compile (mode={compile_mode}, dynamic=True)"
                )
                print("  Expected speedup: 2-3x for forward/backward passes")
            except Exception as e:
                print(f"⚠️ torch.compile not compatible with this model: {e}")
                print("Continuing without compilation optimization")

    model_base_path = osp.join(DATA_ROOT, "models/checkpoints")
    os.makedirs(model_base_path, exist_ok=True)
    checkpoint_dir = osp.join(model_base_path, group)

    # Checkpoint callbacks
    checkpoint_callback_best_mse = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/MSE",
        mode="min",
        filename=f"{run.id}-best-mse-{{epoch:02d}}-{{val/gene_interaction/MSE:.4f}}",
    )

    checkpoint_callback_best_pearson = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/Pearson",
        mode="max",
        filename=f"{run.id}-best-pearson-{{epoch:02d}}-{{val/gene_interaction/Pearson:.4f}}",
    )

    checkpoint_callback_last = ModelCheckpoint(
        dirpath=checkpoint_dir, save_last=True, filename=f"{run.id}-last"
    )

    print(f"devices: {devices}")
    torch.set_float32_matmul_precision("medium")
    num_nodes = get_slurm_nodes()

    print(f"Starting training ({timestamp()})")

    callbacks = [
        checkpoint_callback_best_mse,
        checkpoint_callback_best_pearson,
        checkpoint_callback_last,
    ]

    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,
        num_nodes=num_nodes,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
        precision=wandb.config.trainer.get("precision", "32-true"),
    )

    trainer.fit(model=task, datamodule=data_module)

    # Store metrics in variables first
    mse = trainer.callback_metrics["val/gene_interaction/MSE"].item()
    pearson = trainer.callback_metrics["val/gene_interaction/Pearson"].item()

    # Now finish wandb
    wandb.finish()

    # Return the already-stored metrics
    return (mse, pearson)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
