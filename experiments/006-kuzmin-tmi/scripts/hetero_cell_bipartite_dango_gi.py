# experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi
# [[experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi
# Test file: experiments/006-kuzmin-tmi/scripts/test_hetero_cell_bipartite_dango_gi.py


import hashlib
import json
import logging
import os
import os.path as osp
import uuid
import hydra
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
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.trainers.int_hetero_cell import RegressionTask
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb
import socket
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.hetero_cell_bipartite_dango_gi import GeneInteractionDango
from torchcell.graph.graph import build_gene_multigraph
from torchcell.losses.isomorphic_cell_loss import ICLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.datamodules import CellDataModule
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.data import Neo4jCellDataset
import torch.distributed as dist
from datetime import timedelta
from torchcell.timestamp import timestamp
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.scheduler.cosine_annealing_warmup import CosineAnnealingWarmupRestarts


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
    config_name="hetero_cell_bipartite_dango_gi",
)
def main(cfg: DictConfig) -> None:
    print("Starting GeneInteractionDango Training 🔫")
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Set distributed timeout to 2 hours if using DDP
    if dist.is_available() and dist.is_initialized():
        # This is already initialized by Lightning, we can't change it
        pass
    else:
        # Set environment variable for when dist is initialized later
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

    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
        go_root = osp.join(DATA_ROOT, f"data/go/go_{rank}")
    else:
        genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
        go_root = osp.join(DATA_ROOT, "data/go")
        rank = 0

    # BUG if genome is is overwritten we get issue. Might be able to remove with locking
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

    # HACK
    # don't think we even need this for now the learnable bit is disgusting...
    # git rid of it.
    # Build node embeddings using the NodeEmbeddingBuilder
    node_embeddings = NodeEmbeddingBuilder.build(
        embedding_names=wandb.config.cell_dataset["node_embeddings"],
        data_root=DATA_ROOT,
        genome=genome,
        graph=graph,
    )

    # Check if learnable embedding is requested
    learnable_embedding = NodeEmbeddingBuilder.check_learnable_embedding(
        wandb.config.cell_dataset["node_embeddings"]
    )
    # HACK

    graph_processor = SubgraphRepresentation()

    incidence_graphs = {}
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    if "metabolism_hypergraph" in wandb.config.cell_dataset["incidence_graphs"]:
        incidence_graphs["metabolism_hypergraph"] = yeast_gem.reaction_map
    elif "metabolism_bipartite" in wandb.config.cell_dataset["incidence_graphs"]:
        incidence_graphs["metabolism_bipartite"] = yeast_gem.bipartite_graph

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
            transform=None,  # No transform initially
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
    print(f"Dataset Length: {len(dataset)}")

    seed = 42
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
            gene_subsets={"metabolism": yeast_gem.gene_set},
        )
        data_module.setup()

    # TODO will need for when adding embeddings
    # if not wandb.config.cell_dataset.get("learnable_embedding", False):
    #     input_dim = dataset.num_features["gene"]
    # else:
    #     input_dim = wandb.config.cell_dataset["learnable_embedding_input_channels"]
    # Move this after training is complete
    # dataset.close_lmdb()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)
    devices = get_num_devices()
    # edge_types = [
    #     ("gene", f"{name}_interaction", "gene")
    #     for name in wandb.config.cell_dataset["graphs"]
    # ]

    print(f"Instantiating model ({timestamp()})")
    # Instantiate new IsomorphicCell model
    gene_encoder_config = dict(wandb.config["model"]["gene_encoder_config"])
    if any(
        "learnable" in emb for emb in wandb.config["cell_dataset"]["node_embeddings"]
    ):
        gene_encoder_config.update(
            {
                "embedding_type": "learnable",
                "max_num_nodes": dataset.cell_graph["gene"].num_nodes,
                "learnable_embedding_input_channels": wandb.config["cell_dataset"][
                    "learnable_embedding_input_channels"
                ],
            }
        )

    # Use the gene_multigraph that was already built for the dataset

    # Get local predictor config
    local_predictor_config = dict(
        wandb.config["model"].get("local_predictor_config", {})
    )

    # Instantiate new GeneInteractionDango model using wandb configuration.
    model = GeneInteractionDango(
        gene_num=wandb.config["model"]["gene_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        num_layers=wandb.config["model"]["num_layers"],
        gene_multigraph=gene_multigraph,
        dropout=wandb.config["model"]["dropout"],
        norm=wandb.config["model"]["norm"],
        activation=wandb.config["model"]["activation"],
        gene_encoder_config=gene_encoder_config,
        local_predictor_config=local_predictor_config,
    ).to(device)

    # Log parameter counts using the num_parameters property.
    param_counts = model.num_parameters
    print("Parameter counts:", param_counts)
    wandb.log(
        {
            "model/params_gene_embedding": param_counts.get("gene_embedding", 0),
            "model/params_preprocessor": param_counts.get("preprocessor", 0),
            "model/params_convs": param_counts.get("convs", 0),
            "model/params_gene_interaction_predictor": param_counts.get(
                "gene_interaction_predictor", 0
            ),
            "model/params_global_aggregator": param_counts.get("global_aggregator", 0),
            "model/params_global_interaction_predictor": param_counts.get(
                "global_interaction_predictor", 0
            ),
            "model/params_gate_mlp": param_counts.get("gate_mlp", 0),
            "model/params_total": param_counts.get("total", 0),
        }
    )

    if wandb.config.regression_task["is_weighted_phenotype_loss"]:
        # BUG
        phenotype_counts = {}
        for phase in ["train", "val", "test"]:
            phenotypes = getattr(data_module.index_details, phase).phenotype_label_index
            temp_counts = {k: v.count for k, v in phenotypes.items()}
            for k, v in temp_counts.items():
                phenotype_counts[k] = phenotype_counts.get(k, 0) + v
        weights = torch.tensor(
            [1 - v / sum(phenotype_counts.values()) for v in phenotype_counts.values()]
        ).to(device)
    else:
        weights = torch.ones(1).to(device)

    if wandb.config.regression_task["loss"] == "icloss":
        loss_func = ICLoss(
            lambda_dist=wandb.config.regression_task["lambda_dist"],
            lambda_supcr=wandb.config.regression_task["lambda_supcr"],
            weights=weights,
        )
    elif wandb.config.regression_task["loss"] == "mle_dist_supcr":
        # Get loss configuration
        loss_config = wandb.config.regression_task.get("loss_config", {})

        loss_func = MleDistSupCR(
            # Lambda weights
            lambda_mse=wandb.config.regression_task.get("lambda_mse", 1.0),
            lambda_dist=wandb.config.regression_task.get("lambda_dist", 0.1),
            lambda_supcr=wandb.config.regression_task.get("lambda_supcr", 0.001),
            # Component-specific parameters
            dist_bandwidth=loss_config.get("dist_bandwidth", 2.0),
            supcr_temperature=loss_config.get("supcr_temperature", 0.1),
            embedding_dim=wandb.config.model["hidden_channels"],
            # Buffer configuration
            use_buffer=loss_config.get("use_buffer", True),
            buffer_size=loss_config.get("buffer_size", 256),
            min_samples_for_dist=loss_config.get("min_samples_for_dist", 64),
            min_samples_for_supcr=loss_config.get("min_samples_for_supcr", 32),
            # DDP configuration
            use_ddp_gather=loss_config.get("use_ddp_gather", True),
            gather_interval=loss_config.get("gather_interval", 1),
            # Adaptive weighting
            use_adaptive_weighting=loss_config.get("use_adaptive_weighting", True),
            warmup_epochs=loss_config.get("warmup_epochs", 100),
            stable_epoch=loss_config.get("stable_epoch", 500),
            # Temperature scheduling
            use_temp_scheduling=loss_config.get("use_temp_scheduling", True),
            init_temperature=loss_config.get("init_temperature", 1.0),
            final_temperature=loss_config.get("final_temperature", 0.1),
            temp_schedule=loss_config.get("temp_schedule", "exponential"),
            # Other parameters
            weights=weights,
            max_epochs=loss_config.get("max_epochs", wandb.config.trainer["max_epochs"]),
        )
    elif wandb.config.regression_task["loss"] == "mle_wass_supcr":
        loss_config = wandb.config.regression_task.get("loss_config", {})

        loss_func = MleWassSupCR(
            # Lambda weights (passed individually, not as dict)
            lambda_mse=wandb.config.regression_task.get("lambda_mse", 1.0),
            lambda_wasserstein=wandb.config.regression_task.get("lambda_wasserstein", 0.1),
            lambda_supcr=wandb.config.regression_task.get("lambda_supcr", 0.001),
            # Wasserstein-specific parameters
            wasserstein_blur=loss_config.get("wasserstein_blur", 0.05),
            wasserstein_p=loss_config.get("wasserstein_p", 2),
            wasserstein_scaling=loss_config.get("wasserstein_scaling", 0.9),
            min_samples_for_wasserstein=loss_config.get("min_samples_for_wasserstein", 512),
            # SupCR-specific parameters
            supcr_temperature=loss_config.get("supcr_temperature", 0.1),
            embedding_dim=wandb.config.model["hidden_channels"],
            # Buffer configuration
            use_buffer=loss_config.get("use_buffer", True),
            buffer_size=loss_config.get("buffer_size", 256),
            min_samples_for_supcr=loss_config.get("min_samples_for_supcr", 32),
            # DDP configuration
            use_ddp_gather=loss_config.get("use_ddp_gather", True),
            gather_interval=loss_config.get("gather_interval", 1),
            # Adaptive weighting
            use_adaptive_weighting=loss_config.get("use_adaptive_weighting", True),
            warmup_epochs=loss_config.get("warmup_epochs", 100),
            stable_epoch=loss_config.get("stable_epoch", 500),
            # Temperature scheduling
            use_temp_scheduling=loss_config.get("use_temp_scheduling", True),
            init_temperature=loss_config.get("init_temperature", 1.0),
            final_temperature=loss_config.get("final_temperature", 0.1),
            temp_schedule=loss_config.get("temp_schedule", "exponential"),
            # Training duration for loss scheduling
            max_epochs=loss_config.get("max_epochs", wandb.config.trainer["max_epochs"]),
        )
    elif wandb.config.regression_task["loss"] == "logcosh":
        loss_func = LogCoshLoss(reduction="mean")

    print(f"Creating GeneInteractionTask ({timestamp()})")
    checkpoint_path = wandb.config["model"].get("checkpoint_path")

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load the checkpoint with all required arguments
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
        )
        print("Successfully loaded model weights from checkpoint")
    else:
        # Create a fresh task with the model
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
        )

    # Try to compile the model for better performance (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        compile_mode = wandb.config.get(
            "compile_mode", "default"
        )  # Can be null, "default", "reduce-overhead", "max-autotune"
        if compile_mode is not None:
            try:
                print(
                    f"Attempting torch.compile optimization (PyTorch {torch.__version__})..."
                )
                print(f"  Mode: {compile_mode}, Dynamic: True")

                # For PyTorch 2.8.0, inductor config is now available
                import torch._inductor.config as inductor_config

                # Increase the overall precompilation budget to 2 hours (default is 3600s = 1 hour)
                inductor_config.precompilation_timeout_seconds = 2 * 60 * 60  # 2 hours
                print(f"  Set precompilation timeout to 2 hours")

                # If using max-autotune mode, also increase autotune subprocess timeouts
                if compile_mode == "max-autotune":
                    inductor_config.max_autotune_subproc_result_timeout_seconds = (
                        300.0  # 5 min (was 60s)
                    )
                    inductor_config.max_autotune_subproc_graceful_timeout_seconds = 10.0
                    inductor_config.max_autotune_subproc_terminate_timeout_seconds = (
                        20.0
                    )
                    print(f"  Set autotune subprocess timeout to 5 minutes")

                # Compile the model within the task
                # Use dynamic=True since batch sizes and gene counts vary
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

    # Update checkpoint configuration to monitor gene interaction metrics
    # Checkpoint for best MSE
    checkpoint_callback_best_mse = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/MSE",
        mode="min",
        filename=f"{run.id}-best-mse-{{epoch:02d}}-{{val/gene_interaction/MSE:.4f}}",
    )

    # Checkpoint for best Pearson correlation
    checkpoint_callback_best_pearson = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/Pearson",
        mode="max",  # Pearson correlation should be maximized
        filename=f"{run.id}-best-pearson-{{epoch:02d}}-{{val/gene_interaction/Pearson:.4f}}",
    )

    checkpoint_callback_last = ModelCheckpoint(
        dirpath=checkpoint_dir, save_last=True, filename=f"{run.id}-last"
    )

    print(f"devices: {devices}")
    torch.set_float32_matmul_precision("medium")
    num_nodes = get_slurm_nodes()
    profiler = None
    print(f"Starting training ({timestamp()})")
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,
        num_nodes=num_nodes,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[
            checkpoint_callback_best_mse,
            checkpoint_callback_best_pearson,
            checkpoint_callback_last,
        ],
        profiler=profiler,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
        precision=wandb.config.trainer.get(
            "precision", "32-true"
        ),  # Use precision from config
        # limit_val_batches=0,  # FLAG
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
