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
import torch.distributed as dist
import torch._inductor.config as inductor_config
import torch._dynamo
import socket
import sys
import random
import time
import multiprocessing as mp
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb
from torch_geometric.transforms import Compose

# Torchcell imports
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data.graph_processor import Perturbation, DCellGraphProcessor
from torchcell.trainers.int_dcell import RegressionTask as DCellRegressionTask
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.dcell import DCell
from torchcell.losses.dcell import DCellLoss
from torchcell.models.dango import Dango
from torchcell.losses.dango import (
    DangoLoss,
    PreThenPost,
    LinearUntilUniform,
    SCHEDULER_MAP,
)
from torchcell.datamodules import CellDataModule
from torchcell.data import Neo4jCellDataset
from torchcell.timestamp import timestamp
from torchcell.graph import build_gene_multigraph
from torchcell.graph import (
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
    filter_by_contained_genes,
)


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
    config_name="dcell_kuzmin2018_tmi",
)
def main(cfg: DictConfig) -> None:
    print("Starting DCell Training 🧬")
    os.environ["WANDB__SERVICE_WAIT"] = "600"
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
    
    print(f"DEBUG: About to init wandb - experiment_dir: {experiment_dir}")
    print(f"DEBUG: WANDB_MODE = {WANDB_MODE}")
    
    try:
        run = wandb.init(
            mode=WANDB_MODE,
            project=wandb_cfg["wandb"]["project"],
            config=wandb_cfg,
            group=group,
            tags=wandb_cfg["wandb"]["tags"],
            dir=experiment_dir,
            name=f"run_{group}",
        )
        print("DEBUG: wandb.init successful")

        wandb_logger = WandbLogger(
            project=wandb_cfg["wandb"]["project"],
            log_model=True,
            save_dir=experiment_dir,
            name=f"run_{group}",
        )
        print("DEBUG: WandbLogger created")
        
        print(f"Process initialized - PID: {os.getpid()}")
    except Exception as e:
        print(f"ERROR during wandb initialization: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("DEBUG: After wandb initialization block", flush=True)
    
    try:
        print(f"DEBUG: Checking CUDA availability: {torch.cuda.is_available()}", flush=True)
        print(f"DEBUG: Checking dist.is_initialized: {dist.is_initialized()}", flush=True)
        
        if torch.cuda.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            genome_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
            go_root = osp.join(DATA_ROOT, f"data/go/go_{rank}")
            print(f"DEBUG: Using distributed paths - rank {rank}")
        else:
            genome_root = osp.join(DATA_ROOT, "data/sgd/genome")
            go_root = osp.join(DATA_ROOT, "data/go")
            rank = 0
            print(f"DEBUG: Using non-distributed paths")

        print(f"DEBUG: About to create genome - genome_root: {genome_root}, go_root: {go_root}")
    except Exception as e:
        print(f"ERROR in path setup: {e}")
        import traceback
        traceback.print_exc()
        raise
    genome = SCerevisiaeGenome(
        genome_root=genome_root, go_root=go_root, overwrite=False
    )
    print("DEBUG: Genome created successfully")
    genome.drop_empty_go()

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Check which type of graph to use based on config
    # The new format uses incidence_graphs="go" to indicate GO hierarchy
    # and graphs=null to indicate no STRING networks
    incidence_type = wandb.config.cell_dataset.get("incidence_graphs", None)
    graph_names = wandb.config.cell_dataset.get("graphs", [])

    if isinstance(graph_names, list) and not graph_names:
        graph_names = []

    # Initialize variables
    gene_multigraph = None
    incidence_graphs = {}

    # Check if we need to use GO graph
    if incidence_type == "go":
        print("Using Gene Ontology hierarchy")
        G_go = graph.G_go.copy()

        # Apply DCell-specific GO graph filters
        date_filter = wandb.config.model.get("go_date_filter", None)
        if date_filter:
            G_go = filter_by_date(G_go, date_filter)
            print(f"After date filter ({date_filter}): {G_go.number_of_nodes()}")

        G_go = filter_go_IGI(G_go)
        print(f"After IGI filter: {G_go.number_of_nodes()}")

        G_go = filter_redundant_terms(G_go)
        print(f"After redundant filter: {G_go.number_of_nodes()}")

        min_genes = wandb.config.model.get("go_min_genes", 4)
        G_go = filter_by_contained_genes(G_go, n=min_genes, gene_set=genome.gene_set)
        print(
            f"After containment filter (min_genes={min_genes}): {G_go.number_of_nodes()}"
        )

        # Add GO graph to incidence graphs
        incidence_graphs["gene_ontology"] = G_go

        # Use DCellGraphProcessor when using GO hierarchy
        graph_processor = DCellGraphProcessor()

        # Set gene_multigraph to None when using GO hierarchy
        gene_multigraph = None

    # Build gene multigraph if using STRING networks
    elif graph_names:
        print(f"Using STRING networks: {graph_names}")
        gene_multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
        # Use standard Perturbation graph processor for STRING networks
        graph_processor = Perturbation()
    else:
        # If neither GO nor STRING networks are specified
        raise ValueError(
            "Either incidence_graphs='go' or valid graphs list must be specified"
        )

    print(EXPERIMENT_ROOT)
    with open(
        osp.join(EXPERIMENT_ROOT, "006-kuzmin-tmi/queries/001_small_build.cql"), "r"
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
        node_embeddings=wandb.config.cell_dataset["node_embeddings"],
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
    )

    seed = 42
    # Define follow_batch based on whether we're using DCellGraphProcessor
    follow_batch = ["perturbation_indices"]
    if isinstance(graph_processor, DCellGraphProcessor):
        follow_batch.append(
            "go_gene_strata_state"
        )  # Include go_gene_strata_state for DCell

    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=seed,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
        follow_batch=follow_batch,
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
            follow_batch=follow_batch,
        )
        data_module.setup()

    dataset.close_lmdb()

    # Determine device based on accelerator config - only support CPU and CUDA
    if wandb.config.trainer["accelerator"] == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        # Force CPU if MPS would be selected
        if (
            wandb.config.trainer["accelerator"] == "gpu"
            and torch.backends.mps.is_available()
        ):
            print("MPS detected but not supported. Using CPU instead.")

    log.info(f"Using device: {device}")
    devices = get_num_devices()

    print(f"Instantiating model ({timestamp()})")

    max_num_nodes = dataset.cell_graph["gene"].num_nodes

    # Always instantiate DCell based on the updated config
    print("Instantiating DCell")

    # Get required parameters from config - use direct access for required params
    subsystem_output_min = wandb.config.model["subsystem_output_min"]
    subsystem_output_max_mult = wandb.config.model["subsystem_output_max_mult"]
    output_size = wandb.config.model["output_size"]

    # Explicitly move dataset cell_graph to the correct device before model initialization
    dataset.cell_graph = dataset.cell_graph.to(device)

    # Select model version based on config
    model_version = wandb.config.model.get("model_version", "dcell")

    if model_version == "dcell_opt":
        print("Using optimized DCell (DCellOpt) for better torch.compile compatibility")
        from torchcell.models.dcell_opt import DCellOpt

        model = DCellOpt(
            hetero_data=dataset.cell_graph,
            min_subsystem_size=subsystem_output_min,
            subsystem_ratio=subsystem_output_max_mult,
            output_size=output_size,
        )
    else:
        print("Using original DCell model")
        model = DCell(
            hetero_data=dataset.cell_graph,
            min_subsystem_size=subsystem_output_min,
            subsystem_ratio=subsystem_output_max_mult,
            output_size=output_size,
        )

    # Explicitly move model to device
    model = model.to(device)

    # Log parameter counts using the num_parameters property.
    param_counts = model.num_parameters
    print("Parameter counts:", param_counts)
    wandb.log(
        {
            "model/params_dcell": param_counts.get("dcell", 0),
            "model/params_dcell_linear": param_counts.get("dcell_linear", 0),
            "model/params_subsystems": param_counts.get("subsystems", 0),
            "model/params_total": param_counts.get("total", 0),
            "model/num_go_terms": param_counts.get("num_go_terms", 0),
            "model/num_subsystems": param_counts.get("num_subsystems", 0),
        }
    )

    # Configure DCellLoss
    alpha = wandb.config.regression_task.get("dcell_loss", {}).get("alpha", 0.3)
    use_auxiliary_losses = wandb.config.regression_task.get("dcell_loss", {}).get(
        "use_auxiliary_losses", True
    )

    loss_func = DCellLoss(alpha=alpha, use_auxiliary_losses=use_auxiliary_losses)

    # Always use DCellRegressionTask based on the updated config
    RegressionTask = DCellRegressionTask

    print(f"Creating RegressionTask ({timestamp()})")
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
            inverse_transform=None,
            forward_transform=None,
            plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
            plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
            grad_accumulation_schedule=wandb.config["regression_task"][
                "grad_accumulation_schedule"
            ],
        )
        print(f"Successfully loaded model weights from checkpoint")
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
            inverse_transform=None,
            forward_transform=None,
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
                # Increase the overall precompilation budget to 2 hours (default is 3600s = 1 hour)
                inductor_config.precompilation_timeout_seconds = 2 * 60 * 60  # 2 hours
                print(f"  Set precompilation timeout to 2 hours")

                # Enable capture of scalar outputs to handle .item() operations better
                torch._dynamo.config.capture_scalar_outputs = True
                print("  Enabled capture_scalar_outputs for better .item() handling")
                
                # Set recompile limits from config or use defaults
                recompile_limit = cfg.get("recompile_limit", 32)
                accumulated_limit = cfg.get("accumulated_recompile_limit", 64)
                
                # Increase recompile limit to handle shape variations better
                torch._dynamo.config.recompile_limit = recompile_limit
                print(f"  Set recompile limit to {recompile_limit} (default is 8)")
                
                # Also increase accumulated limit for overall recompilations
                torch._dynamo.config.accumulated_recompile_limit = accumulated_limit
                print(f"  Set accumulated recompile limit to {accumulated_limit}")

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
                # Try with fullgraph=True to detect graph breaks
                task.model = torch.compile(
                    task.model, mode=compile_mode, dynamic=True, fullgraph=False
                )
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
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/MSE",  # Monitor MSE for gene interactions
        mode="min",
        filename=f"{run.id}-best-{{epoch:02d}}-{{val/gene_interaction/MSE:.4f}}",
    )
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=checkpoint_dir, save_last=True, filename=f"{run.id}-last"
    )

    print(f"devices: {devices}")
    torch.set_float32_matmul_precision("medium")
    num_nodes = get_slurm_nodes()
    profiler = None
    print(f"Starting training ({timestamp()})")

    # Add progress bar callback for better visibility
    from lightning.pytorch.callbacks import TQDMProgressBar

    progress_bar = TQDMProgressBar(refresh_rate=1)  # Update every iteration

    # Use the strategy from config
    strategy = wandb.config.trainer["strategy"]
    print(f"Using strategy: {strategy}")

    trainer = L.Trainer(
        strategy=strategy,
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,
        num_nodes=num_nodes,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, progress_bar],
        profiler=profiler,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
        enable_progress_bar=True,  # Explicitly enable progress bar
        enable_model_summary=True,  # Keep model summary
        precision=wandb.config.trainer.get(
            "precision", "32-true"
        ),  # Use precision from config
    )

    print(f"Starting trainer.fit - PID: {os.getpid()}")
    trainer.fit(model=task, datamodule=data_module)

    # Store metrics in variables first
    mse = trainer.callback_metrics["val/gene_interaction/MSE"].item()
    pearson = trainer.callback_metrics["val/gene_interaction/Pearson"].item()

    # Now finish wandb
    wandb.finish()

    # Return the already-stored metrics
    return (mse, pearson)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
