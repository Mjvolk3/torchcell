# experiments/005-kuzmin2018-tmi/scripts/dcell
# [[experiments.005-kuzmin2018-tmi.scripts.dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dcell
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_dcell.py


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
from torchcell.trainers.int_dango import RegressionTask as DangoRegressionTask
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.dcell import DCellModel
from torchcell.losses.dcell import DCellLoss
from torchcell.models.dango import Dango
from torchcell.losses.dango import (
    DangoLoss,
    PreThenPost,
    LinearUntilUniform,
    SCHEDULER_MAP,
)
from torchcell.datamodules import CellDataModule
from torchcell.metabolism.yeast_GEM import YeastGEM
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
        osp.join(EXPERIMENT_ROOT, "005-kuzmin2018-tmi/queries/001_small_build.cql"), "r"
    ) as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build"
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
        follow_batch.append("mutant_state")  # Include mutant_state for DCell

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

    # Determine device based on accelerator config - be more explicit
    if wandb.config.trainer["accelerator"] == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    log.info(f"Using device: {device}")
    devices = get_num_devices()

    print(f"Instantiating model ({timestamp()})")

    max_num_nodes = dataset.cell_graph["gene"].num_nodes

    # Parse activation function from config
    def get_activation_from_config(activation_name: str) -> nn.Module:
        """Get activation function from string name in config"""
        activation_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "linear": nn.Identity(),  # No activation
        }
        return activation_map.get(activation_name.lower(), nn.Tanh())

    # Get activation function from config - use direct access for required params
    activation_name = wandb.config.model["activation"]
    activation = get_activation_from_config(activation_name)

    # Get norm type and other parameters - use direct access for required params
    norm_type = wandb.config.model["norm_type"]
    norm_before_act = wandb.config.model["norm_before_act"]
    subsystem_num_layers = wandb.config.model["subsystem_num_layers"]
    init_range = wandb.config.model["init_range"]

    # Get embedding parameters if applicable
    learnable_embedding_dim = wandb.config.model.get("learnable_embedding_dim", None)

    # Always instantiate DCellModel based on the updated config
    print("Instantiating DCellModel")
    
    # Get required parameters from config - use direct access for required params
    subsystem_output_min = wandb.config.model["subsystem_output_min"]
    subsystem_output_max_mult = wandb.config.model["subsystem_output_max_mult"]
    output_size = wandb.config.model["output_size"]

    # Create model first 
    model = DCellModel(
        gene_num=max_num_nodes,
        subsystem_output_min=subsystem_output_min,
        subsystem_output_max_mult=subsystem_output_max_mult,
        output_size=output_size,
        norm_type=norm_type,
        norm_before_act=norm_before_act,
        subsystem_num_layers=subsystem_num_layers,
        activation=activation,
        init_range=init_range,
        learnable_embedding_dim=learnable_embedding_dim,
    )
    
    # Make a dummy forward pass to initialize parameters before moving to device
    # This ensures all dynamically created parameters will be moved to the right device
    with torch.no_grad():
        # Create minimal dummy data
        dummy_batch = HeteroData()
        dummy_batch["gene"] = {}
        dummy_batch["gene"]["x"] = torch.ones(1, 1)
        dummy_batch["gene_ontology"] = {}
        dummy_batch["gene_ontology"]["x"] = torch.ones(1, 1)
        dummy_batch["gene_ontology"]["mutant_state"] = torch.zeros(1, 3)
        dummy_batch["gene"]["phenotype_values"] = torch.zeros(1, 1)
        dummy_batch.num_graphs = 1
        
        try:
            # Try to initialize with a dummy forward pass
            # This will trigger parameter creation for all dynamic modules
            model(dataset.cell_graph.to('cpu'), dummy_batch)
            print("Successfully pre-initialized model with dummy forward pass")
        except Exception as e:
            print(f"Pre-initialization failed (this is expected): {str(e)}")
    
    # Now move to device AFTER all parameters are created
    model = model.to(device)
    
    # Ensure dataset cell_graph is also on the same device
    dataset.cell_graph = dataset.cell_graph.to(device)
    
    # Verify model device
    try:
        param_device = next(model.parameters()).device
        print(f"Model parameters are on device: {param_device}")
        # No need to check if param_device != device since we just moved it
    except StopIteration:
        print("WARNING: Model has no parameters yet, will be initialized during forward pass")

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
    
    # No need to move dataset.cell_graph again - already moved above
    print(f"Using dataset.cell_graph on device: {dataset.cell_graph['gene'].x.device}")

    # Configure data module to use the same device
    if hasattr(data_module, 'device'):
        data_module.device = device
    
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
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,
        num_nodes=num_nodes,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback_best, checkpoint_callback_last],
        profiler=profiler,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
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
    import random
    import time

    mp.set_start_method("spawn", force=True)
    main()
