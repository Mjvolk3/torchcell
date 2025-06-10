# experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi
# [[experiments.005-kuzmin2018-tmi.scripts.hetero_cell_bipartite_dango_gi]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_hetero_cell_bipartite_dango_gi.py


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
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
    InverseCompose,
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
from torchcell.datamodules import CellDataModule
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.data import Neo4jCellDataset
import torch.distributed as dist
from torchcell.timestamp import timestamp
from torchcell.losses.logcosh import LogCoshLoss


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

    # BUG
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
        node_embeddings=node_embeddings,
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
    )

    # TODO - Check norms work - Start
    # After creating dataset and before creating data_module
    # Configure label normalization
    # norm_configs = {
    #     # "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
    #     "gene_interaction": {"strategy": "standard"}  # z-score: (x - mean) / std
    # }

    # # Create the transform
    # normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    # inverse_normalize_transform = InverseCompose([normalize_transform])

    # # Apply transform to dataset
    # dataset.transform = normalize_transform

    # # Pass transforms to the RegressionTask
    # forward_transform = normalize_transform
    # inverse_transform = inverse_normalize_transform

    # # Print normalization parameters
    # for label, stats in normalize_transform.stats.items():
    #     print(f"Normalization parameters for {label}:")
    #     for key, value in stats.items():
    #         if isinstance(value, (int, float)) and key != "strategy":
    #             print(f"  {key}: {value:.6f}")
    #         else:
    #             print(f"  {key}: {value}")

    # # Log standardization parameters to wandb
    # for label, stats in normalize_transform.stats.items():
    #     for key, value in stats.items():
    #         if isinstance(value, (int, float)) and key != "strategy":
    #             wandb.log({f"standardization/{label}/{key}": value})
    # TODO - Check norms work - End
    forward_transform = None
    inverse_transform = None

    seed = 42
    data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=wandb.config.data_module["batch_size"],
        random_seed=seed,
        num_workers=wandb.config.data_module["num_workers"],
        pin_memory=wandb.config.data_module["pin_memory"],
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

    if wandb.config.regression_task ["is_weighted_phenotype_loss"]:
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
            forward_transform=forward_transform,
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
            forward_transform=forward_transform,
            plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
        )

    model_base_path = osp.join(DATA_ROOT, "models/checkpoints")
    os.makedirs(model_base_path, exist_ok=True)
    checkpoint_dir = osp.join(model_base_path, group)

    # Update checkpoint configuration to monitor gene interaction metrics
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val/gene_interaction/MSE",  # Changed from combined metrics
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
    import random
    import time

    # Random delay between 0-90 seconds
    # delay = random.uniform(0, 90)
    # print(f"Delaying job start by {delay:.2f} seconds to avoid GPU contention")
    # time.sleep(delay)

    mp.set_start_method("spawn", force=True)
    main()
