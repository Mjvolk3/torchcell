# experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_test
# [[experiments.006-kuzmin-tmi.scripts.hetero_cell_bipartite_dango_gi_test]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/hetero_cell_bipartite_dango_gi_test
# Test file: experiments/006-kuzmin-tmi/scripts/test_hetero_cell_bipartite_dango_gi_test.py


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
from torchcell.datamodules import CellDataModule
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.data import Neo4jCellDataset
import torch.distributed as dist
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
    config_name="hetero_cell_bipartite_dango_gi_test",
)
def main(cfg: DictConfig):
    print("Starting GeneInteractionDango Testing 🧪")
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
        project=wandb_cfg["wandb"]["project"] + "_test",
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"] + ["test"],
        dir=experiment_dir,
        name=f"test_{group}",
    )

    wandb_logger = WandbLogger(
        project=wandb_cfg["wandb"]["project"] + "_test",
        log_model=False,
        save_dir=experiment_dir,
        name=f"test_{group}",
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
            bin_transform = COOLabelBinningTransform(dataset, bin_config, norm_transform)
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
        prefetch_factor=wandb.config.data_module["prefetch_factor"]
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

    print(f"Loading model from checkpoint ({timestamp()})")
    checkpoint_path = wandb.config["model"].get("checkpoint_path")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found or doesn't exist: {checkpoint_path}")

    # First, create the model architecture matching the checkpoint
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

    # Set up loss function
    if wandb.config.regression_task["loss"] == "icloss":
        loss_func = ICLoss(
            lambda_dist=wandb.config.regression_task["lambda_dist"],
            lambda_supcr=wandb.config.regression_task["lambda_supcr"],
            weights=torch.ones(1).to(device),  # Simple weights for testing
        )
    elif wandb.config.regression_task["loss"] == "logcosh":
        loss_func = LogCoshLoss(reduction="mean")

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
        clip_grad_norm_max_norm=wandb_cfg["regression_task"]["clip_grad_norm_max_norm"],
        inverse_transform=inverse_transform,
        plot_every_n_epochs=wandb.config["regression_task"]["plot_every_n_epochs"],
        plot_sample_ceiling=wandb.config["regression_task"]["plot_sample_ceiling"],
        grad_accumulation_schedule=wandb.config["regression_task"]["grad_accumulation_schedule"],
        weights_only=False,  # Allow loading of PyTorch Geometric objects
    )
    print("Successfully loaded model from checkpoint")

    print(f"devices: {devices}")
    torch.set_float32_matmul_precision("medium")
    num_nodes = get_slurm_nodes()
    profiler = None
    print(f"Starting testing ({timestamp()})")
    trainer = L.Trainer(
        accelerator=wandb.config.trainer["accelerator"],
        devices=1,  # Use single device for testing
        num_nodes=1,
        logger=wandb_logger,
        max_epochs=1,  # Not used for testing
        callbacks=[],  # No callbacks needed for testing
        profiler=profiler,
        log_every_n_steps=10,
        overfit_batches=0,
    )

    # Run test instead of fit
    test_results = trainer.test(model=task, datamodule=data_module)

    # Print test results
    print("\nTest Results:")
    for key, value in test_results[0].items():
        print(f"{key}: {value:.6f}")

    # Now finish wandb
    wandb.finish()

    # Return test metrics
    return test_results[0]


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()