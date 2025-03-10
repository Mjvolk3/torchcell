# experiments/003-fit-int/scripts/hetero_cell
# [[experiments.003-fit-int.scripts.hetero_cell]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/hetero_cell
# Test file: experiments/003-fit-int/scripts/test_hetero_cell.py


import hashlib
import json
import logging
import os
import os.path as osp
import uuid
import hydra
import lightning as L
import torch
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.transforms.regression_to_classification import (
    LabelNormalizationTransform,
    InverseCompose,
)
from torch_geometric.transforms import Compose

from torchcell.data.neo4j_cell import SubgraphRepresentation
from torchcell.trainers.fit_int_hetero_cell import RegressionTask
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import wandb
import socket
from torchcell.datasets import (
    CodonFrequencyDataset,
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
    NucleotideTransformerDataset,
    ProtT5Dataset,
    Esm2Dataset,
    CalmDataset,
    GraphEmbeddingDataset,
    RandomEmbeddingDataset,
)
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.graph import SCerevisiaeGraph
from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
from torchcell.models.hetero_cell import HeteroCell
from torchcell.losses.isomorphic_cell_loss import ICLoss
from torchcell.datamodules import CellDataModule
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.data import Neo4jCellDataset
import torch.distributed as dist
from torchcell.timestamp import timestamp

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
    config_name="hetero_cell",
)
def main(cfg: DictConfig) -> None:
    print("Starting HeteroCell Training 🐂")
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

    wandb.init(
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
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    graphs = {}
    if wandb.config.cell_dataset["graphs"] is None:
        graphs = None
    if "physical" in wandb.config.cell_dataset["graphs"]:
        graphs["physical"] = graph.G_physical
    if "regulatory" in wandb.config.cell_dataset["graphs"]:
        graphs["regulatory"] = graph.G_regulatory

    node_embeddings = {}
    # one hot gene
    if "one_hot_gene" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["one_hot_gene"] = OneHotGeneDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/one_hot_gene_embedding"),
            genome=genome,
        )
    # codon frequency
    if "codon_frequency" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )
    # codon embedding
    if "calm" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["calm"] = CalmDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/calm_embedding"),
            genome=genome,
            model_name="calm",
        )
    # fudt
    if "fudt_downstream" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["fudt_downstream"] = FungalUpDownTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
            genome=genome,
            model_name="species_downstream",
        )

    if "fudt_upstream" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["fudt_upstream"] = FungalUpDownTransformerDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/fudt_embedding"),
            genome=genome,
            model_name="species_upstream",
        )
    # nucleotide transformer
    if "nt_window_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_5979",
        )
    if "nt_window_5979_max" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_5979_max"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_5979_max",
        )
    if "nt_window_three_prime_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_three_prime_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="window_three_prime_5979",
        )
    if "nt_window_five_prime_5979" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_five_prime_5979"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_five_prime_5979",
        )
    if "nt_window_three_prime_300" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_three_prime_300"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_three_prime_300",
        )
    if "nt_window_five_prime_1003" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["nt_window_five_prime_1003"] = NucleotideTransformerDataset(
            root=osp.join(
                DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embedding"
            ),
            genome=genome,
            model_name="nt_window_five_prime_1003",
        )
    # protT5
    if "prot_T5_all" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["prot_T5_all"] = ProtT5Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
            genome=genome,
            model_name="prot_t5_xl_uniref50_all",
        )
    if "prot_T5_no_dubious" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["prot_T5_no_dubious"] = ProtT5Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embedding"),
            genome=genome,
            model_name="prot_t5_xl_uniref50_no_dubious",
        )
    # esm
    if "esm2_t33_650M_UR50D_all" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_all",
        )
    if "esm2_t33_650M_UR50D_no_dubious" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious",
        )
    if (
        "esm2_t33_650M_UR50D_no_dubious_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_dubious_uncharacterized",
        )
    if (
        "esm2_t33_650M_UR50D_no_uncharacterized"
        in wandb.config.cell_dataset["node_embeddings"]
    ):
        node_embeddings["esm2_t33_650M_UR50D_all"] = Esm2Dataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/esm2_embedding"),
            genome=genome,
            model_name="esm2_t33_650M_UR50D_no_uncharacterized",
        )
    # sgd_gene_graph
    if "normalized_chrom_pathways" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["normalized_chrom_pathways"] = GraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
            graph=graph.G_gene,
            model_name="normalized_chrom_pathways",
        )
    if "chrom_pathways" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["chrom_pathways"] = GraphEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/sgd_gene_graph_hot"),
            graph=graph.G_gene,
            model_name="chrom_pathways",
        )
    # random
    if "random_1000" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_1000"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_1000",
        )
    if "random_100" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_100"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_100",
        )
    if "random_10" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_10"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_10",
        )
    if "random_1" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["random_1"] = RandomEmbeddingDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/random_embedding"),
            genome=genome,
            model_name="random_1",
        )
    learnable_embedding = False
    if "learnable_embedding" == wandb.config.cell_dataset["node_embeddings"][0]:
        learnable_embedding = True

    graph_processor = SubgraphRepresentation()

    incidence_graphs = {}
    yeast_gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    if "metabolism" in wandb.config.cell_dataset["incidence_graphs"]:
        incidence_graphs["metabolism"] = yeast_gem.reaction_map

    print(EXPERIMENT_ROOT)
    with open(
        osp.join(EXPERIMENT_ROOT, "003-fit-int/queries/001-small-build.cql"), "r"
    ) as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs=graphs,
        incidence_graphs=incidence_graphs,
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=graph_processor,
    )

    # TODO - Check norms work - Start
    # After creating dataset and before creating data_module
    # Configure label normalization
    norm_configs = {
        "fitness": {"strategy": "standard"},  # z-score: (x - mean) / std
        "gene_interaction": {"strategy": "standard"},  # z-score: (x - mean) / std
    }

    # Create the transform
    normalize_transform = LabelNormalizationTransform(dataset, norm_configs)
    inverse_normalize_transform = InverseCompose([normalize_transform])

    # Apply transform to dataset
    dataset.transform = normalize_transform

    # Pass transforms to the RegressionTask
    forward_transform = normalize_transform
    inverse_transform = inverse_normalize_transform

    # Print normalization parameters
    for label, stats in normalize_transform.stats.items():
        print(f"Normalization parameters for {label}:")
        for key, value in stats.items():
            if isinstance(value, (int, float)) and key != "strategy":
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

    # Log standardization parameters to wandb
    for label, stats in normalize_transform.stats.items():
        for key, value in stats.items():
            if isinstance(value, (int, float)) and key != "strategy":
                wandb.log({f"standardization/{label}/{key}": value})
    # TODO - Check norms work - End

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
    dataset.close_lmdb()

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

    # Instantiate new HeteroCell model using wandb configuration.
    model = HeteroCell(
        gene_num=wandb.config["model"]["gene_num"],
        reaction_num=wandb.config["model"]["reaction_num"],
        metabolite_num=wandb.config["model"]["metabolite_num"],
        hidden_channels=wandb.config["model"]["hidden_channels"],
        out_channels=wandb.config["model"]["out_channels"],
        num_layers=wandb.config["model"]["num_layers"],
        dropout=wandb.config["model"]["dropout"],
        norm=wandb.config["model"]["norm"],
        activation=wandb.config["model"]["activation"],
        gene_encoder_config=wandb.config["model"]["gene_encoder_config"],
        metabolism_config=wandb.config["model"]["metabolism_config"],
        prediction_head_config=wandb.config["model"]["prediction_head_config"],
        gpr_conv_config=wandb.config["model"]["gpr_conv_config"],
    ).to(device)

    # Log parameter counts using the num_parameters property.
    param_counts = model.num_parameters
    print("Parameter counts:", param_counts)
    wandb.log(
        {
            "model/params_embeddings": param_counts.get("gene_embedding", 0)
            + param_counts.get("reaction_embedding", 0)
            + param_counts.get("metabolite_embedding", 0),
            "model/params_preprocessor": param_counts.get("preprocessor", 0),
            "model/params_convs": param_counts.get("convs", 0),
            "model/params_aggregators": param_counts.get("global_aggregator", 0)
            + param_counts.get("perturbed_aggregator", 0),
            "model/params_fitness_head": param_counts.get("fitness_head", 0),
            "model/params_interaction_head": param_counts.get("interaction_head", 0),
            "model/params_total": param_counts.get("total", 0),
        }
    )

    if wandb.config.regression_task["is_weighted_phenotype_loss"]:
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
        weights = torch.ones(2).to(device)

    loss_func = ICLoss(
        lambda_dist=wandb.config.regression_task["lambda_dist"],
        lambda_supcr=wandb.config.regression_task["lambda_supcr"],
        weights=weights,
    )

    print(f"Creating regression task ({timestamp()})")
    task = RegressionTask(
        model=model,
        cell_graph=dataset.cell_graph,  # pass cell graph here
        optimizer_config=wandb_cfg["regression_task"]["optimizer"],
        lr_scheduler_config=wandb_cfg["regression_task"]["lr_scheduler"],
        batch_size=wandb_cfg["data_module"]["batch_size"],
        clip_grad_norm=wandb_cfg["regression_task"]["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb_cfg["regression_task"]["clip_grad_norm_max_norm"],
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
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(model_base_path, group),
        save_top_k=1,
        monitor="val/transformed/combined/MSE",
        mode="min",
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
        callbacks=[checkpoint_callback],
        profiler=profiler,
        log_every_n_steps=10,
        overfit_batches=wandb.config.trainer["overfit_batches"],
    )

    trainer.fit(model=task, datamodule=data_module)

    # Store metrics in variables first
    mse = trainer.callback_metrics["val/transformed/combined/MSE"].item()
    pearson = trainer.callback_metrics["val/transformed/combined/Pearson"].item()

    # Now finish wandb
    wandb.finish()

    # Return the already-stored metrics
    return (mse, pearson)


if __name__ == "__main__":
    import multiprocessing as mp
    import random
    import time

    # Random delay between 0-90 seconds
    delay = random.uniform(0, 90)
    print(f"Delaying job start by {delay:.2f} seconds to avoid GPU contention")
    time.sleep(delay)

    mp.set_start_method("spawn", force=True)
    main()
