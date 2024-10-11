# experiments/003-fit-int/scripts/deep_set
# [[experiments.003-fit-int.scripts.deep_set]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/deep_set
# Test file: experiments/003-fit-int/scripts/test_deep_set.py

import hashlib
import json
import logging
import os
import os.path as osp
import uuid
from torch.nn import ModuleDict
import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchcell.graph import SCerevisiaeGraph

import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    FungalUpDownTransformerDataset,
    OneHotGeneDataset,
    ProtT5Dataset,
    GraphEmbeddingDataset,
    Esm2Dataset,
    NucleotideTransformerDataset,
    CodonFrequencyDataset,
    CalmDataset,
    RandomEmbeddingDataset,
)
from torchcell.models import DeepSet, Mlp
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers.fit_int_deep_set_regression import RegressionTask
from torchcell.utils import format_scientific_notation
import torch.distributed as dist
import socket
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import PhenotypeProcessor


log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="deep_set",
)
def main(cfg: DictConfig) -> None:
    print("Starting Deep Set 🌋")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    print("wandb_cfg", wandb_cfg)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_slurm_job_id}_{hashed_cfg}"
    experiment_dir = osp.join(
        DATA_ROOT, "wandb-experiments", str(hostname_slurm_job_id)
    )
    os.makedirs(experiment_dir, exist_ok=True)
    wandb.init(
        mode="online",  # "online", "offline", "disabled"
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        tags=wandb_cfg["wandb"]["tags"],
        dir=experiment_dir,
    )

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    # Handle sql genome access error for ddp
    if torch.cuda.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        genome_data_root = osp.join(DATA_ROOT, f"data/sgd/genome_{rank}")
    else:
        # Fallback to default DATA_ROOT if not running in distributed mode or no GPU available
        genome_data_root = osp.join(DATA_ROOT, "data/sgd/genome")
        rank = 0  #

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=genome_data_root, overwrite=False)
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Graph data
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    graphs = {}
    if wandb.config.cell_dataset["graphs"] is None:
        graphs = None
    elif "physical" in wandb.config.cell_dataset["graphs"]:
        graphs = {"physical": graph.G_physical}
    elif "regulatory" in wandb.config.cell_dataset["graphs"]:
        graphs = {"regulatory": graph.G_regulatory}

    # Node embedding datasets
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

    print("=============")
    print("node.embeddings")
    print(node_embeddings)
    print("=============")

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=PhenotypeProcessor(),
    )

    # Base Module
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

    # Subset Module
    if wandb.config.data_module["is_perturbation_subset"]:
        data_module = PerturbationSubsetDataModule(
            cell_data_module=data_module,
            size=int(wandb.config.data_module["perturbation_subset_size"]),
            batch_size=wandb.config.data_module["batch_size"],
            num_workers=wandb.config.data_module["num_workers"],
            pin_memory=wandb.config.data_module["pin_memory"],
            prefetch=wandb.config.data_module["prefetch"],
            seed=seed,
        )
        data_module.setup()

    # Anytime data is accessed lmdb must be closed.
    input_dim = dataset.num_features["gene"]
    dataset.close_lmdb()

    model = ModuleDict(
        {
            "main": DeepSet(
                in_channels=input_dim,
                hidden_channels=wandb.config.models["graph"]["hidden_channels"],
                out_channels=wandb.config.models["graph"]["out_channels"],
                num_node_layers=wandb.config.models["graph"]["num_node_layers"],
                num_set_layers=wandb.config.models["graph"]["num_set_layers"],
                norm=wandb.config.models["graph"]["norm"],
                activation=wandb.config.models["graph"]["activation"],
                skip_node=wandb.config.models["graph"]["skip_node"],
                skip_set=wandb.config.models["graph"]["skip_set"],
                aggregation=wandb.config.models["graph"]["aggregation"],
            ),
            "top": Mlp(
                in_channels=wandb.config.models["graph"]["out_channels"],
                hidden_channels=wandb.config.models["pred_head"]["hidden_channels"],
                out_channels=wandb.config.models["pred_head"]["out_channels"],
                num_layers=wandb.config.models["pred_head"]["num_layers"],
                dropout_prob=wandb.config.models["pred_head"]["dropout_prob"],
                norm=wandb.config.models["pred_head"]["norm"],
                activation=wandb.config.models["pred_head"]["activation"],
                output_activation=wandb.config.models["pred_head"]["output_activation"],
            ),
        }
    )
    task = RegressionTask(
        model=model,
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        batch_size=wandb.config.data_module["batch_size"],
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{group}",
        save_top_k=1,
        monitor="val/loss",
        mode="min",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)

    num_devices = torch.cuda.device_count()
    if num_devices == 0:  # if there are no GPUs available, use 1 CPU
        devices = 1
    else:
        devices = num_devices

    torch.set_float32_matmul_precision("medium")
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=1, #devices
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
        # callbacks=[checkpoint_callback, TriggerWandbSyncLightningCallback()],
    )

    # Start the training
    trainer.fit(model=task, datamodule=data_module)


if __name__ == "__main__":
    main()
