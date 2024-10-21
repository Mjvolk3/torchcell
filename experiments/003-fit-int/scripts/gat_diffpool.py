# experiments/003-fit-int/scripts/gat_diffpool
# [[experiments.003-fit-int.scripts.gat_diffpool]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/003-fit-int/scripts/gat_diffpool
# Test file: experiments/003-fit-int/scripts/test_gat_diffpool.py

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
from lightning.pytorch.callbacks import GradientAccumulationScheduler
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
from torchcell.models import Mlp
from torchcell.models.gat_diffpool import GatDiffPool
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers.fit_int_gat_diffpool_regression import RegressionTask
from torchcell.utils import format_scientific_notation
import torch.distributed as dist
import socket
from torchcell.datamodels.fitness_composite_conversion import CompositeFitnessConverter
from torchcell.data import MeanExperimentDeduplicator
from torchcell.data import GenotypeAggregator
from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
from torchcell.data import Neo4jCellDataset
from torchcell.data.neo4j_cell import PhenotypeProcessor
from lightning.pytorch.profilers import AdvancedProfiler
from typing import Any
from lightning.pytorch.profilers import AdvancedProfiler
import cProfile

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


class CustomAdvancedProfiler(AdvancedProfiler):
    def __init__(self, dirpath: str, filename: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dirpath = dirpath
        self.filename = filename
        self.profiler = cProfile.Profile()

    def start(self, action_name: str) -> None:
        super().start(action_name)
        self.profiler.enable()

    def stop(self, action_name: str) -> None:
        self.profiler.disable()
        super().stop(action_name)

    def summary(self) -> str:
        os.makedirs(self.dirpath, exist_ok=True)
        file_path = os.path.join(self.dirpath, self.filename)
        self.profiler.dump_stats(file_path)
        return f"Profiler output saved to {file_path}"


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="gat_diffpool",
)
def main(cfg: DictConfig) -> None:
    print("Starting GatDiffPool 🌋")
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
        mode="offline",  # "online", "offline", "disabled"
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
    if "physical" in wandb.config.cell_dataset["graphs"]:
        graphs["physical"] = graph.G_physical
    if "regulatory" in wandb.config.cell_dataset["graphs"]:
        graphs["regulatory"] = graph.G_regulatory

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
        gene_set=genome.gene_set,
        graphs=graphs,
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
    max_num_nodes = len(dataset.gene_set)
    dataset.close_lmdb()
    num_graphs = len(wandb.config.cell_dataset["graphs"])

    model = ModuleDict(
        {
            "main": GatDiffPool(
                in_channels=input_dim,
                initial_gat_hidden_channels=wandb.config.models["graph"][
                    "initial_gat_hidden_channels"
                ],
                initial_gat_out_channels=wandb.config.models["graph"][
                    "initial_gat_out_channels"
                ],
                diffpool_hidden_channels=wandb.config.models["graph"][
                    "diffpool_hidden_channels"
                ],
                diffpool_out_channels=wandb.config.models["graph"][
                    "diffpool_out_channels"
                ],
                num_initial_gat_layers=wandb.config.models["graph"][
                    "num_initial_gat_layers"
                ],
                num_diffpool_layers=wandb.config.models["graph"]["num_diffpool_layers"],
                cluster_size_decay_factor=wandb.config.models["graph"][
                    "cluster_size_decay_factor"
                ],
                num_post_pool_gat_layers=wandb.config.models["graph"][
                    "num_post_pool_gat_layers"
                ],
                num_graphs=num_graphs,
                max_num_nodes=max_num_nodes,
                gat_dropout_prob=wandb.config.models["graph"]["gat_dropout_prob"],
                last_layer_dropout_prob=wandb.config.models["graph"][
                    "last_layer_dropout_prob"
                ],
                norm=wandb.config.models["graph"]["norm"],
                activation=wandb.config.models["graph"]["activation"],
                gat_skip_connection=wandb.config.models["graph"]["gat_skip_connection"],
                pruned_max_average_node_degree=wandb.config.models["graph"][
                    "pruned_max_average_node_degree"
                ],
                weight_init=wandb.config.models["graph"]["weight_init"],
            ),
            "top": Mlp(
                in_channels=wandb.config.models["graph"]["diffpool_out_channels"],
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
    wandb.watch(model["main"], log="gradients", log_freq=100, log_graph=False)
    task = RegressionTask(
        model=model,
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        batch_size=wandb.config.data_module["batch_size"],
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        link_pred_loss_weight=wandb.config.regression_task["link_pred_loss_weight"],
        entropy_loss_weight=wandb.config.regression_task["entropy_loss_weight"],
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
    if wandb.config.trainer["devices"] != "auto":
        devices = wandb.config.trainer["devices"]
    elif wandb.config.trainer["devices"] == "auto":
        devices = num_devices
    elif num_devices == 0:
        # if there are no GPUs available, use 1 CPU
        devices = 1

    # In your main function:
    # profiler = CustomAdvancedProfiler(
    #     dirpath="profiles", filename="advanced_profiler_output.prof"
    # )
    torch.set_float32_matmul_precision("medium")

    # accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator=wandb.config.trainer["accelerator"],
        devices=devices,  # FLAG
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],  # [checkpoint_callback, accumulator],
        # profiler=profiler,  #
        # log_every_n_steps=2,
        # callbacks=[checkpoint_callback, TriggerWandbSyncLightningCallback()],
    )

    # Start the training
    trainer.fit(model=task, datamodule=data_module)


if __name__ == "__main__":
    main()
