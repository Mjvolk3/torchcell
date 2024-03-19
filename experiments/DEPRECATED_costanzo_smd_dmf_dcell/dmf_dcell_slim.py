# experiments/costanzo_smd_dmf_dcell/dmf_dcell_slim.py
# [[experiments.costanzo_smd_dmf_dcell.dmf_dcell_slim]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/costanzo_smd_dmf_dcell/dmf_dcell_slim.py
# Test file: experiments/costanzo_smd_dmf_dcell/test_dmf_dcell_slim.py
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    PyTorchProfiler,
    SimpleProfiler,
)
import copy
import datetime
import hashlib
import json
import logging
import os
import os.path as osp
import pickle
import re
import shutil
import threading
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from os import environ
from typing import List, Optional, Tuple, Union

import hydra
import lightning as L
import lmdb
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from attrs import define
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Extra, Field, ValidationError, validator
from sklearn import experimental
from torch_geometric.data import Batch, Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.separate import separate
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    from_networkx,
    k_hop_subgraph,
    subgraph,
)
from tqdm import tqdm

import wandb
from torchcell.data import Dataset
from torchcell.datamodels import ModelStrictArbitrary
from torchcell.datamodules import CellDataModule
from torchcell.datasets import DCellDataset, OneHotGeneDataset
from torchcell.datasets.codon_frequency import CodonFrequencyDataset
from torchcell.datasets.embedding import BaseEmbeddingDataset
from torchcell.datasets.fungal_up_down_transformer import FungalUpDownTransformerDataset
from torchcell.datasets.nucleotide_transformer import NucleotideTransformerDataset
from torchcell.datasets.scerevisiae import (
    DmfCostanzo2016Dataset,
    SmfCostanzo2016Dataset,
)
from torchcell.graph import (
    SCerevisiaeGraph,
    filter_by_contained_genes,
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.models import (
    DCell,
    DCellLinear,
    DeepSet,
    Mlp,
    dcell,
    dcell_from_networkx,
)
from torchcell.models.llm import NucleotideModel
from torchcell.models.nucleotide_transformer import NucleotideTransformer
from torchcell.prof import prof, prof_input
from torchcell.sequence import GeneSet, Genome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import DCellRegressionSlimTask

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(version_base=None, config_path="conf", config_name="dmf_dcell_slim")
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        tags=wandb_cfg["wandb"]["tags"],
        group=group,
    )

    torch.set_float32_matmul_precision("medium")

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)

    # Get reference genome
    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=False
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Graph
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    # HACK temporary dataset for genes set
    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(
            DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["experiments"]
        ),
    )

    # TODO Need to include smf dataset
    dcell_gene_set = experiments.gene_set
    #
    G = graph.G_go.copy()
    # Filtering
    G = filter_by_date(G, "2017-07-19")
    G = filter_go_IGI(G)
    G = filter_redundant_terms(G)
    G = filter_by_contained_genes(
        G, n=wandb.config.model["contained_genes"], gene_set=dcell_gene_set
    )

    # replace graph
    graph.G_go = G

    # Instantiate Dcell models
    dcell_model = DCell(
        go_graph=graph.G_go,
        subsystem_output_min=wandb.config.model["subsystem_output_min"],
        subsystem_output_max_mult=wandb.config.model["subsystem_output_max_mult"],
    ).to(device)
    dcell_linear = DCellLinear(dcell_model.subsystems, output_size=1).to(device)
    models = {"dcell": dcell_model, "dcell_linear": dcell_linear}

    # Embeddings datasets
    # TODO in the grand scheme of things,
    # we should have embedding selection function or class.
    # HACK haven't dealt with None embeddings
    one_hot = OneHotGeneDataset(root="data/scerevisiae/gene_one_hot", genome=genome)
    embeddings = one_hot

    # DCell dataset with embeddings
    cell_dataset = DCellDataset(
        root=osp.join(
            osp.join(DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["name"])
        ),
        genome=genome,
        graph=graph,
        embeddings=embeddings,
        experiments=experiments,
    )

    # Instantiate your data module and model
    data_module = CellDataModule(
        dataset=cell_dataset,
        batch_size=wandb.config.data_module["batch_size"],
        num_workers=wandb.config.data_module["num_workers"],
    )

    model = DCellRegressionSlimTask(
        models=models,
        target=wandb.config.regression_task["target"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        alpha=wandb.config.regression_task["alpha"],
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{group}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    num_devices = torch.cuda.device_count()
    if num_devices == 0:  # if there are no GPUs available, use 1 CPU
        devices = 1
    else:
        devices = num_devices

    # profiler
    # profiler = PyTorchProfiler(
    #     dirpath="tb_logs/profiler0",
    #     filename="profiler_output",
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20),
    # )

    # profiler = SimpleProfiler(dirpath="profiles", filename="dcell_simple_prof")

    # Update the Trainer to use the profiler
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        accelerator="auto",
        devices=devices,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
        # profiler=profiler,
    )

    # Start the training
    trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
