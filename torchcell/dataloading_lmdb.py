import logging
import os
import os.path as osp

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    CellDataset,
    FungalUtrTransformerDataset,
    NucleotideTransformerDataset,
)
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.models import DeepSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import RegressionTask


def main():
    log = logging.getLogger(__name__)
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()
    genome_gene_set = genome.gene_set

    # Sequence transformers
    nt_dataset = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
        genome=genome,
        transformer_model_name="nt_window_5979",
    )

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess="low_dmf_std",
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016"),
    )
    print(len(experiments))

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell")),
        genome_gene_set=genome_gene_set,
        seq_embeddings=nt_dataset,
        experiments=experiments,
    )
    print(len(cell_dataset))
    data_loader = DataLoader(cell_dataset, batch_size=32, shuffle=True, num_workers=2)
    print(cell_dataset)
    print(cell_dataset[0])

    for batch in tqdm(data_loader):
        # print(f"Accessing LMDB from process {os.getpid()}")
        pass


if __name__ == "__main__":
    main()
