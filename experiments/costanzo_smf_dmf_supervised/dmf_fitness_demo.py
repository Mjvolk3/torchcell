# experiments/dmf_fitness_demo.py
# [[experiments.dmf_fitness_demo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/dmf_fitness_demo.py
# Test file: experiments/test_dmf_fitness_demo.py

import os
import os.path as osp

# TODO check how front matter changes when outside src.
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


# GENOME
genome = SCerevisiaeGenome(DATA_ROOT)

# SEQUENCE EMBEDDING DATASETS
nt_dataset = NucleotideTransformerDataset(
    root=DATA_ROOT, genome=genome, transformer_model_name="nt_window_5979"
)
futr3_dataset = FungalUtrTransformerDataset(
    root=DATA_ROOT,
    genome=genome,
    transformer_model_name="fut_species_window_3utr_300_undersize",
)
futr5_dataset = FungalUtrTransformerDataset(
    root=DATA_ROOT,
    genome=genome,
    transformer_model_name="fut_species_window_5utr_1000_undersize",
)
# combine datasets sequence embeddings
seq_embeddings = nt_dataset + fut3_dataset + fut5_dataset

# EXPERIMENTS
experiments = DmfCostanzo2016Dataset(root=DATA_ROOT)

# CELL DATASET
cell_dataset = CellDataset(
    root=DATA_ROOT,
    genome=genome,
    seq_embeddings=seq_embeddings,
    experiments=experiments,
)

# pytorch_lightning - data module and model
data_module = CellDataModule(dataset=cell_dataset, batch_size=2, num_workers=10)
model = RegressionTask(
    model=DeepSet(
        input_dim=cell_dataset.num_features,
        instance_layers=[1024, 256, 128],
        set_layers=[128, 32, 16],
        output_activation="relu",
    )
)

# pytorch_lightning - define trainer
wandb_logger = WandbLogger(project="torchcell", log_model=True)
trainer = L.Trainer(logger=wandb_logger, max_epochs=100)

# Train
trainer.fit(model, data_module)
