# experiments/dmf_fitness.py
# [[experiments.dmf_fitness]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/dmf_fitness.py
# Test file: experiments/test_dmf_fitness.py

# TODO check how front matter changes when outside src.
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os.path as osp
from torchcell.datamodules import CellDataModule
from torchcell.datasets import NucleotideTransformerDataset, FungalUtrTransformerDataset
from torchcell.datasets.scerevisiae import (
    DMFCostanzo2016Dataset,
    DMFCostanzo2016SmallDataset,
)
import torch
from torchcell.models import DeepSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import RegressionTask
from torchcell.datasets import CellDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


# Initialize the WandbLogger
wandb_logger = WandbLogger(project="torchcell", log_model=True)
# Build dataset
genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
# Sequence transformers
nt_dataset = NucleotideTransformerDataset(
    root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
    genome=genome,
    transformer_model_name="nt_window_5979",
)
# fut3_dataset = FungalUtrTransformerDataset(
#     root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_utr_embed"),
#     genome=genome,
#     transformer_model_name="fut_species_window_3utr_300_undersize",
# )
# fut5_dataset = FungalUtrTransformerDataset(
#     root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_utr_embed"),
#     genome=genome,
#     transformer_model_name="fut_species_window_5utr_1000_undersize",
# )
# seq_embeddings = nt_dataset + fut3_dataset + fut5_dataset

experiments = DMFCostanzo2016SmallDataset(
    root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016")
)
cell_dataset = CellDataset(
    root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell")),
    genome=genome,
    seq_embeddings=nt_dataset,
    experiments=experiments,
)

# Instantiate your data module and model
data_module = CellDataModule(dataset=cell_dataset, batch_size=2, num_workers=0)
input_dim = cell_dataset.num_features
instance_layers = [1024, 256, 128]
set_layers = [128, 32, 16]
model = RegressionTask(
    model=DeepSet(input_dim, instance_layers, set_layers, output_activation="relu")
)

checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoints")
# Initialize the Trainer with the WandbLogger
device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    logger=wandb_logger, max_epochs=10, callbacks=[checkpoint_callback]
)

# Start the training
trainer.fit(model, data_module)
