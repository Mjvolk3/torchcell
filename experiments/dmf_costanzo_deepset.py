# experiments/dmf_costanzo_deepset.py
# [[experiments.dmf_costanzo_deepset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/dmf_costanzo_deepset.py
# Test file: experiments/test_dmf_costanzo_deepset.py

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os.path as osp
from torchcell.datamodules import CellDataModule
from torchcell.datasets import NucleotideTransformerDataset, FungalUtrTransformerDataset
from torchcell.datasets.scerevisiae import (
    DMFCostanzo2016Dataset,
    DMFCostanzo2016SmallDataset,
    DMFCostanzo2016LargeDataset,
)
import torch
from torchcell.models import DeepSet
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import RegressionTask
from torchcell.datasets import CellDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from dotenv import load_dotenv
import logging
from omegaconf import DictConfig, OmegaConf

import wandb
import hydra

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# @hydra.main(
#     version_base=None,
#     config_path=osp.join(os.getcwd(), "conf"),
#     config_name="config",
# )

@hydra.main(version_base=None, config_path="conf", config_name="dmf_costanzo_deepset")
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        tags=wandb_cfg["wandb"]["tags"],
    )

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))

    # Sequence transformers
    nt_dataset = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
        genome=genome,
        transformer_model_name="nt_window_5979",
    )

    # Experiments
    experiments = DMFCostanzo2016LargeDataset(
        preprocess="low_dmf_std",
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_0"),
    )

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell")),
        genome=genome,
        seq_embeddings=nt_dataset,
        experiments=experiments,
    )

    # Instantiate your data module and model
    data_module = CellDataModule(dataset=cell_dataset, batch_size=16, num_workers=0)
    input_dim = cell_dataset.num_features
    model = RegressionTask(
        model=DeepSet(
            input_dim,
            wandb.model["instance_layers"],
            wandb.model["set_layers"],
            output_activation="relu",
        )
    )

    checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoints")
    # Initialize the Trainer with the WandbLogger
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)
    trainer = pl.Trainer(
        logger=wandb_logger, max_epochs=10, callbacks=[checkpoint_callback]
    )

    # Start the training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
