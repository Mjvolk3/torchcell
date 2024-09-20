# experiments/dmf_costanzo_deepset.py
# [[experiments.dmf_costanzo_deepset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/dmf_costanzo_deepset.py
# Test file: experiments/test_dmf_costanzo_deepset.py

import datetime
import hashlib
import json
import logging
import os
import os.path as osp
import uuid

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    CellDataset,
    FungalUpDownTransformerDataset,
    NucleotideTransformerDataset,
)
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.models import DeepSetTransformer, Mlp
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import RegressionTaskDeepSetTransformer

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="dmf_costanzo_deep_set_transformer",
)
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

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Sequence transformers
    fungal_down_dataset = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
        genome=genome,
        transformer_model_name="species_downstream",
    )
    fungal_up_dataset = FungalUpDownTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
        genome=genome,
        transformer_model_name="species_upstream",
    )
    seq_embeddings = fungal_down_dataset + fungal_up_dataset

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e3"),
    )

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell_1e3")),
        genome=genome,
        seq_embeddings=seq_embeddings,
        experiments=experiments,
    )

    # Instantiate your data module and model
    data_module = CellDataModule(
        dataset=cell_dataset,
        batch_size=wandb.config.data_module["batch_size"],
        num_workers=wandb.config.data_module["num_workers"],
    )
    input_dim = cell_dataset.num_features
    models = {
        "deep_set_transformer": DeepSetTransformer(
            input_dim=input_dim,
            node_layers=wandb.config.models["graph"]["node_layers"],
            set_layers=wandb.config.models["graph"]["set_layers"],
            norm=wandb.config.models["graph"]["norm"],
            activation=wandb.config.models["graph"]["activation"],
            skip_node=wandb.config.models["graph"]["skip_node"],
            skip_set=wandb.config.models["graph"]["skip_set"],
            num_heads=wandb.config.models["graph"]["num_heads"],
            is_concat_attention=wandb.config.models["graph"]["is_concat_attention"],
        ),
        "mlp_ref_set": Mlp(
            input_dim=wandb.config.models["graph"]["set_layers"][-1],
            layer_dims=wandb.config.models["mlp_refset"]["layer_dims"],
        ),
    }
    # could also have mlp_ref_nodes

    model = RegressionTaskDeepSetTransformer(
        models=models,
        wt=cell_dataset.wt,
        wt_train_ratio=wandb.config.regression_task["wt_train_ratio"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        loss=wandb.config.regression_task["loss"],
    )

    checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoints")
    # Initialize the Trainer with the WandbLogger
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)

    num_devices = torch.cuda.device_count()
    if num_devices == 0:  # if there are no GPUs available, use 1 CPU
        devices = 1
    else:
        devices = num_devices

    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        devices=devices,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
    )

    # Start the training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
