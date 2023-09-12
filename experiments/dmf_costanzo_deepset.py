# experiments/dmf_costanzo_deepset.py
# [[experiments.dmf_costanzo_deepset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/dmf_costanzo_deepset.py
# Test file: experiments/test_dmf_costanzo_deepset.py

from fileinput import filename
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os.path as osp
from torchcell.datamodules import CellDataModule
from torchcell.datasets import NucleotideTransformerDataset, FungalUtrTransformerDataset
from torchcell.datasets.scerevisiae import DMFCostanzo2016Dataset
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
from datetime import datetime
log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

pl.seed_everything(42)

# Set the precision for matrix multiplication
torch.set_float32_matmul_precision("medium")  # or 'high'

# Set the seed


@hydra.main(version_base=None, config_path="conf", config_name="dmf_costanzo_deepset")
def main(cfg: DictConfig) -> None:
    # wandb.init(
    #     mode=wandb_cfg["wandb"]["mode"],
    #     project=wandb_cfg["wandb"]["project"],
    #     config=wandb_cfg,
    #     tags=wandb_cfg["wandb"]["tags"],
    # )
    # Initialize the WandbLogger
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'NA')
    current_date = datetime.now().strftime("%Y.%m.%d")
    # Grouping for ddp
    if torch.cuda.device_count() > 1:
        group = f"ddp_{slurm_job_id}_{current_date}"
    else:
        group = f"single_device_{slurm_job_id}_{current_date}"
    # If we don't init here we get an error on all other but 0 gpus.
    wandb.init(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        tags=wandb_cfg["wandb"]["tags"],
        group=group,
    )
    wandb_logger = WandbLogger(
        mode=wandb_cfg["wandb"]["mode"],
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        tags=wandb_cfg["wandb"]["tags"],
        group=group,
        log_model=True,
    )
    wandb.log({"slurm_job_id": slurm_job_id})
    
    # Your database initialization code here
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Sequence transformers
    nt_dataset = NucleotideTransformerDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/nucleotide_transformer_embed"),
        genome=genome,
        transformer_model_name="nt_window_5979",
    )

    # Experiments
    experiments = DMFCostanzo2016Dataset(
        preprocess="low_dmf_std",
        root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
    )

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(osp.join(DATA_ROOT, "data/scerevisiae/cell_1e5")),
        genome_gene_set=genome.gene_set,
        seq_embeddings=nt_dataset,
        experiments=experiments,
    )

    wandb.log({"dataset/size": len(cell_dataset)})
    # Instantiate your data module and model
    data_module = CellDataModule(
        dataset=cell_dataset,
        batch_size=wandb_cfg["data_module"]["batch_size"],
        num_workers=wandb_cfg["data_module"]["num_workers"],
        pin_memory=wandb_cfg["data_module"]["pin_memory"],
    )
    input_dim = cell_dataset.num_features
    model = RegressionTask(
        model=DeepSet(
            input_dim,
            wandb_cfg["model"]["instance_layers"],
            wandb_cfg["model"]["set_layers"],
            output_activation="relu",
        )
    )

    wandb.watch(model, log="gradients", log_freq=1000, idx=None, log_graph=True)

    checkpoint_callback = ModelCheckpoint(dirpath="models/checkpoints")
    # Initialize the Trainer with the WandbLogger
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    log.info(f"devices: {devices}")

    num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', 1)
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=wandb_cfg["trainer"]["max_epochs"],
        callbacks=[checkpoint_callback],
        num_nodes=num_nodes,
        devices=devices,
        strategy="ddp",
        accelerator="gpu",
        profiler="simple",
    )
    # Start the training
    trainer.fit(model, data_module)
    wandb.finish()


if __name__ == "__main__":
    main()
