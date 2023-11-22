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
import torch.distributed as dist
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
    OneHotGeneDataset,
    ProtT5Dataset,
)
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.models import GraphAttention, GraphConvolution, Mlp
from torchcell.multidigraph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import GraphConvRegressionTask

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


@hydra.main(
    version_base=None, config_path="conf", config_name="dmf_costanzo_graph_convolution"
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
    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=False
    )
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Embeddings datasets
    embeddings = []
    if "fungal_down" in wandb.config.cell_dataset["embeddings"]:
        # Sequence transformers
        embeddings.append(
            FungalUpDownTransformerDataset(
                root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
                genome=genome,
                model_name="species_downstream",
            )
        )

    if "fungal_up" in wandb.config.cell_dataset["embeddings"]:
        embeddings.append(
            FungalUpDownTransformerDataset(
                root=osp.join(DATA_ROOT, "data/scerevisiae/fungal_up_down_embed"),
                genome=genome,
                model_name="species_upstream",
            )
        )
    if "one_hot_gene" in wandb.config.cell_dataset["embeddings"]:
        embeddings.append(
            OneHotGeneDataset(root="data/scerevisiae/gene_one_hot", genome=genome)
        )
    if "prot_T5_all" in wandb.config.cell_dataset["embeddings"]:
        embeddings.append(
            ProtT5Dataset(
                root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
                genome=genome,
                model_name="prot_t5_xl_uniref50_all",
            )
        )
    if "prot_T5_no_dubious" in wandb.config.cell_dataset["embeddings"]:
        embeddings.append(
            ProtT5Dataset(
                root=osp.join(DATA_ROOT, "data/scerevisiae/protT5_embed"),
                genome=genome,
                model_name="prot_t5_xl_uniref50_no_dubious",
            )
        )

    embeddings = sum(embeddings)

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(
            DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["experiments"]
        ),
    )

    # Graph Data
    # graph = SCerevisiaeGraph(
    #     data_root=osp.join(DATA_ROOT, "data/sgd/genes"), genome=genome
    # )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genes"), gene_set=genome.gene_set
    )

    # Gather into CellDataset
    cell_dataset = CellDataset(
        root=osp.join(
            osp.join(DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["name"])
        ),
        genome=genome,
        graph=graph,
        embeddings=embeddings,
        experiments=experiments,
        zero_pert=wandb.config.cell_dataset["zero_pert"],
    )

    # Instantiate your data module and model
    data_module = CellDataModule(
        dataset=cell_dataset,
        batch_size=wandb.config.data_module["batch_size"],
        num_workers=wandb.config.data_module["num_workers"],
    )

    input_dim = cell_dataset.num_features

    if wandb.config.models["cell"]["name"] == "GCN":
        cell_model = GraphConvolution
    elif wandb.config.models["cell"]["name"] == "GAT":
        cell_model = GraphAttention

    models = {
        "cell": cell_model(
            input_dim=input_dim,
            node_layers=wandb.config.models["cell"]["node_layers"],
            hidden_channels=wandb.config.models["cell"]["hidden_channels"],
            num_layers=wandb.config.models["cell"]["num_layers"],
            set_layers=wandb.config.models["cell"]["set_layers"],
            norm=wandb.config.models["cell"]["norm"],
            activation=wandb.config.models["cell"]["activation"],
            skip_node=wandb.config.models["cell"]["skip_node"],
            skip_set=wandb.config.models["cell"]["skip_set"],
            skip_mp=wandb.config.models["cell"]["skip_mp"],
        ),
        "readout": Mlp(
            input_dim=wandb.config.models["cell"]["set_layers"][-1],
            layer_dims=[len(wandb.config.regression_task["target"])],
        ),
    }

    # could also have mlp_ref_nodes
    if wandb.config.regression_task["loss"]:
        fitness_mean_value = experiments.df["Double mutant fitness"].mean()
    else:
        fitness_mean_value = None
    kwargs = {"fitness_mean_value": fitness_mean_value}

    len(experiments)
    model = GraphConvRegressionTask(
        models=models,
        wt=cell_dataset.wt,
        target=wandb.config.regression_task["target"],
        wt_train_per_epoch=wandb.config.regression_task["wt_train_per_epoch"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        loss=wandb.config.regression_task["loss"],
        weighted_mse_penalty=wandb.config.regression_task["weighted_mse_penalty"],
        batch_size=wandb.config.data_module["batch_size"],
        train_wt_node_loss=wandb.config.regression_task["train_wt_node_loss"],
        train_epoch_size=data_module.train_epoch_size,
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        order_penalty=wandb.config.regression_task["order_penalty"],
        lambda_order=wandb.config.regression_task["lambda_order"],
        train_mode=wandb.config.regression_task["train_mode"],
        **kwargs,
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{group}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    # Initialize the Trainer with the WandbLogger
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(device)

    num_devices = torch.cuda.device_count()
    if num_devices == 0:  # if there are no GPUs available, use 1 CPU
        devices = 1
    else:
        devices = num_devices

    # TODO remove num_sanity_val_steps=0
    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        devices=devices,
        accelerator=wandb.config.trainer["accelerator"],
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
    )

    # Start the training
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
