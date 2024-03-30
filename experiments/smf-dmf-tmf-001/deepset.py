# experiments/smf-dmf-tmf/deepset
# [[experiments.smf-dmf-tmf.deepset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf/deepset
# Test file: experiments/smf-dmf-tmf/test_deepset.py

import datetime
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
import torch.distributed as dist
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torchcell.graph import SCerevisiaeGraph

import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    FungalUpDownTransformerDataset,
    NucleotideTransformerDataset,
    OneHotGeneDataset,
    ProtT5Dataset,
)
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.models import DeepSet, Mlp
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.data import Neo4jCellDataset, ExperimentDeduplicator

# from torchcell.trainers import RegressionTask
from torchcell.trainers import RegressionTask
from torchcell.utils import format_scientific_notation


log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

os.environ['WANDB_SERVICE_WAIT'] = 300

@hydra.main(version_base=None, config_path="conf", config_name="deepset")
def main(cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{slurm_job_id}_{hashed_cfg}"
    wandb.init(
        mode="online",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
    )

    # Initialize the WandbLogger
    wandb_logger = WandbLogger(project=wandb_cfg["wandb"]["project"], log_model=True)

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
    genome.drop_chrmt()
    genome.drop_empty_go()

    # Node embedding datasets
    node_embeddings = {}
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
    if "one_hot_gene" in wandb.config.cell_dataset["node_embeddings"]:
        node_embeddings["one_hot_gene"] = OneHotGeneDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/one_hot_gene_embedding"),
            genome=genome,
        )
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
    # Experiments
    with open((osp.join(osp.dirname(__file__), "query.cql")), "r") as f:
        query = f.read()

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

    deduplicator = ExperimentDeduplicator()

    # Convert max_size to float, then format it in concise scientific notation
    max_size_str = format_scientific_notation(
        float(wandb.config.cell_dataset["max_size"])
    )
    dataset_root = osp.join(
        DATA_ROOT, f"data/torchcell/experiments/smf-dmf-tmf_{max_size_str}"
    )

    cell_dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        genome=genome,
        graphs=graphs,
        node_embeddings=node_embeddings,
        deduplicator=deduplicator,
        max_size=int(wandb.config.cell_dataset["max_size"]),
    )

    # Instantiate your data module and model
    data_module = CellDataModule(
        dataset=cell_dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        batch_size=8,
        random_seed=42,
        num_workers=4,
        pin_memory=False,
    )

    # Anytime data is accessed lmdb must be closed.
    input_dim = cell_dataset.num_features["gene"]
    cell_dataset.close_lmdb()

    model = ModuleDict(
        {
            "main": DeepSet(
                input_dim=input_dim,
                node_layers=wandb.config.models["graph"]["node_layers"],
                set_layers=wandb.config.models["graph"]["set_layers"],
                norm=wandb.config.models["graph"]["norm"],
                activation=wandb.config.models["graph"]["activation"],
                skip_node=wandb.config.models["graph"]["skip_node"],
                skip_set=wandb.config.models["graph"]["skip_set"],
            ),
            "top": Mlp(
                input_dim=wandb.config.models["graph"]["set_layers"][-1],
                layer_dims=wandb.config.models["pred_head"]["layer_dims"],
            ),
        }
    )
    task = RegressionTask(
        model=model,
        target=wandb.config.regression_task["target"],
        learning_rate=wandb.config.regression_task["learning_rate"],
        weight_decay=wandb.config.regression_task["weight_decay"],
        loss=wandb.config.regression_task["loss"],
        batch_size=wandb.config.data_module["batch_size"],
        train_epoch_size=data_module.train_epoch_size,
        clip_grad_norm=wandb.config.regression_task["clip_grad_norm"],
        clip_grad_norm_max_norm=wandb.config.regression_task["clip_grad_norm_max_norm"],
        boxplot_every_n_epochs=wandb.config.regression_task["boxplot_every_n_epochs"],
        # **kwargs,
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{group}",
        save_top_k=1,
        monitor="val/loss",
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

    trainer = L.Trainer(
        strategy=wandb.config.trainer["strategy"],
        devices=devices,
        logger=wandb_logger,
        max_epochs=wandb.config.trainer["max_epochs"],
        callbacks=[checkpoint_callback],
    )

    # Start the training
    trainer.fit(task, data_module)


if __name__ == "__main__":
    main()
