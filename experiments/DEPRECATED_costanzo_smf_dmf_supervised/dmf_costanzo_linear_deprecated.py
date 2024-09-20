# experiments/costanzo_smf_dmf_supervised/dmf_costanzo_linear.py
# [[experiments.costanzo_smf_dmf_supervised.dmf_costanzo_linear]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/costanzo_smf_dmf_supervised/dmf_costanzo_linear.py
# Test file: experiments/costanzo_smf_dmf_supervised/test_dmf_costanzo_linear.py

import datetime
import hashlib
import json
import logging
import os
import os.path as osp
import uuid

import hydra
import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from tqdm import tqdm

import wandb
from torchcell.datamodules import CellDataModule
from torchcell.datasets import (
    CellDataset,
    FungalUpDownTransformerDataset,
    NucleotideTransformerDataset,
    OneHotGeneDataset,
)
from torchcell.datasets.scerevisiae import DmfCostanzo2016Dataset
from torchcell.models import DeepSet, Mlp
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.trainers import RegressionTask
from torchcell.viz import fitness, genetic_interaction_score

log = logging.getLogger(__name__)
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")


def calculate_correlations(y_val, y_pred):
    pearson_coef, _ = pearsonr(y_val, y_pred)
    spearman_coef, _ = spearmanr(y_val, y_pred)
    return pearson_coef, spearman_coef


def log_to_wandb(metrics, prefix=""):
    logs = {}
    for key, value in metrics.items():
        logs[f"{prefix}{key}"] = value
    wandb.log(logs)


def train_and_evaluate_linear_model(X_train, y_train, X_val, y_val, cv):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["r2", "neg_mean_squared_error"],
        return_train_score=True,
    )

    avg_r_squared = scores["test_r2"].mean()
    dev_r_squared = scores["test_r2"].std()

    avg_mse = -scores[
        "test_neg_mean_squared_error"
    ].mean()  # Negate because it's negative MSE
    dev_mse = scores["test_neg_mean_squared_error"].std()

    return model, mse, avg_r_squared, dev_r_squared, avg_mse, dev_mse, y_pred


def aggregate_data(x_pert, x_pert_batch, method="sum"):
    """
    Aggregates data based on the provided method.

    Parameters:
    - x_pert: Data tensor.
    - x_pert_batch: Batch indices tensor.
    - method (str): The aggregation method - "sum", "concat", or "mean".

    Returns:
    - Tensor: The aggregated data tensor.
    """

    # Determine the size for the aggregated data tensor
    aggregated_size = x_pert_batch[-1].item() + 1
    aggregated_data = []

    if method == "sum" or method == "mean":
        aggregated_tensor = torch.zeros((aggregated_size, x_pert.size(1)))
        for i, data in enumerate(x_pert):
            aggregated_tensor[x_pert_batch[i]] += data
        if method == "mean":
            counts = torch.bincount(x_pert_batch).float()
            aggregated_tensor /= counts.unsqueeze(1)
        aggregated_data.extend(aggregated_tensor)

    elif method == "concat":
        # Concatenate data along dim=1
        temp_data = [torch.zeros((x_pert.size(1),)) for _ in range(aggregated_size)]
        for i, data in enumerate(x_pert):
            temp_data[x_pert_batch[i]] = torch.cat(
                (temp_data[x_pert_batch[i]], data), dim=0
            )
        aggregated_data.extend(temp_data)

    return torch.stack(aggregated_data)


def save_data_to_file(data, filename):
    torch.save(data, filename)


def load_data_from_file(filename):
    return torch.load(filename)


@hydra.main(version_base=None, config_path="conf", config_name="dmf_costanzo_linear")
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

    # Get reference genome
    genome = SCerevisiaeGenome(data_root=osp.join(DATA_ROOT, "data/sgd/genome"))
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

    seq_embeddings = sum(embeddings)

    # Experiments
    experiments = DmfCostanzo2016Dataset(
        preprocess={"duplicate_resolution": "low_dmf_std"},
        root=osp.join(
            DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["experiments"]
        ),
    )

    # Gather into CellDatset
    cell_dataset = CellDataset(
        root=osp.join(
            osp.join(DATA_ROOT, "data/scerevisiae", wandb.config.cell_dataset["name"])
        ),
        genome=genome,
        seq_embeddings=seq_embeddings,
        experiments=experiments,
    )

    data_module = CellDataModule(
        dataset=cell_dataset,
        batch_size=wandb.config.data_module["batch_size"],
        num_workers=wandb.config.data_module["num_workers"],
    )
    data_module.setup()

    # Create a unique identifier for the dataset
    # based on configurations or other dataset details
    data_identifier = hashlib.sha256(
        json.dumps(wandb.config.cell_dataset, sort_keys=True).encode()
    ).hexdigest()
    data_dir = osp.join(DATA_ROOT, "data/scerevisiae/linear/")
    os.makedirs(data_dir, exist_ok=True)
    x_perts_filename = osp.join(
        data_dir, f"x_perts_{wandb.config.cell_dataset['name']}_{data_identifier}.pt"
    )
    fitnesses_filename = osp.join(
        data_dir, f"fitnesses_{wandb.config.cell_dataset['name']}_{data_identifier}.pt"
    )
    genetic_interaction_score_filename = osp.join(
        data_dir,
        f"genetic_interaction_score_{wandb.config.cell_dataset['name']}_{data_identifier}.pt",
    )

    # Check if the aggregated data is already saved
    if os.path.exists(x_perts_filename) and os.path.exists(fitnesses_filename):
        x_perts = load_data_from_file(x_perts_filename).numpy()
        fitnesses = load_data_from_file(fitnesses_filename).numpy()
    else:
        x_perts = []
        fitnesses = []
        genetic_interaction_scores = []
        # Looping through all data loaders
        data_loaders = [
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
        ]
        for data_loader in data_loaders:
            for batch in tqdm(data_loader):
                aggregated_x_pert = aggregate_data(
                    batch.x_pert,
                    batch.x_pert_batch,
                    method=wandb.config.regression_task["aggregation_method"],
                )
                x_perts.extend(aggregated_x_pert)
                fitnesses.extend(batch.fitness)
                genetic_interaction_scores.extend(batch.genetic_interaction_score)

        x_perts = torch.stack(x_perts).numpy()
        fitnesses = torch.stack(fitnesses).numpy()
        genetic_interaction_scores = torch.stack(genetic_interaction_scores).numpy()

        # Save the aggregated data
        save_data_to_file(torch.tensor(x_perts), x_perts_filename)
        save_data_to_file(torch.tensor(fitnesses), fitnesses_filename)
        save_data_to_file(
            torch.tensor(genetic_interaction_scores), genetic_interaction_score_filename
        )

    # Splitting the dataset for fitnesses
    X_train_fit, X_val_fit, y_train_fit, y_val_fit = train_test_split(
        x_perts, fitnesses, test_size=0.2, random_state=42
    )

    # Splitting the dataset for genetic_interaction_scores
    X_train_gi, X_val_gi, y_train_gi, y_val_gi = train_test_split(
        x_perts, genetic_interaction_scores, test_size=0.2, random_state=42
    )

    # Training and evaluating the linear model for fitnesses
    # Fitness
    (
        model_fit,
        mse_val_fit,
        avg_r_squared_fit,
        dev_r_squared_fit,
        avg_mse_fit,
        dev_mse_fit,
        y_pred_fit,
    ) = train_and_evaluate_linear_model(
        X_train_fit,
        y_train_fit,
        X_val_fit,
        y_val_fit,
        cv=wandb.config.regression_task["cv"],
    )
    # Genetic Interaction Scores
    (
        model_gi,
        mse_val_gi,
        avg_r_squared_gi,
        dev_r_squared_gi,
        avg_mse_gi,
        dev_mse_gi,
        y_pred_gi,
    ) = train_and_evaluate_linear_model(
        X_train_gi,
        y_train_gi,
        X_val_gi,
        y_val_gi,
        cv=wandb.config.regression_task["cv"],
    )

    # Calculate correlations
    pearson_fit, spearman_fit = calculate_correlations(y_val_fit, y_pred_fit)
    pearson_gi, spearman_gi = calculate_correlations(y_val_gi, y_pred_gi)

    # Logging to wandb
    log_to_wandb(
        {
            "mean_squared_error_validation": mse_val_fit,
            "average_r_squared": avg_r_squared_fit,
            "pearson_correlation": pearson_fit,
            "spearman_correlation": spearman_fit,
            "avg_mse": avg_mse_fit,
            "dev_r_squared": dev_r_squared_fit,
            "dev_mse": dev_mse_fit,
        },
        prefix="Fitness - ",
    )

    log_to_wandb(
        {
            "mean_squared_error_validation": mse_val_gi,
            "average_r_squared": avg_r_squared_gi,
            "pearson_correlation": pearson_gi,
            "spearman_correlation": spearman_gi,
            "avg_mse": avg_mse_gi,
            "dev_r_squared": dev_r_squared_gi,
            "dev_mse": dev_mse_gi,
        },
        prefix="GI - ",
    )

    # Plotting
    fig_fit = fitness.box_plot(torch.tensor(y_val_fit), torch.tensor(y_pred_fit))
    fig_gi = fitness.box_plot(torch.tensor(y_val_gi), torch.tensor(y_pred_gi))

    # Logging plots to wandb and save locally
    wandb.log({"fitness_box_plot": wandb.Image(fig_fit)})
    wandb.log({"genetic_interaction_box_plot": wandb.Image(fig_gi)})

    # Close the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
