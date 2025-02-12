# torchcell/viz/visual_regression
# [[torchcell.viz.visual_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/viz/visual_regression
# Test file: tests/torchcell/viz/test_visual_regression.py


import hydra
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import os
from omegaconf import DictConfig, OmegaConf
import uuid
import json
import hashlib
import socket
import io
from PIL import Image
from typing import Any
from torchcell.losses.multi_dim_nan_tolerant import (
    WeightedDistLoss,
    SupCR,
    WeightedMSELoss,
)
from tqdm import tqdm
from torch import nn
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
import os.path as osp
from scipy import stats
from dotenv import load_dotenv
import logging
import wandb
from torchcell.timestamp import timestamp

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MPLSTYLE_PATH = os.getenv("MPLSTYLE_PATH")
DATA_ROOT = os.getenv("DATA_ROOT")
plt.style.use(MPLSTYLE_PATH)
log = logging.getLogger(__name__)


class Visualization:
    def __init__(self, base_dir: str, max_points: int = 1000):
        self.base_dir = base_dir
        self.artifact_dir = osp.join(base_dir, "figures")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.artifact = None
        self.max_points = max_points

    def save_and_log_figure(
        self, fig: plt.Figure, name: str, timestamp_str: str
    ) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        wandb_image = wandb.Image(Image.open(buf))
        wandb.log({name: wandb_image}, commit=True)
        buf.close()

    def init_wandb_artifact(self, name: str, artifact_type: str = "figures"):
        self.artifact = wandb.Artifact(name, type=artifact_type)

    def get_base_title(self, loss_name: str, num_epochs: int) -> str:
        return f"Training Results\nLoss: {loss_name}\nEpochs: {num_epochs}"

    def plot_correlations(
        self,
        predictions: torch.Tensor,
        true_values: torch.Tensor,
        dim: int,
        loss_name: str,
        num_epochs: int,
        timestamp_str: str,
    ):
        predictions_np = predictions.detach().cpu().numpy()
        true_values_np = true_values.detach().cpu().numpy()

        mask = ~np.isnan(true_values_np[:, dim])
        y = true_values_np[mask, dim]
        x = predictions_np[mask, dim]

        if len(x) > self.max_points:
            idx = np.random.choice(len(x), size=self.max_points, replace=False)
            x = x[idx]
            y = y[idx]

        pearson, _ = stats.pearsonr(x, y)
        spearman, _ = stats.spearmanr(x, y)
        mse = np.mean((y - x) ** 2)

        fig = plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.6, color="#2971A0", s=20)
        plt.xlabel(f"Predicted Target {dim}")
        plt.ylabel(f"True Target {dim}")

        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(
            f"{base_title}\nTarget {dim}\n"
            f"MSE={mse:.3e}, n={len(x)}\nPearson={pearson:.3f}, Spearman={spearman:.3f}"
        )

        min_val = min(plt.xlim()[0], plt.ylim()[0])
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=45, ha="right")
        plt.tight_layout()

        self.save_and_log_figure(fig, f"correlations_target_{dim}", timestamp_str)
        plt.close()

    def plot_distribution(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        loss_name: str,
        dim: int,
        num_epochs: int,
        timestamp_str: str,
    ):
        """Plot distribution matching for a target dimension."""
        true_values_np = true_values.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()

        # Clean data
        mask = ~np.isnan(true_values_np[:, dim])
        y_true = true_values_np[mask, dim]
        y_pred = predictions_np[mask, dim]

        # Calculate metrics
        wasserstein_distance = stats.wasserstein_distance(y_true, y_pred)

        # Calculate JS divergence using histograms
        bins = min(100, int(np.sqrt(len(y_true))))
        true_hist, edges = np.histogram(y_true, bins=bins, density=True)
        pred_hist, _ = np.histogram(y_pred, bins=edges, density=True)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        true_hist = (true_hist + epsilon) / (true_hist.sum() + epsilon * len(true_hist))
        pred_hist = (pred_hist + epsilon) / (pred_hist.sum() + epsilon * len(pred_hist))

        m = 0.5 * (true_hist + pred_hist)
        js_div = 0.5 * (stats.entropy(true_hist, m) + stats.entropy(pred_hist, m))

        # Create plot
        fig = plt.figure(figsize=(10, 6))

        # Plot histograms
        sns.histplot(
            y_true,
            color="blue",
            label="True",
            kde=True,
            stat="density",
            alpha=0.6,
            bins=bins,
        )
        sns.histplot(
            y_pred,
            color="red",
            label="Predicted",
            kde=True,
            stat="density",
            alpha=0.6,
            bins=bins,
        )

        # Add metrics text
        plt.text(
            0.98,
            0.98,
            f"Wasserstein: {wasserstein_distance:.4f}\nJS Div: {js_div:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        plt.legend()
        plt.xlabel(f"Target {dim}")
        plt.ylabel("Density")

        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nDistribution Matching Target {dim}")

        plt.tight_layout()
        self.save_and_log_figure(fig, f"distribution_target_{dim}", timestamp_str)
        plt.close()

    def plot_umap(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        loss_name: str,
        dim: int,
        num_epochs: int,
        timestamp_str: str,
    ):
        # Convert to numpy and subsample if needed
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()  # Remove the dim indexing here

        if features_np.shape[0] > self.max_points:
            idx = np.random.choice(
                features_np.shape[0], size=self.max_points, replace=False
            )
            features_np = features_np[idx]
            labels_np = labels_np[idx]

        # Add debug print to understand shapes
        print(f"Features shape: {features_np.shape}")
        print(f"Labels shape: {labels_np.shape}")

        # Fit UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        embedding = reducer.fit_transform(features_np)

        # Create plot
        fig = plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels_np,  # Use full labels tensor
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.colorbar(scatter, label=f"Target {dim}")
        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nUMAP Projection Target {dim}")
        plt.grid(True, alpha=0.3)

        self.save_and_log_figure(fig, f"umap_target_{dim}", timestamp_str)
        plt.close()

    def plot_loss_curves(
        self, losses: dict, num_epochs: int, timestamp_str: str
    ) -> None:
        """
        Expects `losses` to be a dict with keys like "train" and "val",
        each mapping to another dict where keys are loss component names and
        values are lists of epoch-level numbers.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        epochs = np.arange(1, num_epochs + 1)
        for phase in losses.keys():
            for key, values in losses[phase].items():
                if values:  # only plot if there are values
                    ax.plot(epochs[: len(values)], values, label=f"{phase}_{key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.set_title(f"Loss Curves (Epochs: {num_epochs})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save_and_log_figure(fig, "loss_curves_generic", timestamp_str)
        plt.close()

    def log_artifact(self):
        if self.artifact is not None:
            wandb.log_artifact(self.artifact)


if __name__ == "__main__":
    pass
