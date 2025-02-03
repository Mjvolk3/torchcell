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


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        z = self.conv2(x, edge_index)
        u = global_add_pool(z, batch)
        y_hat = self.fc(u)
        return y_hat, u


class CellLoss(nn.Module):
    def __init__(
        self, lambda_1: float = 1.0, lambda_2: float = 1.0, device: str = "cuda"
    ):
        super().__init__()
        self.mse_loss = WeightedMSELoss(weights=torch.ones(19, device=device))
        self.div_loss = WeightedDistLoss(weights=torch.ones(19, device=device))
        self.con_loss = SupCR()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss using graph-level predictions and embeddings.

        Args:
            predictions: Graph-level predictions [batch_size, num_targets]
            targets: Graph-level targets [batch_size, num_targets]
            graph_embeddings: Graph-level embeddings [batch_size, embedding_dim]
        """
        mse_loss, _ = self.mse_loss(predictions, targets)
        div_loss, _ = self.div_loss(predictions, targets)
        con_loss = self.con_loss(graph_embeddings, targets).mean()

        weighted_mse = mse_loss
        weighted_div = self.lambda_1 * div_loss
        weighted_con = self.lambda_2 * con_loss
        total = weighted_mse + weighted_div + weighted_con

        return {
            "total_loss": total,
            "norm_mse": weighted_mse / total,
            "norm_div": weighted_div / total,
            "norm_con": weighted_con / total,
            "raw_mse": weighted_mse,
            "raw_div": weighted_div,
            "raw_con": weighted_con,
        }


class MetricTracker:
    def log_epoch_metrics(
        self,
        epoch: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss_dict: dict,
        num_targets: int,
        phase: str,  # Add phase parameter
    ):
        """Log all metrics for each target dimension"""
        metrics = {}

        # Log loss components
        for key, value in loss_dict.items():
            metrics[f"{phase}_{key}"] = value.item()

        # Calculate metrics for each target dimension
        for dim in range(num_targets):
            mask = ~torch.isnan(labels[:, dim])
            y_true = labels[mask, dim].cpu().numpy()
            y_pred = predictions[mask, dim].cpu().numpy()

            # Statistical metrics
            pearson, _ = stats.pearsonr(y_true, y_pred)
            spearman, _ = stats.spearmanr(y_true, y_pred)
            wasserstein = stats.wasserstein_distance(y_true, y_pred)

            metrics.update(
                {
                    f"{phase}_target_{dim}/mse": np.mean((y_true - y_pred) ** 2),
                    f"{phase}_target_{dim}/mae": np.mean(np.abs(y_true - y_pred)),
                    f"{phase}_target_{dim}/pearson": pearson,
                    f"{phase}_target_{dim}/spearman": spearman,
                    f"{phase}_target_{dim}/wasserstein": wasserstein,
                }
            )

        wandb.log(metrics, step=epoch, commit=False)  # Set commit=False for training


class Visualization:
    def __init__(self, base_dir: str):
        """Initialize visualization directories for wandb artifacts.

        Args:
            base_dir: Base directory for wandb artifacts
        """
        self.base_dir = base_dir
        self.artifact_dir = osp.join(base_dir, "figures")

        # Create directories
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.artifact = None

    def save_and_log_figure(
        self, fig: plt.Figure, name: str, timestamp_str: str
    ) -> str:
        """Save figure and log to wandb."""
        # Convert the matplotlib figure to a PNG for wandb
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)

        # Create wandb Image from the buffer
        wandb_image = wandb.Image(Image.open(buf))

        # Log to wandb directly
        wandb.log({name: wandb_image}, commit=True)

        # Clean up
        buf.close()

    def init_wandb_artifact(self, name: str, artifact_type: str = "figures"):
        """Initialize a new wandb artifact for storing figures"""
        self.artifact = wandb.Artifact(name, type=artifact_type)

    def get_base_title(self, loss_name: str, num_epochs: int) -> str:
        """Generate base title for plots."""
        return f"Training Results\nLoss: {loss_name}\n Epochs: {num_epochs}"

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

        mask = ~np.isnan(true_values_np)
        y = true_values_np[mask]
        x = predictions_np[mask]

        pearson, _ = stats.pearsonr(x, y)
        spearman, _ = stats.spearmanr(x, y)
        mse = np.mean((y - x) ** 2)

        fig = plt.figure(figsize=(7, 6))
        plt.scatter(x, y, alpha=0.6, color="#2971A0", s=20)
        plt.xlabel(f"Predicted Target {dim}")
        plt.ylabel(f"True Target {dim}")

        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(
            f"{base_title}\nTarget {dim}\nMSE={mse:.3e}, n={len(x)}\n"
            f"Pearson={pearson:.3f}, Spearman={spearman:.3f}"
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
        true_values_np = true_values.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()

        mask = ~np.isnan(true_values_np)
        true_values_clean = true_values_np[mask]
        predictions_clean = predictions_np[mask]

        wasserstein_distance = stats.wasserstein_distance(
            true_values_clean, predictions_clean
        )

        bins = min(100, int(np.sqrt(len(true_values_clean))))
        true_hist, bin_edges = np.histogram(true_values_clean, bins=bins, density=True)
        pred_hist, _ = np.histogram(predictions_clean, bins=bin_edges, density=True)

        epsilon = 1e-10
        true_hist = true_hist + epsilon
        pred_hist = pred_hist + epsilon

        true_hist = true_hist / true_hist.sum()
        pred_hist = pred_hist / pred_hist.sum()

        m = 0.5 * (true_hist + pred_hist)
        js_divergence = 0.5 * (
            stats.entropy(true_hist, m) + stats.entropy(pred_hist, m)
        )

        fig = plt.figure(figsize=(10, 6))  # Wider figure to accommodate legend

        # Create histograms with adjusted parameters
        sns.histplot(
            true_values_clean,
            color="blue",
            label="True",
            kde=True,
            alpha=0.6,
            stat="density",
            common_norm=False,
        )
        sns.histplot(
            predictions_clean,
            color="red",
            label="Predicted",
            kde=True,
            alpha=0.6,
            stat="density",
            common_norm=False,
        )

        # Position metrics text and improve readability
        plt.text(
            0.98,
            0.98,
            f"Wasserstein: {wasserstein_distance:.4f}\nJS Div: {js_divergence:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        # Improve legend placement and formatting
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Rotate and format x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Set title and adjust layout
        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nDistribution Matching Target {dim}")

        # Ensure all elements are visible
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
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        embedding = reducer.fit_transform(features.detach().cpu().numpy())

        fig = plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels.cpu().numpy(),
            cmap="coolwarm",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Target Labels")
        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nUMAP Projection Target {dim}")

        self.save_and_log_figure(fig, f"umap_target_{dim}", timestamp_str)
        plt.close()

    def plot_loss_curves(
        self, losses: dict, loss_name: str, num_epochs: int, timestamp_str: str
    ):
        """Plot training and validation losses."""
        fig = plt.figure(figsize=(12, 12))

        # Get the number of steps per epoch
        steps_per_epoch = len(losses["train"]["total_loss"]) // num_epochs
        epoch_steps = np.linspace(1, num_epochs, num_epochs * steps_per_epoch)

        # Plot normalized losses (top subplot)
        plt.subplot(2, 1, 1)
        for phase in ["train", "val"]:
            for key in ["norm_mse", "norm_div", "norm_con"]:
                if len(losses[phase][key]) > 0:
                    plt.plot(
                        epoch_steps[: len(losses[phase][key])],
                        losses[phase][key],
                        label=f"{phase}_{key}",
                        linestyle="-" if phase == "train" else "--",
                        linewidth=1.0,
                        alpha=0.5,
                    )
        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nNormalized Loss Components")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Proportion")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Set integer ticks
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Dynamic y-axis limits for raw losses
        min_val = float("inf")
        max_val = float("-inf")
        for phase in ["train", "val"]:
            for key in ["total_loss", "raw_mse", "raw_div", "raw_con"]:
                if len(losses[phase][key]) > 0:
                    min_val = min(min_val, min(losses[phase][key]))
                    max_val = max(max_val, max(losses[phase][key]))

        y_min = 10 ** (np.floor(np.log10(min_val)) - 1)
        y_max = 10 ** (np.ceil(np.log10(max_val)) + 1)

        # Plot raw losses with dynamic log scale
        plt.subplot(2, 1, 2)
        for phase in ["train", "val"]:
            plt.plot(
                epoch_steps[: len(losses[phase]["total_loss"])],
                losses[phase]["total_loss"],
                label=f"{phase}_total",
                linewidth=1.0,
                alpha=0.5,
                linestyle="-" if phase == "train" else "--",
            )
            for key in ["raw_mse", "raw_div", "raw_con"]:
                if len(losses[phase][key]) > 0:
                    plt.plot(
                        epoch_steps[: len(losses[phase][key])],
                        losses[phase][key],
                        label=f"{phase}_{key}",
                        alpha=0.5,
                        linewidth=1.0,
                        linestyle="-" if phase == "train" else "--",
                    )
        plt.title(f"{base_title}\nRaw Loss Components")
        plt.xlabel("Epoch")
        plt.ylabel("Log Loss Value")
        plt.yscale("log")
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Set integer ticks
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        self.save_and_log_figure(fig, "loss_curves", timestamp_str)
        plt.close()

    def log_artifact(self):
        """Log the artifact to wandb if it exists"""
        if self.artifact is not None:
            wandb.log_artifact(self.artifact)


class LossTracker:
    def __init__(self):
        # Initialize separate dicts for train and val losses
        self.train_losses = {
            "total_loss": [],
            "norm_mse": [],
            "norm_div": [],
            "norm_con": [],
            "raw_mse": [],
            "raw_div": [],
            "raw_con": [],
        }
        self.val_losses = {
            "total_loss": [],
            "norm_mse": [],
            "norm_div": [],
            "norm_con": [],
            "raw_mse": [],
            "raw_div": [],
            "raw_con": [],
        }

    def update_train(self, loss_dict: dict):
        for key, value in loss_dict.items():
            self.train_losses[key].append(value.item())

    def update_val(self, loss_dict: dict):
        for key, value in loss_dict.items():
            self.val_losses[key].append(value.item())

    def get_losses(self):
        return {"train": self.train_losses, "val": self.val_losses}


def train_and_evaluate(
    model, loss_fn, train_loader, val_loader, optimizer, num_epochs=3, device="cuda"
):
    loss_tracker = LossTracker()
    metric_tracker = MetricTracker()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_preds, train_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, u = model(batch)
            loss_dict = loss_fn(pred, batch.y, u)
            loss_dict["total_loss"].backward()
            optimizer.step()
            loss_tracker.update_train(loss_dict)
            train_preds.append(pred.detach())
            train_labels.append(batch.y)

        # Log training metrics
        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        metric_tracker.log_epoch_metrics(
            epoch=epoch,
            predictions=train_preds,
            labels=train_labels,
            loss_dict=loss_dict,
            num_targets=train_labels.shape[1],
            phase="train",
        )

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                batch = batch.to(device)
                pred, u = model(batch)
                loss_dict = loss_fn(pred, batch.y, u)
                loss_tracker.update_val(loss_dict)
                val_preds.append(pred)
                val_labels.append(batch.y)

        # Log validation metrics
        val_preds = torch.cat(val_preds, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        metric_tracker.log_epoch_metrics(
            epoch=epoch,
            predictions=val_preds,
            labels=val_labels,
            loss_dict=loss_dict,
            num_targets=val_labels.shape[1],
            phase="val",
        )

        # Commit metrics at end of epoch
        wandb.log({}, commit=True)

    # Final evaluation for returning predictions
    model.eval()
    all_preds, all_labels, all_embeddings = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final evaluation"):
            batch = batch.to(device)
            pred, u = model(batch)
            all_preds.append(pred.cpu())
            all_labels.append(batch.y.cpu())
            all_embeddings.append(u.cpu())

    return (
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_embeddings, dim=0),
        loss_tracker.get_losses(),
        num_epochs,
    )

    # Final evaluation for returning predictions
    model.eval()
    all_preds, all_labels, all_embeddings = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final evaluation"):
            batch = batch.to(device)
            pred, u = model(batch)
            all_preds.append(pred.cpu())
            all_labels.append(batch.y.cpu())
            all_embeddings.append(u.cpu())

    return (
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_embeddings, dim=0),
        loss_tracker.get_losses(),
        num_epochs,
    )


@hydra.main(
    version_base=None,
    config_path=osp.join(osp.dirname(__file__), "../conf"),
    config_name="qm9_loss_study",
)
def main(cfg: DictConfig) -> None:
    print("Starting QM9 Loss Study 🧪")
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Convert hydra config to dict for wandb
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Setup unique run identification
    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"

    # Create hash of config for grouping
    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_slurm_job_id}_{hashed_cfg}"

    # Setup wandb experiment directory
    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        mode="online",
        project=wandb_cfg["wandb"]["project"],
        config=wandb_cfg,
        group=group,
        dir=experiment_dir,
        job_type="train",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset setup
    data_root = osp.join(DATA_ROOT, "data/QM9")
    if wandb.config.training["dataset_size"] is not None:
        dataset = QM9(root=data_root)[: wandb.config.training["dataset_size"]]
    else:
        dataset = QM9(root=data_root)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=wandb.config.training["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=wandb.config.training["batch_size"], shuffle=False
    )

    # Model setup
    model = GCN(
        dataset.num_features,
        wandb.config.training["hidden_channels"],
        dataset[0].y.shape[-1],
    ).to(device)

    loss_fn = CellLoss(
        lambda_1=wandb.config.loss_sweep["lambda_1"],
        lambda_2=wandb.config.loss_sweep["lambda_2"],
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=wandb.config.training["learning_rate"]
    )

    # Train and evaluate
    val_preds, val_labels, val_z, losses, num_epochs = train_and_evaluate(
        model,
        loss_fn,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=wandb.config.training["num_epochs"],
        device=device,
    )

    # Initialize visualization
    vis = Visualization(experiment_dir)
    timestamp_str = timestamp()
    vis.init_wandb_artifact(
        name=f"figures_{timestamp_str}", artifact_type="training_figures"
    )

    # Generate and save visualizations
    loss_name = f"MSE+(λ1={wandb.config['loss_sweep']['lambda_1']:.0e})DistLoss+(λ2={wandb.config['loss_sweep']['lambda_2']:.0e})SupCR+(wd={wandb.config['training']['weight_decay']:.0e})L2"

    # Plot loss curves
    vis.plot_loss_curves(losses, loss_name, num_epochs, timestamp_str)

    # Plot correlations, distributions and UMAP for each target dimension
    num_targets = val_labels.shape[1]
    for dim in range(num_targets):
        vis.plot_correlations(
            val_preds, val_labels, dim, loss_name, num_epochs, timestamp_str
        )
        vis.plot_distribution(
            val_labels, val_preds, loss_name, dim, num_epochs, timestamp_str
        )
        vis.plot_umap(
            val_z, val_labels[:, dim], loss_name, dim, num_epochs, timestamp_str
        )

    # Log the artifact
    vis.log_artifact()
    wandb.finish()


if __name__ == "__main__":
    main()
