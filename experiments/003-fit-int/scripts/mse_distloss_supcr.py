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
        self, lambda_dist: float = 1.0, lambda_supcr: float = 1.0, device: str = "cuda"
    ):
        super().__init__()
        self.mse_loss = WeightedMSELoss(weights=torch.ones(19, device=device))
        self.div_loss = WeightedDistLoss(weights=torch.ones(19, device=device))
        self.con_loss = SupCR()
        self.lambda_dist = lambda_dist
        self.lambda_supcr = lambda_supcr

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute combined loss using graph-level predictions and embeddings.
        Logs both weighted (with lambda) and unweighted loss components,
        as well as their normalized versions.
        """
        mse_loss, _ = self.mse_loss(predictions, targets)
        div_loss, _ = self.div_loss(predictions, targets)
        con_loss = self.con_loss(graph_embeddings, targets).mean()

        # Weighted components (with lambda multiplication)
        weighted_mse = mse_loss
        weighted_div = self.lambda_dist * div_loss
        weighted_con = self.lambda_supcr * con_loss
        total_weighted = weighted_mse + weighted_div + weighted_con

        # Unweighted components
        total_unweighted = mse_loss + div_loss + con_loss

        # Normalized (weighted) losses
        norm_mse = weighted_mse / total_weighted
        norm_div = weighted_div / total_weighted
        norm_con = weighted_con / total_weighted

        # Normalized unweighted
        norm_unweighted_mse = mse_loss / total_unweighted
        norm_unweighted_div = div_loss / total_unweighted
        norm_unweighted_con = con_loss / total_unweighted

        return {
            "total_loss": total_weighted,
            "norm_mse": norm_mse,
            "norm_div": norm_div,
            "norm_con": norm_con,
            "raw_mse": weighted_mse,
            "raw_div": weighted_div,
            "raw_con": weighted_con,
            "unweighted_mse": mse_loss,
            "unweighted_div": div_loss,
            "unweighted_con": con_loss,
            "norm_unweighted_mse": norm_unweighted_mse,
            "norm_unweighted_div": norm_unweighted_div,
            "norm_unweighted_con": norm_unweighted_con,
        }


class MetricTracker:
    def log_epoch_metrics(
        self,
        epoch: int,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss_dict: dict,
        num_targets: int,
        phase: str,
    ):
        metrics = {}
        # Log all loss components
        for key, value in loss_dict.items():
            metrics[f"{phase}_{key}"] = value.item()

        for dim in range(num_targets):
            mask = ~torch.isnan(labels[:, dim])
            y_true = labels[mask, dim].cpu().numpy()
            y_pred = predictions[mask, dim].cpu().numpy()

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

        wandb.log(metrics, step=epoch, commit=False)


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

        mask = ~np.isnan(true_values_np)
        y = true_values_np[mask]
        x = predictions_np[mask]

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
        true_values_np = true_values.detach().cpu().numpy()
        predictions_np = predictions.detach().cpu().numpy()

        mask = ~np.isnan(true_values_np)
        true_values_clean = true_values_np[mask]
        predictions_clean = predictions_np[mask]

        wasserstein_distance = stats.wasserstein_distance(
            true_values_clean, predictions_clean
        )

        # For better auto-binning, choose 'auto' or Freedman-Diaconis:
        # e.g., bins = int(2 * (np.percentile(true_values_clean, 75) -
        #                      np.percentile(true_values_clean, 25)) /
        #                      (len(true_values_clean) ** (1/3)))
        # Or let numpy do it automatically:
        bins = "auto"

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

        fig = plt.figure(figsize=(10, 6))
        sns.histplot(
            true_values_clean,
            color="blue",
            label="True",
            kde=True,
            alpha=0.6,
            stat="density",
            common_norm=False,
            bins=bin_edges,
        )
        sns.histplot(
            predictions_clean,
            color="red",
            label="Predicted",
            kde=True,
            alpha=0.6,
            stat="density",
            common_norm=False,
            bins=bin_edges,
        )

        plt.text(
            0.98,
            0.98,
            f"Wasserstein: {wasserstein_distance:.4f}\nJS Div: {js_divergence:.4f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")

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
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        if features_np.shape[0] > self.max_points:
            idx = np.random.choice(
                features_np.shape[0], size=self.max_points, replace=False
            )
            features_np = features_np[idx]
            labels_np = labels_np[idx]

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean")
        embedding = reducer.fit_transform(features_np)

        fig = plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[:, 0], embedding[:, 1], c=labels_np, cmap="coolwarm", alpha=0.6
        )
        plt.colorbar(scatter, label="Target Labels")
        base_title = self.get_base_title(loss_name, num_epochs)
        plt.title(f"{base_title}\nUMAP Projection Target {dim}")

        self.save_and_log_figure(fig, f"umap_target_{dim}", timestamp_str)
        plt.close()

    def plot_loss_curves(
        self, losses: dict, loss_name: str, num_epochs: int, timestamp_str: str
    ):
        """
        Now that we aggregate to exactly 1 data point per epoch for train/val,
        we can simply plot them against range(1, num_epochs+1).
        """
        epochs = range(1, num_epochs + 1)

        fig = plt.figure(figsize=(10, 8))

        # Normalized Weighted
        plt.subplot(2, 1, 1)
        for phase in ["train", "val"]:
            for key in ["norm_mse", "norm_div", "norm_con"]:
                plt.plot(
                    epochs,
                    losses[phase][key],
                    label=f"{phase}_{key}",
                    linestyle="-" if phase == "train" else "--",
                    linewidth=1.5,
                    alpha=0.8,
                )
        plt.title(f"{loss_name}\nNormalized Loss Components (Weighted)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Proportion")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Normalized Unweighted
        plt.subplot(2, 1, 2)
        for phase in ["train", "val"]:
            for key in [
                "norm_unweighted_mse",
                "norm_unweighted_div",
                "norm_unweighted_con",
            ]:
                plt.plot(
                    epochs,
                    losses[phase][key],
                    label=f"{phase}_{key}",
                    linestyle="-" if phase == "train" else "--",
                    linewidth=1.5,
                    alpha=0.8,
                )
        plt.title(f"{loss_name}\nNormalized Loss Components (Unweighted)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Proportion")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_and_log_figure(fig, "loss_curves", timestamp_str)
        plt.close()

    def log_artifact(self):
        if self.artifact is not None:
            wandb.log_artifact(self.artifact)


class LossTracker:
    """
    This tracker now stores only per-epoch losses.
    Each epoch, you collect batch-wise losses in a temporary list,
    compute their mean, then store one final number per epoch.
    """

    def __init__(self):
        self.train_losses = {
            "total_loss": [],
            "norm_mse": [],
            "norm_div": [],
            "norm_con": [],
            "raw_mse": [],
            "raw_div": [],
            "raw_con": [],
            "unweighted_mse": [],
            "unweighted_div": [],
            "unweighted_con": [],
            "norm_unweighted_mse": [],
            "norm_unweighted_div": [],
            "norm_unweighted_con": [],
        }
        self.val_losses = {
            "total_loss": [],
            "norm_mse": [],
            "norm_div": [],
            "norm_con": [],
            "raw_mse": [],
            "raw_div": [],
            "raw_con": [],
            "unweighted_mse": [],
            "unweighted_div": [],
            "unweighted_con": [],
            "norm_unweighted_mse": [],
            "norm_unweighted_div": [],
            "norm_unweighted_con": [],
        }

        # Temporary storage for each batch’s losses in the current epoch
        self._train_batch_losses = {key: [] for key in self.train_losses.keys()}
        self._val_batch_losses = {key: [] for key in self.val_losses.keys()}

    def update_train_batch(self, loss_dict: dict):
        for key, value in loss_dict.items():
            self._train_batch_losses[key].append(value.item())

    def update_val_batch(self, loss_dict: dict):
        for key, value in loss_dict.items():
            self._val_batch_losses[key].append(value.item())

    def finalize_epoch(self, phase: str):
        """
        Compute mean of batch losses collected for the epoch, append to the correct list.
        Then reset the temporary storage for next epoch.
        """
        if phase == "train":
            for key, values in self._train_batch_losses.items():
                epoch_mean = np.mean(values) if len(values) > 0 else np.nan
                self.train_losses[key].append(epoch_mean)
            self._train_batch_losses = {
                key: [] for key in self._train_batch_losses.keys()
            }

        elif phase == "val":
            for key, values in self._val_batch_losses.items():
                epoch_mean = np.mean(values) if len(values) > 0 else np.nan
                self.val_losses[key].append(epoch_mean)
            self._val_batch_losses = {key: [] for key in self._val_batch_losses.keys()}

    def get_losses(self):
        return {"train": self.train_losses, "val": self.val_losses}


def train_and_evaluate(
    model, loss_fn, train_loader, val_loader, optimizer, num_epochs=3, device="cuda"
):
    loss_tracker = LossTracker()
    metric_tracker = MetricTracker()

    for epoch in range(num_epochs):
        # --------------------
        #      TRAIN
        # --------------------
        model.train()
        train_preds, train_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, u = model(batch)
            loss_dict = loss_fn(pred, batch.y, u)
            loss_dict["total_loss"].backward()
            optimizer.step()

            # Store for aggregator
            loss_tracker.update_train_batch(loss_dict)
            train_preds.append(pred.detach())
            train_labels.append(batch.y)

        # finalize epoch stats for training
        loss_tracker.finalize_epoch("train")

        train_preds = torch.cat(train_preds, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # Just log final batch's losses for the epoch:
        metric_tracker.log_epoch_metrics(
            epoch=epoch,
            predictions=train_preds,
            labels=train_labels,
            loss_dict=loss_dict,
            num_targets=train_labels.shape[1],
            phase="train",
        )

        # --------------------
        #      VAL
        # --------------------
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                batch = batch.to(device)
                pred, u = model(batch)
                loss_dict = loss_fn(pred, batch.y, u)

                # Store for aggregator
                loss_tracker.update_val_batch(loss_dict)

                val_preds.append(pred)
                val_labels.append(batch.y)

        # finalize epoch stats for val
        loss_tracker.finalize_epoch("val")

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

        # Force wandb step
        wandb.log({}, commit=True)

    # --------------------
    # final evaluation
    # --------------------
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

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    hostname = socket.gethostname()
    hostname_slurm_job_id = f"{hostname}-{slurm_job_id}"

    sorted_cfg = json.dumps(wandb_cfg, sort_keys=True)
    hashed_cfg = hashlib.sha256(sorted_cfg.encode("utf-8")).hexdigest()
    group = f"{hostname_slurm_job_id}_{hashed_cfg}"

    experiment_dir = osp.join(DATA_ROOT, "wandb-experiments", group)
    os.makedirs(experiment_dir, exist_ok=True)

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

    model = GCN(
        dataset.num_features,
        wandb.config.training["hidden_channels"],
        dataset[0].y.shape[-1],
    ).to(device)

    loss_fn = CellLoss(
        lambda_dist=wandb.config.loss_sweep["lambda_dist"],
        lambda_supcr=wandb.config.loss_sweep["lambda_supcr"],
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=wandb.config.training["learning_rate"]
    )

    val_preds, val_labels, val_z, losses, num_epochs = train_and_evaluate(
        model,
        loss_fn,
        train_loader,
        val_loader,
        optimizer,
        num_epochs=wandb.config.training["num_epochs"],
        device=device,
    )

    max_points = wandb.config["plotting"].get("max_points", 1000)
    vis = Visualization(experiment_dir, max_points=max_points)
    timestamp_str = timestamp()
    vis.init_wandb_artifact(
        name=f"figures_{timestamp_str}", artifact_type="training_figures"
    )

    loss_name = (
        f"MSE+(λ1={wandb.config['loss_sweep']['lambda_dist']:.0e})"
        f"DistLoss+(λ2={wandb.config['loss_sweep']['lambda_supcr']:.0e})"
        f"SupCR+(wd={wandb.config['training']['weight_decay']:.0e})L2"
    )

    # Now we have exactly num_epochs points for train/val
    vis.plot_loss_curves(losses, loss_name, num_epochs, timestamp_str)

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

    vis.log_artifact()
    wandb.finish()


if __name__ == "__main__":
    main()
