import io
import os
import os.path as osp
import wandb
import matplotlib.pyplot as plt
import numpy as np
import umap
import seaborn as sns
from PIL import Image
from scipy import stats
import torch
from typing import Optional


class Visualization:
    def __init__(self, base_dir: str, max_points: int = 1000) -> None:
        self.base_dir = base_dir
        self.artifact_dir = osp.join(base_dir, "figures")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.artifact = None
        self.max_points = max_points

    def save_and_log_figure(
        self, fig: plt.Figure, name: str, timestamp_str: Optional[str]
    ) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        wandb_image = wandb.Image(Image.open(buf))
        wandb.log({name: wandb_image}, commit=True)
        buf.close()

    def init_wandb_artifact(self, name: str, artifact_type: str = "figures") -> None:
        self.artifact = wandb.Artifact(name, type=artifact_type)

    def get_base_title(
        self, name: str, num_epochs: int, title_type: str = "loss"
    ) -> str:
        if title_type == "latent":
            return f"Training Results\nLatent: {name}\nEpochs: {num_epochs}"
        else:
            return f"Training Results\nLoss: {name}\nEpochs: {num_epochs}"

    def plot_correlations(
        self,
        predictions: torch.Tensor,
        true_values: torch.Tensor,
        dim: int,
        loss_name: str,
        num_epochs: int,
        timestamp_str: Optional[str],
        stage: str = "",
    ) -> None:
        predictions_np = predictions.detach().cpu().numpy()
        true_values_np = true_values.detach().cpu().numpy()
        mask = ~np.isnan(true_values_np[:, dim])
        x = predictions_np[mask, dim]
        y = true_values_np[mask, dim]
        if len(x) < 2:
            print(
                f"Not enough valid points for correlation plot for target {dim}. Skipping."
            )
            return
        if len(x) > self.max_points:
            idx = np.random.choice(len(x), size=self.max_points, replace=False)
            x = x[idx]
            y = y[idx]
        try:
            pearson, _ = stats.pearsonr(x, y)
            spearman, _ = stats.spearmanr(x, y)
        except Exception as e:
            print(f"Correlation calculation failed for target {dim}: {e}")
            return
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
        key = (
            f"{stage}/correlations_target_{dim}"
            if stage
            else f"correlations_target_{dim}"
        )
        self.save_and_log_figure(fig, key, timestamp_str)
        plt.close()

    def plot_distribution(
        self,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        loss_name: str,
        dim: int,
        num_epochs: int,
        timestamp_str: Optional[str],
        stage: str = "",
    ) -> None:
        true_np = true_values.detach().cpu().numpy()
        pred_np = predictions.detach().cpu().numpy()
        mask = ~np.isnan(true_np[:, dim])
        y_true = true_np[mask, dim]
        y_pred = pred_np[mask, dim]
        if len(y_true) < 2:
            print(
                f"Not enough valid points for distribution plot for target {dim}. Skipping."
            )
            return
        wasserstein_distance = stats.wasserstein_distance(y_true, y_pred)
        bins = min(100, int(np.sqrt(len(y_true))))
        true_hist, edges = np.histogram(y_true, bins=bins, density=True)
        pred_hist, _ = np.histogram(y_pred, bins=edges, density=True)
        epsilon = 1e-10
        true_hist = (true_hist + epsilon) / (true_hist.sum() + epsilon * len(true_hist))
        pred_hist = (pred_hist + epsilon) / (pred_hist.sum() + epsilon * len(pred_hist))
        m = 0.5 * (true_hist + pred_hist)
        js_div = 0.5 * (stats.entropy(true_hist, m) + stats.entropy(pred_hist, m))

        # Create figure with explicit axes control
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Plot histograms
        sns.histplot(
            y_true,
            color="blue",
            label="True",
            kde=True,
            stat="density",
            alpha=0.6,
            bins=bins,
            ax=ax,
        )
        sns.histplot(
            y_pred,
            color="red",
            label="Predicted",
            kde=True,
            stat="density",
            alpha=0.6,
            bins=bins,
            ax=ax,
        )

        # Position metrics text in top-left corner
        ax.text(
            0.02,
            0.98,
            f"Wasserstein: {wasserstein_distance:.4f}\nJS Div: {js_div:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
            zorder=100,
        )

        # Position legend in top-right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))

        ax.set_xlabel(f"Target {dim}")
        ax.set_ylabel("Density")
        base_title = self.get_base_title(loss_name, num_epochs)
        ax.set_title(f"{base_title}\nDistribution Matching Target {dim}")

        plt.tight_layout()
        key = (
            f"{stage}/distribution_target_{dim}"
            if stage
            else f"distribution_target_{dim}"
        )
        self.save_and_log_figure(fig, key, timestamp_str)
        plt.close()

    def plot_umap(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        source: str,
        dim: int,
        num_epochs: int,
        timestamp_str: Optional[str],  # still accepted, but not used for key
        stage: str = "",
        title_type: str = "loss",
    ) -> None:
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        valid_mask = ~np.isnan(features_np).any(axis=1)
        features_np = features_np[valid_mask]
        labels_np = labels_np[valid_mask]
        if features_np.shape[0] < 2:
            print(
                f"Not enough valid features for UMAP plot for target {dim}. Skipping."
            )
            return
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
        plt.colorbar(scatter, label=f"Target {dim}")
        base_title = self.get_base_title(source, num_epochs, title_type)
        plt.title(f"{base_title}\nUMAP Projection ({source}) for Target {dim}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Remove timestamp from key so that plots remain consistent across runs.
        key = (
            f"{stage}/umap_{source}_target_{dim}"
            if stage
            else f"umap_{source}_target_{dim}"
        )
        self.save_and_log_figure(fig, key, None)
        plt.close()

    def log_sample_metrics(
        self, predictions: torch.Tensor, true_values: torch.Tensor, stage: str = ""
    ) -> None:
        predictions_np = predictions.detach().cpu().numpy()
        true_values_np = true_values.detach().cpu().numpy()
        num_targets = true_values_np.shape[1]
        prefix = stage  # use the provided stage directly as key prefix
        for dim in range(num_targets):
            mask = ~np.isnan(true_values_np[:, dim])
            if np.sum(mask) < 2:
                print(f"Not enough valid samples for sample metrics for target {dim}.")
                continue
            pred_dim = predictions_np[mask, dim]
            true_dim = true_values_np[mask, dim]
            mse = np.mean((pred_dim - true_dim) ** 2)
            mae = np.mean(np.abs(pred_dim - true_dim))
            pearson, _ = stats.pearsonr(pred_dim, true_dim)
            spearman, _ = stats.spearmanr(pred_dim, true_dim)
            wasserstein = stats.wasserstein_distance(true_dim, pred_dim)
            bins = min(100, int(np.sqrt(len(true_dim))))
            true_hist, edges = np.histogram(true_dim, bins=bins, density=True)
            pred_hist, _ = np.histogram(pred_dim, bins=edges, density=True)
            epsilon = 1e-10
            true_hist = (true_hist + epsilon) / (
                true_hist.sum() + epsilon * len(true_hist)
            )
            pred_hist = (pred_hist + epsilon) / (
                pred_hist.sum() + epsilon * len(pred_hist)
            )
            m = 0.5 * (true_hist + pred_hist)
            js_div = 0.5 * (stats.entropy(true_hist, m) + stats.entropy(pred_hist, m))
            wandb.log(
                {
                    f"{prefix}/MSE_target_{dim}": mse,
                    f"{prefix}/MAE_target_{dim}": mae,
                    f"{prefix}/Pearson_target_{dim}": pearson,
                    f"{prefix}/Spearman_target_{dim}": spearman,
                    f"{prefix}/Wasserstein_target_{dim}": wasserstein,
                    f"{prefix}/JS_div_target_{dim}": js_div,
                }
            )

    def visualize_model_outputs(
        self,
        predictions: torch.Tensor,
        true_values: torch.Tensor,
        latents: dict[str, torch.Tensor],
        loss_name: str,
        num_epochs: int,
        timestamp_str: Optional[str],
        stage: str = "",
    ) -> None:
        # Plot correlations and distributions for each target dimension.
        for dim in [0, 1]:
            self.plot_correlations(
                predictions,
                true_values,
                dim,
                loss_name,
                num_epochs,
                timestamp_str,
                stage=stage,
            )
            self.plot_distribution(
                true_values,
                predictions,
                loss_name,
                dim,
                num_epochs,
                timestamp_str,
                stage=stage,
            )
        # Plot UMAP for each latent representation without appending loss info.
        for latent_key, latent in latents.items():
            for dim in [0, 1]:
                self.plot_umap(
                    latent,
                    true_values[:, dim],
                    latent_key,
                    dim,
                    num_epochs,
                    timestamp_str,
                    stage=stage,
                    title_type="latent",
                )
        # Log additional sample metrics.
        self.log_sample_metrics(predictions, true_values, stage=stage)
