# torchcell/trainers/fit_int_gat_diffpool_regression
# [[torchcell.trainers.fit_int_gat_diffpool_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/fit_int_gat_diffpool_regression
# Test file: tests/torchcell/trainers/test_fit_int_gat_diffpool_regression.py


import math
import os.path as osp
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from tqdm import tqdm
import wandb
from torchcell.losses import WeightedMSELoss
from torchcell.viz import fitness, genetic_interaction_score
from torchcell.losses.list_mle import ListMLELoss
import torchcell
from torchmetrics import Metric
import torch
from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


class NaNTolerantCorrelation(Metric):
    def __init__(self, base_metric, default_value=torch.nan):
        super().__init__()
        self.base_metric = base_metric()
        self.default_value = default_value

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Create a mask for non-NaN values
        mask = ~torch.isnan(preds) & ~torch.isnan(target)

        # If there are any non-NaN values, update the metric
        if torch.any(mask):
            self.base_metric.update(preds[mask], target[mask])

    def compute(self):
        try:
            # Compute the metric if there were valid samples
            result = self.base_metric.compute()
        except ValueError:
            # If no valid samples, return torch.nan
            result = self.default_value
        return result


class NaNTolerantPearsonCorrCoef(NaNTolerantCorrelation):
    def __init__(self):
        super().__init__(PearsonCorrCoef)


class NaNTolerantSpearmanCorrCoef(NaNTolerantCorrelation):
    def __init__(self):
        super().__init__(SpearmanCorrCoef)


class MultiDimNaNTolerantMSELoss(nn.Module):
    def __init__(self):
        super(MultiDimNaNTolerantMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure y_pred and y_true have the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "Predictions and targets must have the same shape"

        # Create a mask for non-NaN values
        mask = ~torch.isnan(y_true)

        # Count the number of non-NaN elements per dimension
        n_valid = mask.sum(dim=0)

        # Replace NaN values with 0 in y_true (they won't contribute to the loss due to masking)
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))

        # Calculate squared error
        squared_error = torch.pow(y_pred - y_true_masked, 2)

        # Sum the errors for each dimension, considering only non-NaN elements
        dim_losses = (squared_error * mask.float()).sum(dim=0)

        # Calculate mean loss for each dimension, avoiding division by zero
        dim_means = torch.where(
            n_valid > 0, dim_losses / n_valid.float(), torch.zeros_like(dim_losses)
        )

        return dim_means


class CombinedMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(CombinedMSELoss, self).__init__()
        self.multi_dim_mse = MultiDimNaNTolerantMSELoss()
        self.weights = weights if weights is not None else torch.ones(2)

    def forward(self, y_pred, y_true):
        dim_losses = self.multi_dim_mse(y_pred, y_true)
        weighted_loss = (dim_losses * self.weights).sum() / self.weights.sum()
        return weighted_loss, dim_losses


class RegressionTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
        link_pred_loss_weight: float = 1.0,
        entropy_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.boxplot_every_n_epochs = boxplot_every_n_epochs
        # target for training

        # Lightning settings, doing this for WT embedding
        self.automatic_optimization = False

        self.model = model

        # clip grad norm
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm

        # loss
        self.combined_mse_loss = CombinedMSELoss(weights=torch.ones(2))
        self.link_pred_loss_weight = link_pred_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # batch_size
        self.batch_size = batch_size

        metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(),
                "MAE": MeanAbsoluteError(),
                "PearsonR": NaNTolerantPearsonCorrCoef(),
                "SpearmanR": NaNTolerantSpearmanCorrCoef(),
            }
        )

        self.train_metrics = nn.ModuleDict(
            {
                "fitness": metrics.clone(prefix="train/fitness/"),
                "gene_interaction": metrics.clone(prefix="train/gene_interaction/"),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "fitness": metrics.clone(prefix="val/fitness/"),
                "gene_interaction": metrics.clone(prefix="val/gene_interaction/"),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "fitness": metrics.clone(prefix="test/fitness/"),
                "gene_interaction": metrics.clone(prefix="test/gene_interaction/"),
            }
        )
        # TODO not sure if these are needed for corr ?
        self.true_values = []
        self.predictions = []

        # wandb model artifact logging
        self.last_logged_best_step = None

    def setup(self, stage=None):
        self.model = self.model.to(self.device)
        self.combined_mse_loss.weights = self.combined_mse_loss.weights.to(self.device)

    def forward(self, x, edge_indices, batch):
        (
            out,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        ) = self.model["main"](x, edge_indices, batch)
        y_hat = self.model["top"](out)
        return (
            y_hat,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        )

    def on_train_start(self):
        parameter_size = sum(p.numel() for p in self.parameters())
        parameter_size_float = float(parameter_size)
        self.log("model/parameters_size", parameter_size_float, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x = batch["gene"].x
        edge_indices = [
            batch["gene", "physical_interaction", "gene"].edge_index,
            batch["gene", "regulatory_interaction", "gene"].edge_index,
        ]
        batch_vector = batch["gene"].batch
        y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

        (
            y_hat,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        ) = self(x, edge_indices, batch_vector)

        opt = self.optimizers()
        opt.zero_grad()

        # Combined MSE loss
        mse_loss, dim_losses = self.combined_mse_loss(y_hat, y)

        # Weighted link prediction and entropy losses
        link_pred_loss = sum(link_pred_losses) * self.link_pred_loss_weight
        entropy_loss = sum(entropy_losses) * self.entropy_loss_weight

        # Total loss
        loss = mse_loss + link_pred_loss + entropy_loss

        self.manual_backward(loss)
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.clip_grad_norm_max_norm
            )
        opt.step()
        opt.zero_grad()

        # Logging
        batch_size = batch_vector[-1].item() + 1
        self.log("train/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("train/mse_loss", mse_loss, batch_size=batch_size, sync_dist=True)
        self.log(
            "train/fitness_loss", dim_losses[0], batch_size=batch_size, sync_dist=True
        )
        self.log(
            "train/gene_interaction_loss",
            dim_losses[1],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/link_pred_loss",
            link_pred_loss,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/entropy_loss", entropy_loss, batch_size=batch_size, sync_dist=True
        )

        # Update metrics
        self.train_metrics["fitness"](y_hat[:, 0], y[:, 0])
        self.train_metrics["gene_interaction"](y_hat[:, 1], y[:, 1])

        return loss

    def on_train_epoch_end(self):
        # Compute and log metrics for each metric type
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"train/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()  # Reset metrics after logging

    def validation_step(self, batch, batch_idx):
        x = batch["gene"].x
        edge_indices = [
            batch["gene", "physical_interaction", "gene"].edge_index,
            batch["gene", "regulatory_interaction", "gene"].edge_index,
        ]
        batch_vector = batch["gene"].batch
        y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

        (
            y_hat,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        ) = self(x, edge_indices, batch_vector)

        # Combined MSE loss
        mse_loss, dim_losses = self.combined_mse_loss(y_hat, y)

        # Weighted link prediction and entropy losses
        link_pred_loss = sum(link_pred_losses) * self.link_pred_loss_weight
        entropy_loss = sum(entropy_losses) * self.entropy_loss_weight

        # Total loss
        loss = mse_loss + link_pred_loss + entropy_loss

        batch_size = batch_vector[-1].item() + 1

        # Log the losses
        self.log("val/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("val/mse_loss", mse_loss, batch_size=batch_size, sync_dist=True)
        self.log(
            "val/fitness_loss", dim_losses[0], batch_size=batch_size, sync_dist=True
        )
        self.log(
            "val/gene_interaction_loss",
            dim_losses[1],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/link_pred_loss", link_pred_loss, batch_size=batch_size, sync_dist=True
        )
        self.log(
            "val/entropy_loss", entropy_loss, batch_size=batch_size, sync_dist=True
        )

        # Update metrics
        self.val_metrics["fitness"](y_hat[:, 0], y[:, 0])
        self.val_metrics["gene_interaction"](y_hat[:, 1], y[:, 1])

        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

    def compute_prediction_stats(self, true_values, predictions, stage="val"):
        # Define the bin edges for each dimension
        bin_edges = {
            "fitness": torch.tensor(
                [-float("inf"), 0] + [i * 0.1 for i in range(1, 13)] + [float("inf")]
            ),
            "gene_interaction": torch.tensor(
                [-float("inf"), -0.2, -0.1, 0, 0.1, 0.2, float("inf")]
            ),
        }

        for dim, dim_name in enumerate(["fitness", "gene_interaction"]):
            # Prepare a table with columns for the range, mean, and standard deviation
            wandb_table = wandb.Table(columns=["Range", "Mean", "StdDev"])

            dim_true = true_values[:, dim]
            dim_pred = predictions[:, dim]

            # Calculate mean and std for each bin and add to the table
            for i in range(len(bin_edges[dim_name]) - 1):
                bin_mask = (dim_pred >= bin_edges[dim_name][i]) & (
                    dim_pred < bin_edges[dim_name][i + 1]
                )
                if bin_mask.any():
                    bin_predictions = dim_pred[bin_mask]
                    mean_val = bin_predictions.mean().item()
                    std_val = bin_predictions.std().item()
                    if i == len(bin_edges[dim_name]) - 2:
                        range_str = f"{bin_edges[dim_name][i].item():.2f} - inf"
                    else:
                        range_str = f"{bin_edges[dim_name][i].item():.2f} - {bin_edges[dim_name][i+1].item():.2f}"
                    wandb_table.add_data(range_str, mean_val, std_val)

            # Log the table to wandb
            wandb.log(
                {
                    f"{stage}/{dim_name}_Prediction_Stats_{self.current_epoch}": wandb_table
                }
            )

    def on_validation_epoch_end(self):
        # Compute and log metrics for each metric type
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"val/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()  # Reset metrics after logging

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.boxplot_every_n_epochs != 0
        ):
            return

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="val")

        # Create and log box plots for both fitness and gene interaction
        fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
        wandb.log({"fitness_binned_values_box_plot": wandb.Image(fig_fitness)})
        plt.close(fig_fitness)

        fig_gi = genetic_interaction_score.box_plot(
            true_values[:, 1], predictions[:, 1]
        )
        wandb.log({"gene_interaction_binned_values_box_plot": wandb.Image(fig_gi)})
        plt.close(fig_gi)

        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

        # Logging model artifact
        current_global_step = self.global_step
        if (
            self.trainer.checkpoint_callback.best_model_path
            and current_global_step != self.last_logged_best_step
        ):
            # Save model as a W&B artifact
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model on validation epoch end step - {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(self.trainer.checkpoint_callback.best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = (
                current_global_step  # update the last logged step
            )

    def test_step(self, batch, batch_idx):
        x = batch["gene"].x
        edge_indices = [
            batch["gene", "physical_interaction", "gene"].edge_index,
            batch["gene", "regulatory_interaction", "gene"].edge_index,
        ]
        batch_vector = batch["gene"].batch
        y = torch.stack([batch["gene"].fitness, batch["gene"].gene_interaction], dim=1)

        (
            y_hat,
            attention_weights,
            cluster_assignments,
            link_pred_losses,
            entropy_losses,
        ) = self(x, edge_indices, batch_vector)

        # Combined MSE loss
        mse_loss, dim_losses = self.combined_mse_loss(y_hat, y)

        # Weighted link prediction and entropy losses
        link_pred_loss = sum(link_pred_losses) * self.link_pred_loss_weight
        entropy_loss = sum(entropy_losses) * self.entropy_loss_weight

        # Total loss
        loss = mse_loss + link_pred_loss + entropy_loss

        batch_size = batch_vector[-1].item() + 1

        # Log the losses
        self.log("test/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("test/mse_loss", mse_loss, batch_size=batch_size, sync_dist=True)
        self.log(
            "test/fitness_loss", dim_losses[0], batch_size=batch_size, sync_dist=True
        )
        self.log(
            "test/gene_interaction_loss",
            dim_losses[1],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test/link_pred_loss", link_pred_loss, batch_size=batch_size, sync_dist=True
        )
        self.log(
            "test/entropy_loss", entropy_loss, batch_size=batch_size, sync_dist=True
        )

        # Update metrics
        self.test_metrics["fitness"](y_hat[:, 0], y[:, 0])
        self.test_metrics["gene_interaction"](y_hat[:, 1], y[:, 1])

        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="test")

        # Create and log box plots for both fitness and gene interaction
        fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
        wandb.log({"test_fitness_binned_values_box_plot": wandb.Image(fig_fitness)})
        plt.close(fig_fitness)

        fig_gi = genetic_interaction_score.box_plot(
            true_values[:, 1], predictions[:, 1]
        )
        wandb.log({"test_gene_interaction_binned_values_box_plot": wandb.Image(fig_gi)})
        plt.close(fig_gi)

        # Clear the stored values
        self.true_values = []
        self.predictions = []

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
