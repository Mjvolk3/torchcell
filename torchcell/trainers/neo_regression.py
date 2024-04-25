# torchcell/trainers/regression.py
# [[torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/regression.py
# Test file: torchcell/trainers/test_regression.py

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

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


class ListMLEMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.list_mle_loss = ListMLELoss()  # Renamed for clarity
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loss = self.list_mle_loss(preds, target)  # Corrected method call
        self.sum_loss += loss.detach() * target.size(0)
        self.total += target.size(0)

    def compute(self):
        return self.sum_loss / self.total


class MSEListMLELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MSEListMLELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.list_mle_loss = ListMLELoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        list_mle = self.list_mle_loss(y_pred, y_true)
        combined_loss = mse + (self.alpha * list_mle)
        return combined_loss


class RegressionTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        model: nn.Module,
        target: str,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss: str = "mse",
        batch_size: int = None,
        train_epoch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.boxplot_every_n_epochs = boxplot_every_n_epochs
        # target for training
        self.target = target

        # Lightning settings, doing this for WT embedding
        self.automatic_optimization = False

        self.model = model

        # clip grad norm
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm

        # loss
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "list_mle":
            self.loss = ListMLELoss()
        elif loss == "mse+list_mle":
            self.loss = MSEListMLELoss(alpha=alpha)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid."
                "Currently, supports 'mse', 'mae' loss."
            )

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # batch_size
        self.batch_size = batch_size

        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
                "ListMLE": ListMLEMetric(),
            },
            prefix="train/",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val/")
        self.test_metrics = self.train_metrics.clone(prefix="test/")

        # Separate attributes for Pearson and Spearman correlation coefficients
        self.pearson_corr = PearsonCorrCoef()
        self.spearman_corr = SpearmanCorrCoef()

        # TODO not sure if these are needed for corr ?
        self.true_values = []
        self.predictions = []

        # wandb model artifact logging
        self.last_logged_best_step = None

    def setup(self, stage=None):
        self.model = self.model.to(self.device)

    def forward(self, x, batch):
        x_nodes, x_set = self.model["main"](x, batch)
        y_hat = self.model["top"](x_set)
        return y_hat

    def on_train_start(self):
        parameter_size = sum(p.numel() for p in self.parameters())
        parameter_size_float = float(parameter_size)
        self.log("model/parameters_size", parameter_size_float, on_epoch=True)

    def training_step(self, batch, batch_idx):
        # Extract the batch vector
        print()
        x, y, batch_vector = (
            batch["gene"].x,
            batch["gene"].label_value,
            batch["gene"].batch,
        )
        # Pass the batch vector to the forward method
        y_hat = self(x, batch_vector)

        opt = self.optimizers()
        opt.zero_grad()

        loss = self.loss(y_hat, y)

        self.manual_backward(loss)  # error on this line
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.clip_grad_norm_max_norm
            )
        opt.step()
        opt.zero_grad()
        # logging
        batch_size = batch_vector[-1].item() + 1
        self.log("train/loss", loss, batch_size=batch_size, sync_dist=True)
        self.train_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "train/pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Extract the batch vector
        print()
        x, y, batch_vector = (
            batch["gene"].x,
            batch["gene"].label_value,
            batch["gene"].batch,
        )
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("val/loss", loss, batch_size=batch_size, sync_dist=True)
        self.val_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "val/pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

    def compute_prediction_stats(self, true_values, predictions, stage="val"):
        # Define the bin edges
        bin_edges = torch.tensor(
            [-float("inf"), 0] + [i * 0.1 for i in range(1, 13)] + [float("inf")]
        )

        # Prepare a table with columns for the range, mean, and standard deviation
        wandb_table = wandb.Table(columns=["Range", "Mean", "StdDev"])

        # Calculate mean and std for each bin and add to the table
        for i in range(len(bin_edges) - 1):
            bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if bin_mask.any():
                bin_predictions = predictions[bin_mask]
                mean_val = bin_predictions.mean().item()
                std_val = bin_predictions.std().item()
                if i == len(bin_edges) - 2:
                    range_str = "1.2 - inf"
                else:
                    range_str = (
                        f"{bin_edges[i].item():.1f} - {bin_edges[i+1].item():.1f}"
                    )
                wandb_table.add_data(range_str, mean_val, std_val)

        # Log the table to wandb
        wandb.log({f"{stage}/Prediction_Stats_{self.current_epoch}": wandb_table})

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.boxplot_every_n_epochs != 0
        ):
            return

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="val")

        if self.target == "fitness":
            fig = fitness.box_plot(true_values, predictions)
        elif self.target == "genetic_interaction_score":
            fig = genetic_interaction_score.box_plot(true_values, predictions)
        wandb.log({"binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
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
        # Extract the batch vector
        print()
        x, y, batch_vector = (
            batch["gene"].x,
            batch["gene"].label_value,
            batch["gene"].batch,
        )
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("test/loss", loss, batch_size=batch_size, sync_dist=True)
        self.test_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "test_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test_spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="test")

        if self.target == "fitness":
            fig = fitness.box_plot(true_values, predictions)
        elif self.target == "genetic_interaction_score":
            fig = genetic_interaction_score.box_plot(true_values, predictions)
        wandb.log({"test_binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
        # Clear the stored values
        self.true_values = []
        self.predictions = []

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
