# torchcell/trainers/regression.py
# [[torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/regression.py
# Test file: torchcell/trainers/test_regression.py

import math

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

import torchcell
style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class SimpleLinearRegressionTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        model: nn.Module,
        target: str,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss: str = "mse",
        batch_size: int = None,
        train_epoch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # target for training
        self.target = target

        # legacy from dmf_costanzo_deepeset.py
        self.automatic_optimization = False

        self.model = model
        self.is_wt_init = False
        self.wt_nodes_hat, self.wt_set_hat, self.wt_global_hat = None, None, None

        # clip grad norm
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_norm_max_norm = clip_grad_norm_max_norm

        # loss
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "weighted_mse":
            mean_value = kwargs.get("fitness_mean_value")
            penalty = kwargs.get("penalty", 1.0)
            self.loss = WeightedMSELoss(mean_value=mean_value, penalty=penalty)
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid."
                "Currently, supports 'mse' or 'mae' loss."
            )
        self.loss_node = nn.MSELoss()

        # train epoch size for wt frequency
        self.train_epoch_size = train_epoch_size

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
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

        # Separate attributes for Pearson and Spearman correlation coefficients
        self.pearson_corr = PearsonCorrCoef()
        self.spearman_corr = SpearmanCorrCoef()

        # Used in end for whisker plot
        self.boxplot_every_n_epochs = boxplot_every_n_epochs
        self.true_values = []
        self.predictions = []

        # wandb model artifact logging
        self.last_logged_best_step = None

    def setup(self, stage=None):
        self.model = self.model.to(self.device)

    def forward(self, x, batch):
        y_hat = self.model(x, batch).squeeze()
        return y_hat

    def on_train_start(self):
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log("model/parameters_size", parameter_size)

    def training_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = (batch.x_pert, batch[self.target], batch.x_pert_batch)
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
        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)
        self.train_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "train_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_spearman",
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
        x, y, batch_vector = (batch.x_pert, batch[self.target], batch.x_pert_batch)
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        self.val_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log(
            "val_pearson",
            self.pearson_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_spearman",
            self.spearman_corr(y_hat, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.true_values.append(y.detach())
        self.predictions.append(y_hat.detach())

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

        if self.target == "fitness":
            fig = fitness.box_plot(true_values, predictions)
        elif self.target == "genetic_interaction_score":
            fig = genetic_interaction_score.box_plot(true_values, predictions)

        wandb.log({"binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

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
        x, y, batch_vector = (batch.x_pert, batch[self.target], batch.x_pert_batch)
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
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

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
