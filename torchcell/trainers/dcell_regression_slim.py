# torchcell/trainers/dcell_regression_slim.py
# [[torchcell.trainers.dcell_regression_slim]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/dcell_regression_slim.py
# Test file: torchcell/trainers/test_dcell_regression_slim.py

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

# from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
from tqdm import tqdm

import wandb
from torchcell.losses import DCellLoss, WeightedMSELoss
from torchcell.viz import fitness, genetic_interaction_score

import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class DCellRegressionSlimTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        target: str,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = None,
        alpha: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        # models
        self.models = models

        for key, value in models.items():
            setattr(self, key, value)

        # target for training
        self.target = target

        # Lightning settings, doing this for WT embedding
        self.automatic_optimization = False

        self.x_name = "x"
        self.x_batch_name = "batch"

        self.loss = DCellLoss(alpha)

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
                "Pearson": PearsonCorrCoef(),
                "Spearman": SpearmanCorrCoef(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.train_metrics_root = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
                "Pearson": PearsonCorrCoef(),
                "Spearman": SpearmanCorrCoef(),
            },
            prefix="train_root_",
        )
        self.val_metrics_root = self.train_metrics_root.clone(prefix="val_root_")
        self.test_metrics_root = self.train_metrics_root.clone(prefix="test_root_")

        # Used in end for whisker plot
        self.boxplot_every_n_epochs = boxplot_every_n_epochs

        # wandb model artifact logging
        self.last_logged_best_step = None

    def setup(self, stage=None):
        for model in self.models.values():
            model.to(self.device)

    def forward(self, batch):
        # Implement the forward pass
        dcell_subsystem_output = self.dcell(batch)
        dcell_linear_output = self.dcell_linear(dcell_subsystem_output)
        # if dcell_linear_output.size()[-1] == 1:
        #     dcell_linear_output = dcell_linear_output.squeeze(-1)
        return dcell_linear_output

    def on_train_start(self):
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log(
            "model/parameters_size", torch.tensor(parameter_size, dtype=torch.float32)
        )

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch.fitness
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.loss(y_hat, y, self.dcell.parameters())

        self.manual_backward(loss)  # error on this line
        opt.step()
        opt.zero_grad()
        # logging
        batch_size = batch.batch[-1].item() + 1
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        # Log
        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)
        self.train_metrics(y_hat_subsystems, y)
        self.train_metrics_root(y_hat_root, y)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.log_dict(self.train_metrics_root.compute(), sync_dist=True)
        self.train_metrics.reset()
        self.train_metrics_root.reset()
        pass

    def validation_step(self, batch, batch_idx):
        # Extract the batch vector
        y_hat = self(batch)
        y = batch.fitness
        loss = self.loss(y_hat, y, self.dcell.parameters())
        batch_size = batch.batch[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        # Log
        self.val_metrics(y_hat_subsystems, y)
        self.val_metrics_root(y_hat_root, y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.log_dict(self.val_metrics_root.compute(), sync_dist=True)
        self.val_metrics.reset()
        self.val_metrics_root.reset()

        # Stop tracing memory allocations
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
        y_hat = self(batch)
        y = batch.fitness
        loss = self.loss(y_hat, y, self.dcell.parameters())
        batch_size = batch.batch[-1].item() + 1
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        #
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
        self.test_metrics(y_hat_subsystems, y)
        self.test_metrics_root(y_hat_root, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        params = list(self.models["dcell"].parameters()) + list(
            self.models["dcell_linear"].parameters()
        )
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
