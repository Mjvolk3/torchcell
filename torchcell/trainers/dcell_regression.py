import math
import tracemalloc

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
from torchcell.losses import DCellLoss, WeightedMSELoss
from torchcell.viz import fitness, genetic_interaction_score
import torchcell

style_file_path = osp.join(osp.dirname(torchcell.__file__), 'torchcell.mplstyle')
plt.style.use(style_file_path)

class DCellRegressionTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        target: str,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = None,
        train_wt_diff: bool = True,
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

        self.loss = DCellLoss()

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

        # wandb model artifact logging
        self.last_logged_best_step = None
        tracemalloc.start()

    def setup(self, stage=None):
        for model in self.models.values():
            model.to(self.device)
        self.true_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.predictions = torch.tensor([], dtype=torch.float32, device=self.device)

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
        self.train_metrics(y_hat_root, y)
        # Logging the correlation coefficients
        self.log(
            "train_pearson_subsystems",
            self.pearson_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_spearman_subsystems",
            self.spearman_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_pearson_root",
            self.pearson_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train_spearman_root",
            self.spearman_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

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
        self.val_metrics(y_hat_root, y)
        # Logging the correlation coefficients
        self.log(
            "val_pearson_subsystems",
            self.pearson_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_spearman_subsystems",
            self.spearman_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_pearson",
            self.pearson_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_spearman",
            self.spearman_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.true_values = torch.cat([self.true_values, y.detach()], dim=0)
        self.predictions = torch.cat(
            [self.predictions, y_hat_subsystems.detach()], dim=0
        )

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.boxplot_every_n_epochs != 0
        ):
            return

        if self.target == "fitness":
            fig = fitness.box_plot(self.true_values, self.predictions)
        elif self.target == "genetic_interaction_score":
            fig = genetic_interaction_score.box_plot(true_values, predictions)
        wandb.log({"binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
        # Clear the stored values for the next epoch
        self.true_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.predictions = torch.tensor([], dtype=torch.float32, device=self.device)

        current_global_step = self.global_step
        # HACK
        # Get the current memory usage
        current, peak = tracemalloc.get_traced_memory()
        print("======")
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        print("======")

        # Stop tracing memory allocations
        tracemalloc.stop()
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
        self.test_metrics(y_hat_root, y)
        # Logging the correlation coefficients
        self.log(
            "test_pearson_subsystems",
            self.pearson_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test_spearman_subsystems",
            self.spearman_corr(y_hat_subsystems, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test_pearson_root",
            self.pearson_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "test_spearman_root",
            self.spearman_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )

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
