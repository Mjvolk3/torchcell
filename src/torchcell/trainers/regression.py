# src/torchcell/trainers/regression.py
# [[src.torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/trainers/regression.py
# Test file: src/torchcell/trainers/test_regression.py


import matplotlib.pyplot as plt
import pytorch_lightning as pl
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

import wandb
from torchcell.losses import WeightedMSELoss

# use the specified style
plt.style.use("conf/torchcell.mplstyle")


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    target_key: str = "fitness"

    def __init__(
        self,
        models: dict[str, nn.Module],
        wt: Data,
        wt_step_freq: int = 10,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss: str = "mse",
        batch_size: int = None,
        **kwargs,
    ):
        super().__init__()

        # Lightning settings, doing this for WT embedding
        self.automatic_optimization = False

        self.model_ds = models["deep_set"]
        self.model_lin = models["mlp_ref_set"]
        self.wt = wt
        self.wt_step_freq = wt_step_freq
        self.is_wt_init = False
        self.wt_nodes_hat, self.wt_set_hat, self.wt_global_hat = None, None, None

        # loss
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "weighted_mse":
            mean_value = kwargs.get("mean_value")
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

        # optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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

    def setup(self, stage=None):
        self.model_ds = self.model_ds.to(self.device)
        self.model_lin = self.model_lin.to(self.device)

    def forward(self, x, batch):
        inst_nodes_hat, inst_set_hat = self.model_ds(x, batch)
        # This case is only for sanity checking
        if self.wt_set_hat is None:
            self.wt_set_hat = torch.ones_like(inst_set_hat)
        y_set_hat = self.wt_set_hat.mean(dim=0) - inst_set_hat
        y_hat = self.model_lin(y_set_hat)
        return y_hat

    def on_train_start(self):
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log("model/parameters_size", parameter_size)

    def train_wt(self):
        # CHECK on definition of global_step - refresh with epoch?
        if self.global_step == 0 and not self.is_wt_init:
            wt_batch = Batch.from_data_list([self.wt] * self.batch_size).to(self.device)
            self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                wt_batch.x, wt_batch.batch
            )

            self.is_wt_init = True
        if self.global_step == 0 or self.global_step % self.wt_step_freq == 0:
            # Global Loss
            # set up optimizer
            opt = self.optimizers()
            opt.zero_grad()

            wt_batch = Batch.from_data_list([self.wt] * self.batch_size).to(self.device)
            self.wt_y_hat = self(wt_batch.x, wt_batch.batch)
            loss_wt = self.loss(self.wt_y_hat, wt_batch.fitness)
            self.log("wt loss", loss_wt)
            self.log("wt mean", self.wt_y_hat.detach().mean())

            # get updated wt reference
            self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                wt_batch.x, wt_batch.batch
            )
            # Node Loss
            loss_nodes = self.loss_node(
                self.wt_nodes_hat, torch.ones_like(self.wt_nodes_hat)
            )
            self.log("wt loss_nodes", loss_nodes)
            self.manual_backward(loss_wt + loss_nodes)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            # finish optimization
            opt.step()
            opt.zero_grad()

            # Get updated wt reference
            self.model_ds.eval()
            with torch.no_grad():
                self.wt_nodes_hat, self.wt_set_hat = self.model_ds(
                    wt_batch.x, wt_batch.batch
                )
            self.model_ds.train()

    def training_step(self, batch, batch_idx):
        # Train on wt reference
        self.train_wt()
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.fitness, batch.batch
        # Pass the batch vector to the forward method
        y_hat = self(x, batch_vector)

        opt = self.optimizers()
        opt.zero_grad()
        loss = self.loss(y, y_hat)
        self.manual_backward(loss)  # error on this line
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
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
        x, y, batch_vector = batch.x, batch.fitness, batch.batch
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

        # Define bins
        bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, float("inf")]

        # Bin predictions and collect corresponding true values
        binned_true_values = []
        bin_labels = []
        for i in range(len(bins) - 1):
            mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
            binned_values = true_values[mask].cpu().numpy()
            binned_true_values.append(binned_values)
            bin_labels.append(f"{bins[i]}-{bins[i+1]}")

        # Create a box plot using matplotlib
        fig, ax = plt.subplots()
        ax.boxplot(binned_true_values, labels=bin_labels)
        ax.set_ylabel("True Values")
        ax.set_xlabel("Prediction Bins")
        ax.set_title("Box plot of True Values for each Prediction Bin")

        # Log the plot to wandb
        wandb.log({"binned_values_box_plot": wandb.Image(fig)})

        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

        # Close the matplotlib plot
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.fitness, batch.batch
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
        params = list(self.model_ds.parameters()) + list(self.model_lin.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
