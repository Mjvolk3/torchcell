# src/torchcell/trainers/regression.py
# [[src.torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/trainers/regression.py
# Test file: src/torchcell/trainers/test_regression.py


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

import wandb

# use the specified style
plt.style.use("conf/torchcell.mplstyle")


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    target_key: str = "dmf"

    def __init__(
        self, model: nn.Module, learning_rate: float = 1e-3, loss: str = "mse"
    ):
        super().__init__()

        self.model = model

        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid."
                "Currently, supports 'mse' or 'mae' loss."
            )

        self.learning_rate = learning_rate

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
        self.plot_every_n_epochs = 5
        self.true_values = []
        self.predictions = []

    def forward(self, x, batch):
        return self.model(x, batch)

    def on_train_start(self):
        # Calculate the model size (number of parameters) 
        parameter_size = sum(p.numel() for p in self.parameters())

        # Log it using wandb
        self.log("model/parameters_size", parameter_size)

    def training_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.dmf, batch.batch
        # Pass the batch vector to the forward method
        y_hat = self(x, batch_vector)

        loss = self.loss(y_hat, y)
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
        x, y, batch_vector = batch.x, batch.dmf, batch.batch
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
            self.current_epoch % self.plot_every_n_epochs != 0
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
        x, y, batch_vector = batch.x, batch.dmf, batch.batch
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
