# src/torchcell/trainers/regression
# [[src.torchcell.trainers.regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/src/torchcell/trainers/regression
# Test file: src/torchcell/trainers/test_regression

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


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    target_key: str = "dmf_fitness"

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

        # Used in end
        self.true_values = []
        self.predictions = []

    def forward(self, x, batch):
        return self.model(x, batch)

    def training_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.dmf_fitness, batch.batch
        # Pass the batch vector to the forward method
        y_hat = self(x, batch_vector)

        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("train_loss", loss, batch_size=batch_size)
        self.train_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log("train_pearson", self.pearson_corr(y_hat, y), batch_size=batch_size)
        self.log("train_spearman", self.spearman_corr(y_hat, y), batch_size=batch_size)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.dmf_fitness, batch.batch
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size)
        self.val_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log("val_pearson", self.pearson_corr(y_hat, y), batch_size=batch_size)
        self.log("val_spearman", self.spearman_corr(y_hat, y), batch_size=batch_size)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        # Extract the batch vector
        x, y, batch_vector = batch.x, batch.dmf_fitness, batch.batch
        y_hat = self(x, batch_vector)
        loss = self.loss(y_hat, y)
        batch_size = batch_vector[-1].item() + 1
        self.log("test_loss", loss, batch_size=batch_size)
        self.test_metrics(y_hat, y)
        # Logging the correlation coefficients
        self.log("test_pearson", self.pearson_corr(y_hat, y), batch_size=batch_size)
        self.log("test_spearman", self.spearman_corr(y_hat, y), batch_size=batch_size)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
