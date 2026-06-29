# torchcell/trainers/dcell_regression_slim.py
# [[torchcell.trainers.dcell_regression_slim]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/dcell_regression_slim.py
# Test file: torchcell/trainers/test_dcell_regression_slim.py
"""Slim Lightning trainer for DCell regression with subsystem and root metrics."""

import os.path as osp
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch_geometric.data import HeteroData
from torchmetrics import (
    MeanAbsoluteError,
    MeanSquaredError,
    MetricCollection,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)

import torchcell

# TODO name change
# from torchcell.losses import DCellLoss, WeightedMSELoss
from torchcell.losses import DCellLoss

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
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
        batch_size: int | None = None,
        alpha: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """Register the DCell submodels, DCell loss, optimizer settings, and metrics.

        Args:
            models: Dict of named submodels (expects "dcell" and "dcell_linear").
            target: Target name being regressed.
            boxplot_every_n_epochs: Epoch interval for box plots.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            batch_size: Batch size used for logging.
            alpha: Regularization weight for the DCell loss.
            **kwargs: Additional unused keyword arguments.
        """
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
        self.last_logged_best_step: int | None = None

    def _submodel(self, name: str) -> nn.Module:
        """Return a submodel registered via ``setattr`` in ``__init__``.

        Submodels are registered dynamically, so ``nn.Module.__getattr__``
        types them as ``Tensor | Module``; the invariant that these attributes
        are always ``nn.Module`` holds.
        """
        return cast(nn.Module, getattr(self, name))

    def setup(self, stage: str | None = None) -> None:
        """Move all submodels to the active device at the start of each stage."""
        for model in self.models.values():
            model.to(self.device)

    def forward(self, batch: HeteroData) -> dict[str, torch.Tensor]:
        """Run the batch through the DCell subsystems and linear head."""
        # Implement the forward pass
        dcell_subsystem_output = self._submodel("dcell")(batch)
        dcell_linear_output = self._submodel("dcell_linear")(dcell_subsystem_output)
        # if dcell_linear_output.size()[-1] == 1:
        #     dcell_linear_output = dcell_linear_output.squeeze(-1)
        return cast(dict[str, torch.Tensor], dcell_linear_output)

    def on_train_start(self) -> None:
        """Log the total model parameter count when training starts."""
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log(
            "model/parameters_size", torch.tensor(parameter_size, dtype=torch.float32)
        )

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a manual training step and log loss plus subsystem and root metrics."""
        y_hat = self(batch)
        y = batch.fitness
        opt = cast(LightningOptimizer, self.optimizers())
        opt.zero_grad()
        loss: torch.Tensor = self.loss(y_hat, y, self._submodel("dcell").parameters())

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

    def on_train_epoch_end(self) -> None:
        """Log and reset the subsystem and root training metrics at epoch end."""
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.log_dict(self.train_metrics_root.compute(), sync_dist=True)
        self.train_metrics.reset()
        self.train_metrics_root.reset()
        pass

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Run a validation step and update subsystem and root metrics."""
        # Extract the batch vector
        y_hat = self(batch)
        y = batch.fitness
        loss = self.loss(y_hat, y, self._submodel("dcell").parameters())
        batch_size = batch.batch[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        # Log
        self.val_metrics(y_hat_subsystems, y)
        self.val_metrics_root(y_hat_root, y)

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and log the best model as a wandb artifact."""
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.log_dict(self.val_metrics_root.compute(), sync_dist=True)
        self.val_metrics.reset()
        self.val_metrics_root.reset()

        # Stop tracing memory allocations
        current_global_step = self.global_step
        ckpt = cast(ModelCheckpoint, self.trainer.checkpoint_callback)
        if ckpt.best_model_path and current_global_step != self.last_logged_best_step:
            # Save model as a W&B artifact
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model on validation epoch end step - {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(ckpt.best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = (
                current_global_step  # update the last logged step
            )

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Run a test step and update subsystem and root metrics."""
        y_hat = self(batch)
        y = batch.fitness
        loss = self.loss(y_hat, y, self._submodel("dcell").parameters())
        batch_size = batch.batch[-1].item() + 1
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        #
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
        self.test_metrics(y_hat_subsystems, y)
        self.test_metrics_root(y_hat_root, y)

    def on_test_epoch_end(self) -> None:
        """Log and reset the test metrics at epoch end."""
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return an Adam optimizer over the DCell and linear-head parameters."""
        params = list(self.models["dcell"].parameters()) + list(
            self.models["dcell_linear"].parameters()
        )
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
