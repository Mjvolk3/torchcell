"""Lightning training task for DCell graph-based fitness regression."""

import os.path as osp
from typing import cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

# TODO will need to fix
# from torchcell.losses import DCellLoss, WeightedMSELoss
from torchcell.losses import DCellLoss
from torchcell.viz import fitness, genetic_interaction_score

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)

PLOTTED_TARGETS = ("fitness", "genetic_interaction_score")


class DCellRegressionTask(L.LightningModule):
    """LightningModule for training models on graph-based regression datasets."""

    def __init__(
        self,
        models: dict[str, nn.Module],
        target: str,
        boxplot_every_n_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int | None = None,
        train_wt_diff: bool = True,
        **kwargs: object,
    ) -> None:
        """Set up models, loss, optimizer config, and metric collections.

        Args:
            models: Mapping of submodel name to module (expects "dcell" and
                "dcell_linear").
            target: Target column name ("fitness" or "genetic_interaction_score").
            boxplot_every_n_epochs: Epoch interval for logging box plots.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            batch_size: Batch size used for logging.
            train_wt_diff: Whether to train on wild-type differences.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If ``target`` has no box plot (not in ``PLOTTED_TARGETS``).
        """
        super().__init__()
        if target not in PLOTTED_TARGETS:
            raise ValueError(
                f"Unknown target '{target}': expected one of {PLOTTED_TARGETS}."
            )
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
        self.last_logged_best_step: int | None = None

    def _submodel(self, name: str) -> nn.Module:
        """Return a submodel registered via ``setattr`` in ``__init__``.

        Submodels are registered dynamically, so ``nn.Module.__getattr__``
        types them as ``Tensor | Module``; the invariant that these attributes
        are always ``nn.Module`` holds.
        """
        return cast(nn.Module, getattr(self, name))

    def setup(self, stage: str | None = None) -> None:
        """Move submodels to device and init prediction/true-value buffers."""
        for model in self.models.values():
            model.to(self.device)
        self.true_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.predictions = torch.tensor([], dtype=torch.float32, device=self.device)

    def forward(self, batch: HeteroData) -> dict[str, torch.Tensor]:
        """Run the DCell subsystem and linear heads, returning per-node outputs."""
        # Implement the forward pass
        dcell_subsystem_output = self._submodel("dcell")(batch)
        dcell_linear_output = self._submodel("dcell_linear")(dcell_subsystem_output)
        # if dcell_linear_output.size()[-1] == 1:
        #     dcell_linear_output = dcell_linear_output.squeeze(-1)
        return cast(dict[str, torch.Tensor], dcell_linear_output)

    def _loss(self, y_hat: dict[str, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """Apply ``DCellLoss(predictions, outputs, target)`` to the head outputs.

        The root head is the prediction; every head (the loss skips ``GO:ROOT``)
        goes in ``outputs["linear_outputs"]`` for the auxiliary term.
        """
        linear_outputs = {name: out.squeeze(-1) for name, out in y_hat.items()}
        loss, _ = self.loss(
            linear_outputs["GO:ROOT"], {"linear_outputs": linear_outputs}, y
        )
        return cast(torch.Tensor, loss)

    def on_train_start(self) -> None:
        """Log the total parameter count at the start of training."""
        # Calculate the model size (number of parameters)
        parameter_size = sum(p.numel() for p in self.parameters())
        # Log it using wandb
        self.log(
            "model/parameters_size", torch.tensor(parameter_size, dtype=torch.float32)
        )

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a manual-optimization training step and log loss and metrics."""
        y_hat = self(batch)
        y = batch.fitness
        opt = cast(LightningOptimizer, self.optimizers())
        opt.zero_grad()
        loss = self._loss(y_hat, y)

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

    def on_train_epoch_end(self) -> None:
        """Log and reset accumulated training metrics."""
        self.log_dict(self.train_metrics.compute(), sync_dist=True)
        self.train_metrics.reset()

    def validation_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Run a validation step, logging loss/metrics and storing predictions."""
        # Extract the batch vector
        y_hat = self(batch)
        y = batch.fitness
        loss = self._loss(y_hat, y)
        batch_size = batch.batch[-1].item() + 1
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        # Log
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
            "val_pearson_root",
            self.pearson_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val_spearman_root",
            self.spearman_corr(y_hat_root, y),
            batch_size=batch_size,
            sync_dist=True,
        )
        # the sanity check runs before training and is never plotted
        if self.trainer.sanity_checking:
            return
        self.true_values = torch.cat([self.true_values, y.detach()], dim=0)
        self.predictions = torch.cat([self.predictions, y_hat_root.detach()], dim=0)

    def on_validation_epoch_end(self) -> None:
        """Log val metrics and, on plotting epochs, the root-prediction box plot."""
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.boxplot_every_n_epochs != 0
        ):
            return

        if self.target == "fitness":
            fig = fitness.box_plot(self.true_values, self.predictions)
        else:
            fig = genetic_interaction_score.box_plot(self.true_values, self.predictions)
        wandb.log({"binned_values_box_plot": wandb.Image(fig)})
        plt.close(fig)
        # Clear the stored values for the next epoch
        self.true_values = torch.tensor([], dtype=torch.float32, device=self.device)
        self.predictions = torch.tensor([], dtype=torch.float32, device=self.device)

    def on_train_epoch_start(self) -> None:
        """Log the checkpoint the previous epoch saved, if it is a new best."""
        self._log_best_checkpoint()

    def on_train_end(self) -> None:
        """Log the checkpoint the final epoch saved, if it is a new best."""
        self._log_best_checkpoint()

    def _log_best_checkpoint(self) -> None:
        """Log the best checkpoint as a W&B artifact once per global step.

        With validation every epoch, ``ModelCheckpoint`` saves in ``on_train_epoch_end``
        after every module hook of that epoch, so the first module hooks that see the
        epoch's checkpoint are the next ``on_train_epoch_start`` and, for the last
        epoch, ``on_train_end``; the global step there is the step it was saved at.
        With checkpointing disabled there is nothing to log.
        """
        current_global_step = self.global_step
        ckpt = cast(ModelCheckpoint | None, self.trainer.checkpoint_callback)
        if (
            ckpt is None
            or not ckpt.best_model_path
            or current_global_step == self.last_logged_best_step
        ):
            return
        # Save model as a W&B artifact
        artifact = wandb.Artifact(
            name=f"model-global_step-{current_global_step}",
            type="model",
            description=f"Best model checkpoint at step - {current_global_step}",
            metadata=dict(self.hparams),
        )
        artifact.add_file(ckpt.best_model_path)
        wandb.log_artifact(artifact)
        self.last_logged_best_step = current_global_step

    def test_step(self, batch: HeteroData, batch_idx: int) -> None:
        """Run a test step and log loss, regression metrics, and correlations."""
        y_hat = self(batch)
        y = batch.fitness
        loss = self._loss(y_hat, y)
        batch_size = batch.batch[-1].item() + 1
        # Flatten
        y_hat_root = y_hat["GO:ROOT"].squeeze(1)
        y_hat_stacked = torch.stack([v.squeeze() for v in y_hat.values()])
        y_hat_subsystems = y_hat_stacked.mean(0)
        #
        self.log("test_loss", loss, batch_size=batch_size, sync_dist=True)
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

    def on_test_epoch_end(self) -> None:
        """Log and reset accumulated test metrics."""
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build an Adam optimizer over the DCell and linear-head parameters."""
        params = list(self.models["dcell"].parameters()) + list(
            self.models["dcell_linear"].parameters()
        )
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer
