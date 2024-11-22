import lightning as L
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
import logging
import sys
import torch.optim as optim
from typing import Optional
import os.path as osp
from torch.nn.functional import binary_cross_entropy_with_logits
from torchcell.metrics.nan_tolerant_classification_metrics import (
    NaNTolerantAccuracy,
    NaNTolerantF1Score,
    NaNTolerantAUROC,
)
from torchcell.viz import fitness, genetic_interaction_score
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform

log = logging.getLogger(__name__)


class ClassificationTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        inverse_transform: BaseTransform,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        batch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
        intermediate_loss_weight: float = 0.1,
        loss_func: nn.Module = None,
        grad_accumulation_schedule: Optional[dict[int, int]] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func
        self.current_accumulation_steps = 1

        # Define classification metrics
        metrics = MetricCollection(
            {
                "Accuracy": NaNTolerantAccuracy(task="binary"),
                "F1": NaNTolerantF1Score(task="binary"),
                # "AUROC": NaNTolerantAUROC(task="binary"),
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

        self.true_values = []
        self.predictions = []
        self.last_logged_best_step = None
        self.automatic_optimization = False

    def forward(self, x_dict, edge_index_dict, batch_dict):
        return self.model(x_dict, edge_index_dict, batch_dict)

    def _shared_step(self, batch, batch_idx, stage="train"):
        """
        Shared step for training, validation and testing.
        """
        # Create input dictionaries
        x_dict = {"gene": batch["gene"].x}
        edge_index_dict = {
            ("gene", "physical_interaction", "gene"): batch[
                "gene", "physical_interaction", "gene"
            ].edge_index,
            ("gene", "regulatory_interaction", "gene"): batch[
                "gene", "regulatory_interaction", "gene"
            ].edge_index,
        }
        batch_dict = {"gene": batch["gene"].batch}

        # Forward pass to get logits
        logits = self(x_dict, edge_index_dict, batch_dict)
        batch_size = x_dict["gene"].size(0)
        num_classes = self.loss_func.num_classes

        # Extract and reshape targets - these are one-hot encoded from your transform
        fitness = batch["gene"].fitness  # Shape: [batch_size, 2]
        gene_interaction = batch["gene"].gene_interaction  # Shape: [batch_size, 2]
        y = torch.cat([fitness, gene_interaction], dim=1)  # Shape: [batch_size, 4]

        # Calculate loss
        loss, dim_losses = self.loss_func(logits, y)

        # Logging
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log(
            f"{stage}/fitness_loss",
            dim_losses[0],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/gene_interaction_loss",
            dim_losses[1],
            batch_size=batch_size,
            sync_dist=True,
        )

        # Update metrics
        metrics = getattr(self, f"{stage}_metrics")

        # Split logits and targets for each task
        fitness_logits = logits[:, :2]  # First task's logits [batch_size, 2]
        gene_int_logits = logits[:, 2:]  # Second task's logits [batch_size, 2]

        # Convert targets to class indices for metrics (from one-hot to indices)
        fitness_target = torch.argmax(fitness, dim=1)  # Shape: [batch_size]
        gene_int_target = torch.argmax(gene_interaction, dim=1)  # Shape: [batch_size]

        # Update metrics for each task
        metrics["fitness"](fitness_logits, fitness_target)
        metrics["gene_interaction"](gene_int_logits, gene_int_target)

        if stage in ["val", "test"]:
            # Store original one-hot targets and predicted probabilities
            self.true_values.append(y.detach())
            self.predictions.append(torch.softmax(logits, dim=-1).detach())


        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        if self.hparams.grad_accumulation_schedule is not None:
            loss = loss / self.current_accumulation_steps

        opt = self.optimizers()
        self.manual_backward(loss)

        if (
            self.hparams.grad_accumulation_schedule is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self.hparams.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.hparams.clip_grad_norm_max_norm
                )
            opt.step()
            opt.zero_grad()

        self.log(
            "learning_rate",
            self.optimizers().param_groups[0]["lr"],
            batch_size=batch["gene"].x.size(0),
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def on_validation_epoch_end(self):
        # Compute and log metrics
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"val/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.hparams.boxplot_every_n_epochs != 0
        ):
            return

        # Combine predictions and true values from all batches
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Create box plots for both fitness and gene interaction
        if not self.trainer.sanity_checking:
            # Fitness box plot
            fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
            wandb.log({"val/fitness_box_plot": wandb.Image(fig_fitness)})

            # Gene interaction box plot
            fig_gi = genetic_interaction_score.box_plot(
                true_values[:, 1], predictions[:, 1]
            )
            wandb.log({"val/gene_interaction_box_plot": wandb.Image(fig_gi)})

        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

        # Log model artifact
        current_global_step = self.global_step
        if (
            self.trainer.checkpoint_callback.best_model_path
            and current_global_step != self.last_logged_best_step
        ):
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model checkpoint at step {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(self.trainer.checkpoint_callback.best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = current_global_step

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.hparams.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hparams.optimizer_config.items() if k != "type"
        }

        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        scheduler_params = {
            k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"
        }
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
