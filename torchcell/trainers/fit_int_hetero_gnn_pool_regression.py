"""Lightning task for fitness and gene-interaction regression on hetero GNN pools."""

import logging
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef

from torchcell.viz import fitness, genetic_interaction_score

log = logging.getLogger(__name__)


class RegressionTask(L.LightningModule):
    """Lightning task training a GNN to predict fitness and gene interaction."""

    train_metrics: nn.ModuleDict
    val_metrics: nn.ModuleDict
    test_metrics: nn.ModuleDict

    def __init__(
        self,
        model: nn.Module,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any],
        batch_size: int | None = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
        loss_func: nn.Module | None = None,
        grad_accumulation_schedule: dict[int, int] | None = None,
        device: str = "cuda",
        forward_transform: nn.Module | None = None,
        inverse_transform: nn.Module | None = None,
    ):
        """Store the model, loss, transforms, and per-stage regression metrics."""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.inverse_transform = inverse_transform
        self.current_accumulation_steps = 1
        self.loss_func = loss_func

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
                # "R2": R2Score(),
                # "Spearman": SpearmanCorrCoef(),
            }
        )

        # Create metrics for each stage and task
        for stage in ["train", "val", "test"]:
            metrics_dict = nn.ModuleDict(
                {
                    "fitness": reg_metrics.clone(prefix=f"{stage}/fitness/"),
                    "gene_interaction": reg_metrics.clone(
                        prefix=f"{stage}/gene_interaction/"
                    ),
                }
            )
            setattr(self, f"{stage}_metrics", metrics_dict)

        self.true_values: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.last_logged_best_step: int | None = None
        self.automatic_optimization = False

    def _log_prediction_table(
        self,
        stage: str,
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        dim_losses: torch.Tensor,
    ) -> None:
        """Log prediction tables for both fitness and gene interaction."""
        task_mapping = [("Fitness", "fitness"), ("GI", "gene_interaction")]

        for task_idx, (display_name, metadata_key) in enumerate(task_mapping):
            columns = [
                f"True {display_name}",
                f"Predicted {display_name}",
                f"{display_name} Loss",
            ]

            table_data = []
            for i in range(len(true_values)):
                row = [
                    true_values[i, task_idx].item(),
                    predictions[i, task_idx].item(),
                    dim_losses[task_idx].item(),
                ]
                table_data.append(row)

            table = wandb.Table(columns=columns, data=table_data)
            wandb.log({f"{stage}/{metadata_key}_predictions": table}, commit=False)

    def _compute_metrics_safely(self, metrics_dict: MetricCollection) -> dict[str, Any]:
        """Safely compute metrics with sufficient samples."""
        results: dict[str, Any] = {}
        for metric_name, metric in metrics_dict.items():
            try:
                results[metric_name] = metric.compute()
            except ValueError as e:
                # Skip metrics that need more samples or have no samples during sanity checking
                if any(
                    msg in str(e)
                    for msg in [
                        "Needs at least two samples",
                        "No samples to concatenate",
                    ]
                ):
                    continue
                raise e
        return results

    def forward(self, batch: Any) -> tuple[torch.Tensor, Any]:
        """Run the wrapped model on a batch."""
        return cast(tuple[torch.Tensor, Any], self.model(batch))

    def _shared_step(
        self, batch: Any, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass
        predictions, _ = self(batch)
        batch_size = predictions.size(0)

        # Extract targets
        fitness = batch["gene"].fitness_original.view(-1, 1)
        gene_interaction = batch["gene"].gene_interaction_original.view(-1, 1)
        targets = torch.cat([fitness, gene_interaction], dim=1)

        # Calculate loss and get per-dimension losses
        assert self.loss_func is not None
        loss, dim_losses = self.loss_func(predictions, targets)

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

        # Update metrics with NaN masking
        metrics = getattr(self, f"{stage}_metrics")

        # Fitness metrics
        fitness_mask = ~torch.isnan(targets[:, 0])
        if fitness_mask.any():
            metrics["fitness"](predictions[fitness_mask, 0], targets[fitness_mask, 0])

        # Gene interaction metrics
        gi_mask = ~torch.isnan(targets[:, 1])
        if gi_mask.any():
            metrics["gene_interaction"](predictions[gi_mask, 1], targets[gi_mask, 1])

        # Store predictions for validation/test plotting
        if stage in ["val", "test"]:
            self.true_values.append(targets.detach())
            self.predictions.append(predictions.detach())

        # Handle prediction table logging
        num_batches = (
            self.trainer.num_training_batches
            if stage == "train"
            else (
                self.trainer.num_val_batches
                if stage == "val"
                else self.trainer.num_test_batches
            )
        )

        num_batches = num_batches[0] if isinstance(num_batches, list) else num_batches

        if batch_idx == num_batches - 2:
            # Log prediction table
            self._log_prediction_table(
                stage=stage,
                true_values=targets,
                predictions=predictions,
                dim_losses=dim_losses,
            )

        return loss, predictions, targets

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a training step with manual optimization and gradient accumulation."""
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        if self.hparams["grad_accumulation_schedule"] is not None:
            loss = loss / self.current_accumulation_steps

        opt = cast(LightningOptimizer, self.optimizers())
        self.manual_backward(loss)

        if (
            self.hparams["grad_accumulation_schedule"] is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self.hparams["clip_grad_norm"]:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.hparams["clip_grad_norm_max_norm"]
                )
            opt.step()
            opt.zero_grad()

        self.log(
            "learning_rate",
            cast(LightningOptimizer, self.optimizers()).param_groups[0]["lr"],
            batch_size=batch["gene"].x.size(0),
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a validation step and return its loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a test step and return its loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def on_train_epoch_end(self) -> None:
        """Compute, log, and reset the training metrics at epoch end."""
        for metric_name, metric_dict in self.train_metrics.items():
            metric_collection = cast(MetricCollection, metric_dict)
            computed_metrics = self._compute_metrics_safely(metric_collection)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_collection.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics, box plots, and the best-model artifact."""
        # Log metrics
        for metric_name, metric_dict in self.val_metrics.items():
            metric_collection = cast(MetricCollection, metric_dict)
            computed_metrics = self._compute_metrics_safely(metric_collection)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_collection.reset()

        if self.trainer.sanity_checking or (
            self.current_epoch % self.hparams["boxplot_every_n_epochs"] != 0
        ):
            return

        true_values = torch.cat(self.true_values, dim=0) if self.true_values else None
        predictions = torch.cat(self.predictions, dim=0) if self.predictions else None

        if (
            not self.trainer.sanity_checking
            and true_values is not None
            and predictions is not None
        ):
            # Create box plots for fitness
            if torch.any(~torch.isnan(true_values[:, 0])):
                fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
                wandb.log({"val/fitness_box_plot": wandb.Image(fig_fitness)})
                plt.close(fig_fitness)

            # Create box plots for genetic interaction
            if torch.any(~torch.isnan(true_values[:, 1])):
                fig_gi = genetic_interaction_score.box_plot(
                    true_values[:, 1], predictions[:, 1]
                )
                wandb.log({"val/gene_interaction_box_plot": wandb.Image(fig_gi)})
                plt.close(fig_gi)

        self.true_values = []
        self.predictions = []

        # Log best model artifact
        current_global_step = self.global_step
        ckpt = cast(ModelCheckpoint, self.trainer.checkpoint_callback)
        assert ckpt is not None
        best_model_path = ckpt.best_model_path
        if best_model_path and current_global_step != self.last_logged_best_step:
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model checkpoint at step {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = current_global_step

    def on_test_epoch_end(self) -> None:
        """Log test metrics and box plots at epoch end."""
        # Log metrics
        for metric_name, metric_dict in self.test_metrics.items():
            metric_collection = cast(MetricCollection, metric_dict)
            computed_metrics = self._compute_metrics_safely(metric_collection)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_collection.reset()

        if self.trainer.sanity_checking:
            return

        true_values = torch.cat(self.true_values, dim=0) if self.true_values else None
        predictions = torch.cat(self.predictions, dim=0) if self.predictions else None

        if true_values is not None and predictions is not None:
            # Create box plots for test set
            if torch.any(~torch.isnan(true_values[:, 0])):
                fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
                wandb.log({"test/fitness_box_plot": wandb.Image(fig_fitness)})
                plt.close(fig_fitness)

            if torch.any(~torch.isnan(true_values[:, 1])):
                fig_gi = genetic_interaction_score.box_plot(
                    true_values[:, 1], predictions[:, 1]
                )
                wandb.log({"test/gene_interaction_box_plot": wandb.Image(fig_gi)})
                plt.close(fig_gi)

        self.true_values = []
        self.predictions = []

    def configure_optimizers(self) -> Any:  # Lightning accepts optimizer or config dict
        """Build the optimizer and LR scheduler from the hyperparameter config."""
        optimizer_class = getattr(torch.optim, self.hparams["optimizer_config"]["type"])
        optimizer_params = {
            k: v for k, v in self.hparams["optimizer_config"].items() if k != "type"
        }

        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        scheduler_params = {
            k: v for k, v in self.hparams["lr_scheduler_config"].items() if k != "type"
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
