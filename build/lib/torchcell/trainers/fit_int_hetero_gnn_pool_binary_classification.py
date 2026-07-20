"""Lightning task training a hetero GNN pool model for binary fitness classification."""

import logging
from typing import Any, Literal, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.parsing import AttributeDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from torchmetrics import MetricCollection

from torchcell.metrics.nan_tolerant_classification_metrics import (
    NaNTolerantAccuracy,
    NaNTolerantF1Score,
    NaNTolerantPrecision,
    NaNTolerantRecall,
)
from torchcell.metrics.nan_tolerant_metrics import (
    NaNTolerantMSE,
    NaNTolerantPearsonCorrCoef,
    NaNTolerantR2Score,
)
from torchcell.transforms.regression_to_classification import LabelBinningTransform
from torchcell.viz import fitness, genetic_interaction_score

log = logging.getLogger(__name__)


class ClassificationTask(L.LightningModule):
    """Lightning module training a hetero GNN pool model for binned classification."""

    train_metrics: nn.ModuleDict
    val_metrics: nn.ModuleDict
    test_metrics: nn.ModuleDict

    def __init__(
        self,
        model: nn.Module,
        bins: int,
        inverse_transform: BaseTransform,
        forward_transform: BaseTransform,
        label_type: str,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any],
        batch_size: int | None = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
        loss_func: nn.Module | None = None,
        grad_accumulation_schedule: dict[int, int] | None = None,
        device: str = "cuda",
    ):
        """Store the model, transforms, and configs and set up classification metrics.

        Args:
            model: The hetero GNN pool model to train.
            bins: Number of label bins (2 for binary classification).
            inverse_transform: Transform mapping classes back to values.
            forward_transform: Transform mapping values to binned labels.
            label_type: Name of the label being classified.
            optimizer_config: Optimizer hyperparameters.
            lr_scheduler_config: Learning-rate scheduler hyperparameters.
            batch_size: Batch size for logging.
            clip_grad_norm: Whether to clip gradient norm.
            clip_grad_norm_max_norm: Max gradient norm when clipping.
            boxplot_every_n_epochs: Boxplot logging frequency in epochs.
            loss_func: Loss module.
            grad_accumulation_schedule: Optional epoch-to-steps accumulation map.
            device: Device string.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func
        self.current_accumulation_steps = 1

        task: Literal["binary", "multiclass"]
        if bins == 2:
            task = "binary"
        else:
            task = "multiclass"
        # Define classification metrics
        class_metrics = MetricCollection(
            {
                "Accuracy": NaNTolerantAccuracy(task=task),
                "F1": NaNTolerantF1Score(task=task),
                "Precision": NaNTolerantPrecision(task=task),
                "Recall": NaNTolerantRecall(task=task),
            }
        )

        # Define regression metrics
        reg_metrics = MetricCollection(
            {
                "MSE": NaNTolerantMSE(squared=True),
                "RMSE": NaNTolerantMSE(squared=False),
                "R2": NaNTolerantR2Score(),
                "Pearson": NaNTolerantPearsonCorrCoef(),
            }
        )

        # Create metrics for each stage (train/val/test)
        for stage in ["train", "val", "test"]:
            metrics_dict = nn.ModuleDict(
                {
                    "fitness": class_metrics.clone(prefix=f"{stage}/fitness/"),
                    "gene_interaction": class_metrics.clone(
                        prefix=f"{stage}/gene_interaction/"
                    ),
                    "fitness_reg": reg_metrics.clone(prefix=f"{stage}/fitness_reg/"),
                    "gene_interaction_reg": reg_metrics.clone(
                        prefix=f"{stage}/gene_interaction_reg/"
                    ),
                }
            )
            setattr(self, f"{stage}_metrics", metrics_dict)

        self.true_reg_values: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.last_logged_best_step: int | None = None
        self.automatic_optimization = False

    @property
    def _hp(self) -> AttributeDict:
        """Typed view of ``self.hparams`` (always an ``AttributeDict`` at runtime)."""
        return cast(AttributeDict, self.hparams)

    def _log_prediction_table(
        self,
        stage: str,
        true_reg_values: torch.Tensor,
        true_class_values: torch.Tensor,
        logits: torch.Tensor,
        inverse_preds: torch.Tensor,
        dim_losses: torch.Tensor,
    ) -> None:
        """Log two tables (one per task) with regression values, bin information, and ranges."""
        num_bins = true_class_values.shape[1] // 2
        task_mapping = [("Fitness", "fitness"), ("GI", "gene_interaction")]

        # Get the binning transform from the forward transform composition
        forward_transform = self._hp.forward_transform
        binning_transform = next(
            t
            for t in forward_transform.transforms
            if isinstance(t, LabelBinningTransform)
        )

        # Process each task separately
        for task_idx, (display_name, metadata_key) in enumerate(task_mapping):
            # Get logits for this task
            start_idx = task_idx * num_bins
            end_idx = (task_idx + 1) * num_bins
            task_logits = logits[:, start_idx:end_idx]

            # Get denormalized bin edges from forward transform metadata
            bin_edges_denorm = binning_transform.label_metadata[metadata_key][
                "bin_edges_denormalized"
            ]
            bin_edges_denorm = torch.tensor(bin_edges_denorm, dtype=torch.float32)

            # Determine true and predicted bins based on label type
            if self._hp.label_type == "ordinal":
                # For ordinal, count number of 1s for true bins
                true_bins = torch.sum(
                    true_class_values[:, start_idx:end_idx] > 0.5, dim=1
                )
                # For predictions, count number of positive logits
                pred_bins = torch.sum(task_logits > 0, dim=1)
            else:  # categorical
                true_bins = torch.argmax(true_class_values[:, start_idx:end_idx], dim=1)
                pred_bins = torch.argmax(task_logits, dim=1)

            # Create table columns
            columns = [
                f"True Reg ({display_name})",
                "True Bin",
                "True Bin Start",
                "True Bin End",
                "Predicted Bin",
                "Predicted Bin Start",
                "Predicted Bin End",
                f"Predicted Reg ({display_name})",
                f"{display_name} Loss",
            ]

            # Prepare table data
            table_data = []
            for i in range(len(true_reg_values)):
                true_bin = true_bins[i].item()
                pred_bin = pred_bins[i].item()

                # Get bin ranges for true bin (clamp to valid indices)
                true_bin_clamped = cast(
                    int, max(0, min(true_bin, len(bin_edges_denorm) - 2))
                )
                true_bin_start = bin_edges_denorm[true_bin_clamped].item()
                true_bin_end = bin_edges_denorm[true_bin_clamped + 1].item()

                # Get bin ranges for predicted bin (clamp to valid indices)
                pred_bin_clamped = cast(
                    int, max(0, min(pred_bin, len(bin_edges_denorm) - 2))
                )
                pred_bin_start = bin_edges_denorm[pred_bin_clamped].item()
                pred_bin_end = bin_edges_denorm[pred_bin_clamped + 1].item()

                row = [
                    true_reg_values[i, task_idx].item(),
                    true_bin,
                    true_bin_start,
                    true_bin_end,
                    pred_bin,
                    pred_bin_start,
                    pred_bin_end,
                    inverse_preds[i, task_idx].item(),
                    dim_losses[task_idx].item(),
                ]
                table_data.append(row)

            # Create and log table
            table = wandb.Table(columns=columns, data=table_data)
            wandb.log({f"{stage}/{metadata_key}_predictions": table}, commit=False)

    # def forward(self, x_dict, edge_index_dict, batch_dict):
    #     if self.model.learnable_embedding:
    #         return self.model(None, edge_index_dict, batch_dict)
    #     else:
    #         return self.model(x_dict, edge_index_dict, batch_dict)

    def forward(self, batch: Any) -> tuple[torch.Tensor, Any]:
        """Run the model on the batch and return its outputs."""
        result: tuple[torch.Tensor, Any] = self.model(batch)
        return result

    def _shared_step(
        self, batch: Any, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass to get logits
        logits, _ = self(batch)
        batch_size = logits.size(0)

        # Extract targets
        fitness = batch["gene"].fitness
        gene_interaction = batch["gene"].gene_interaction
        y = torch.cat([fitness, gene_interaction], dim=1)

        # Calculate loss
        loss, dim_losses = cast(nn.Module, self.loss_func)(logits, y)

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

        # Split logits for each task
        num_classes = self._hp.bins
        fitness_logits = logits[:, :num_classes]
        gene_int_logits = logits[:, num_classes:]

        # Convert targets to class indices based on label type
        if self._hp.label_type == "ordinal":
            # For ordinal labels, count number of 1s to determine class
            # Handle NaN values
            fitness_nan_mask = torch.isnan(fitness).any(dim=1)
            gene_int_nan_mask = torch.isnan(gene_interaction).any(dim=1)

            # Create float tensors to store targets
            fitness_targets = torch.sum(fitness > 0.5, dim=1).float()
            gene_int_targets = torch.sum(gene_interaction > 0.5, dim=1).float()
            fitness_targets[fitness_nan_mask] = float("nan")
            gene_int_targets[gene_int_nan_mask] = float("nan")
        else:  # categorical or soft
            fitness_targets = torch.argmax(fitness, dim=1)
            gene_int_targets = torch.argmax(gene_interaction, dim=1)

        # Update classification metrics
        metrics = getattr(self, f"{stage}_metrics")
        metrics["fitness"](fitness_logits, fitness_targets)
        metrics["gene_interaction"](gene_int_logits, gene_int_targets)

        pred_data = HeteroData()
        if self._hp.label_type == "ordinal":
            # For ordinal, convert logits to binary thresholds
            pred_data["gene"] = {
                "fitness": (fitness_logits > 0)
                .float()
                .detach(),  # Convert to ordinal thresholds
                "gene_interaction": (gene_int_logits > 0).float().detach(),
            }
        else:  # categorical
            # For categorical, keep raw logits for softmax in inverse transform
            pred_data["gene"] = {
                "fitness": fitness_logits.detach(),
                "gene_interaction": gene_int_logits.detach(),
            }

        reg_pred_data = self.inverse_transform(pred_data)

        # Get original regression values
        y_original = torch.cat(
            [
                batch["gene"].fitness_original.view(-1, 1),
                batch["gene"].gene_interaction_original.view(-1, 1),
            ],
            dim=1,
        )

        # Update regression metrics
        metrics["fitness_reg"](reg_pred_data["gene"]["fitness"], y_original[:, 0])
        metrics["gene_interaction_reg"](
            reg_pred_data["gene"]["gene_interaction"], y_original[:, 1]
        )

        # Store for validation/test plotting
        if stage in ["val", "test"]:
            self.true_reg_values.append(y_original.detach())
            self.predictions.append(logits.detach())

        # log table
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
            # Get inverse predictions
            pred_reg_values = torch.cat(
                [
                    reg_pred_data["gene"]["fitness"].view(-1, 1),
                    reg_pred_data["gene"]["gene_interaction"].view(-1, 1),
                ],
                dim=1,
            )
            self._log_prediction_table(
                stage=stage,
                true_reg_values=y_original,
                true_class_values=y,
                logits=logits,
                inverse_preds=pred_reg_values,
                dim_losses=dim_losses,
            )

        return loss, logits, y

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a training step and return the loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        if self._hp.grad_accumulation_schedule is not None:
            loss = loss / self.current_accumulation_steps

        opt = cast(LightningOptimizer, self.optimizers())
        self.manual_backward(loss)

        if (
            self._hp.grad_accumulation_schedule is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self._hp.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self._hp.clip_grad_norm_max_norm
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
        """Run a validation step over the batch."""
        loss, _, _ = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run a test step over the batch."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def on_train_epoch_end(self) -> None:
        """Aggregate and log training metrics at epoch end."""
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = cast(Any, metric_dict).compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            cast(Any, metric_dict).reset()

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log validation metrics and plots at epoch end."""
        # Log metrics
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = cast(Any, metric_dict).compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            cast(Any, metric_dict).reset()

        if self.trainer.sanity_checking or (
            self.current_epoch % self._hp.boxplot_every_n_epochs != 0
        ):
            return

        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Create HeteroData object for predictions
        pred_data = HeteroData()
        num_bins = self.model.out_channels
        if self._hp.label_type == "ordinal":
            # For ordinal, convert predictions to binary thresholds
            pred_data["gene"] = {
                "fitness": (predictions[:, :num_bins] > 0).float(),
                "gene_interaction": (predictions[:, num_bins:] > 0).float(),
            }
        else:  # categorical
            # For categorical, keep raw logits
            pred_data["gene"] = {
                "fitness": predictions[:, :num_bins],
                "gene_interaction": predictions[:, num_bins:],
            }

        # Apply inverse transform only to predictions
        pred_data = self.inverse_transform(pred_data)

        # Extract values for plotting
        true_fitness = true_reg_values[:, 0]
        true_gi = true_reg_values[:, 1]
        pred_fitness = pred_data["gene"]["fitness"]
        pred_gi = pred_data["gene"]["gene_interaction"]

        if not self.trainer.sanity_checking:
            fig_fitness = fitness.box_plot(true_fitness, pred_fitness)
            wandb.log({"val/fitness_box_plot": wandb.Image(fig_fitness)})
            plt.close(fig_fitness)

            fig_gi = genetic_interaction_score.box_plot(true_gi, pred_gi)
            wandb.log({"val/gene_interaction_box_plot": wandb.Image(fig_gi)})
            plt.close(fig_gi)

        self.true_reg_values = []
        self.predictions = []

        current_global_step = self.global_step
        if (
            cast(Any, self.trainer.checkpoint_callback).best_model_path
            and current_global_step != self.last_logged_best_step
        ):
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model checkpoint at step {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(
                cast(Any, self.trainer.checkpoint_callback).best_model_path
            )
            wandb.log_artifact(artifact)
            self.last_logged_best_step = current_global_step

    def on_test_epoch_end(self) -> None:
        """Aggregate and log test metrics and artifacts at epoch end."""
        # Log metrics
        for metric_name, metric_dict in self.test_metrics.items():
            computed_metrics = cast(Any, metric_dict).compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            cast(Any, metric_dict).reset()

        if self.trainer.sanity_checking:
            return

        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Create HeteroData object for predictions
        pred_data = HeteroData()
        num_bins = self.model.out_channels
        if self._hp.label_type == "ordinal":
            # For ordinal, convert predictions to binary thresholds
            pred_data["gene"] = {
                "fitness": (predictions[:, :num_bins] > 0).float(),
                "gene_interaction": (predictions[:, num_bins:] > 0).float(),
            }
        else:  # categorical
            # For categorical, keep raw logits
            pred_data["gene"] = {
                "fitness": predictions[:, :num_bins],
                "gene_interaction": predictions[:, num_bins:],
            }

        # Apply inverse transform only to predictions
        pred_data = self.inverse_transform(pred_data)

        # Extract values for plotting
        true_fitness = true_reg_values[:, 0]
        true_gi = true_reg_values[:, 1]
        pred_fitness = pred_data["gene"]["fitness"]
        pred_gi = pred_data["gene"]["gene_interaction"]

        # Create box plots - note the test/ prefix instead of val/
        fig_fitness = fitness.box_plot(true_fitness, pred_fitness)
        wandb.log({"test/fitness_box_plot": wandb.Image(fig_fitness)})

        fig_gi = genetic_interaction_score.box_plot(true_gi, pred_gi)
        wandb.log({"test/gene_interaction_box_plot": wandb.Image(fig_gi)})

        self.true_reg_values = []
        self.predictions = []

    def configure_optimizers(self) -> Any:
        """Build and return the optimizer and learning-rate scheduler."""
        optimizer_class = getattr(optim, self._hp.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self._hp.optimizer_config.items() if k != "type"
        }

        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        scheduler_params = {
            k: v for k, v in self._hp.lr_scheduler_config.items() if k != "type"
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
