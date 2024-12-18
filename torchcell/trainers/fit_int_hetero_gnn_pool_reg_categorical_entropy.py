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
from torchcell.losses.multi_dim_nan_tolerant import MseCategoricalEntropyRegLoss
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics import MeanSquaredError, R2Score, PearsonCorrCoef, SpearmanCorrCoef

from torchcell.viz import fitness, genetic_interaction_score
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from torchcell.transforms.regression_to_classification import LabelBinningTransform
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class RegCategoricalEntropyTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        bins: int,
        inverse_transform: BaseTransform,
        forward_transform: BaseTransform,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        batch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        boxplot_every_n_epochs: int = 1,
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

        if bins == 2:
            task = "binary"
        else:
            task = "multiclass"
        # Define classification metrics
        class_metrics = MetricCollection(
            {
                "Accuracy": Accuracy(task=task, num_classes=bins),
                "F1": F1Score(task=task, num_classes=bins),
                "Precision": Precision(task=task, num_classes=bins),
                "Recall": Recall(task=task, num_classes=bins),
            }
        )

        # Define regression metrics
        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                # "R2": R2Score(),
                "Pearson": PearsonCorrCoef(),
                # "Spearman": SpearmanCorrCoef(),
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

        self.true_reg_values = []
        self.predictions = []
        self.last_logged_best_step = None
        self.automatic_optimization = False

    def _log_prediction_table(
        self,
        stage: str,
        true_reg_values: torch.Tensor,
        true_class_values: torch.Tensor,
        logits: torch.Tensor,
        inverse_preds: torch.Tensor,
        dim_losses: torch.Tensor,
    ):
        """Log two tables (one per task) with essential regression and bin information."""
        num_bins = true_class_values.shape[1] // 2
        task_mapping = [("Fitness", "fitness"), ("GI", "gene_interaction")]

        # Get the binning transform from the forward transform composition
        forward_transform = self.hparams.forward_transform
        binning_transform = next(
            t
            for t in forward_transform.transforms
            if isinstance(t, LabelBinningTransform)
        )

        # Get mean losses per dimension
        mean_dim_losses = dim_losses.mean(dim=0)  # Average across batch dimension

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

            # Determine true and predicted bins
            true_bins = torch.argmax(true_class_values[:, start_idx:end_idx], dim=1)
            pred_bins = torch.argmax(task_logits, dim=1)

            # Create table columns
            columns = [
                f"True Reg ({display_name})",
                f"True Bin",
                f"True Bin Start",
                f"True Bin End",
                f"Predicted Bin",
                f"Predicted Bin Start",
                f"Predicted Bin End",
                f"Predicted Reg ({display_name})",
                f"{display_name} Loss",
                f"{display_name} Sample Loss",
            ]

            # Prepare table data
            table_data = []
            for i in range(len(true_reg_values)):
                true_bin = true_bins[i].item()
                pred_bin = pred_bins[i].item()

                # Get bin ranges for true bin (clamp to valid indices)
                true_bin_clamped = max(0, min(true_bin, len(bin_edges_denorm) - 2))
                true_bin_start = bin_edges_denorm[true_bin_clamped].item()
                true_bin_end = bin_edges_denorm[true_bin_clamped + 1].item()

                # Get bin ranges for predicted bin (clamp to valid indices)
                pred_bin_clamped = max(0, min(pred_bin, len(bin_edges_denorm) - 2))
                pred_bin_start = bin_edges_denorm[pred_bin_clamped].item()
                pred_bin_end = bin_edges_denorm[pred_bin_clamped + 1].item()

                # Basic information
                row = [
                    true_reg_values[i, task_idx].item(),
                    true_bin,
                    true_bin_start,
                    true_bin_end,
                    pred_bin,
                    pred_bin_start,
                    pred_bin_end,
                    inverse_preds[i, task_idx].item(),
                    mean_dim_losses[task_idx].item(),  # Mean loss for this dimension
                    dim_losses[i, task_idx].item(),  # Individual sample loss
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

    def _compute_metrics_safely(self, metrics_dict):
        """Safely compute metrics with sufficient samples."""
        results = {}
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

    def forward(self, batch):
        return self.model(batch)

    def _shared_step(self, batch, batch_idx, stage="train"):
        continuous_pred, pooled_features = self(batch)
        batch_size = continuous_pred.size(0)

        # Extract targets
        fitness = batch["gene"].fitness
        gene_interaction = batch["gene"].gene_interaction
        y = torch.cat([fitness, gene_interaction], dim=1)
        y_cont_target_fitness = batch["gene"].fitness_original.view(-1, 1)
        y_cont_target_gene_interaction = batch["gene"].gene_interaction_original.view(
            -1, 1
        )
        y_cont_target = torch.cat(
            [y_cont_target_fitness, y_cont_target_gene_interaction], dim=1
        )

        # Keep gradients when creating HeteroData
        temp_data = HeteroData()
        with torch.set_grad_enabled(True):
            temp_data["gene"].fitness = continuous_pred[:, 0].clone()
            temp_data["gene"].gene_interaction = continuous_pred[:, 1].clone()
            binned_data = self.hparams.forward_transform(temp_data)

        logits = torch.cat(
            [binned_data["gene"].fitness, binned_data["gene"].gene_interaction], dim=1
        ).requires_grad_(True)

        # Get loss and components
        loss, loss_components = self.loss_func(
            continuous_pred=continuous_pred,
            continuous_target=y_cont_target,
            logits=logits,
            categorical_target=y,
            pooled_features=pooled_features,
        )

        # Log all loss components
        self.log(
            f"{stage}/loss",
            loss_components["total_loss"],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/mse_total",
            loss_components["mse_total"],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/entropy_total",
            loss_components["entropy_total"],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/diversity_loss",
            loss_components["diversity_loss"],
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/tightness_loss",
            loss_components["tightness_loss"],
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log per-dimension losses
        dim_losses = loss_components["dim_losses"]
        self.log(
            f"{stage}/fitness_loss",
            (dim_losses[:, 0]).mean(),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/gene_interaction_loss",
            (dim_losses[:, 1]).mean(),
            batch_size=batch_size,
            sync_dist=True,
        )

        # Split logits for classification metrics
        num_classes = self.hparams.bins
        fitness_logits = logits[:, :num_classes]
        gene_int_logits = logits[:, num_classes:]

        # Convert targets for classification metrics
        fitness_targets = torch.argmax(fitness, dim=1)
        gene_int_targets = torch.argmax(gene_interaction, dim=1)

        # Update metrics with NaN masking
        metrics = getattr(self, f"{stage}_metrics")

        # Fitness metrics
        fitness_mask = ~torch.isnan(y_cont_target[:, 0])
        if fitness_mask.any():
            metrics["fitness"](
                fitness_logits[fitness_mask], fitness_targets[fitness_mask]
            )
            metrics["fitness_reg"](
                continuous_pred[fitness_mask, 0], y_cont_target[fitness_mask, 0]
            )

        # Gene interaction metrics
        gi_mask = ~torch.isnan(y_cont_target[:, 1])
        if gi_mask.any():
            metrics["gene_interaction"](
                gene_int_logits[gi_mask], gene_int_targets[gi_mask]
            )
            metrics["gene_interaction_reg"](
                continuous_pred[gi_mask, 1], y_cont_target[gi_mask, 1]
            )

        # Store for validation/test plotting - use continuous predictions directly
        if stage in ["val", "test"]:
            self.true_reg_values.append(y_cont_target.detach())
            self.predictions.append(continuous_pred.detach())

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
            self._log_prediction_table(
                stage=stage,
                true_reg_values=y_cont_target,
                true_class_values=y,
                logits=logits,
                inverse_preds=continuous_pred,
                dim_losses=dim_losses,
            )

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

    def on_train_epoch_end(self):
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

    def on_validation_epoch_end(self):
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        if self.trainer.sanity_checking or (
            self.current_epoch % self.hparams.boxplot_every_n_epochs != 0
        ):
            return

        # Get the stored values
        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Extract values for plotting - use continuous predictions directly
        true_fitness = true_reg_values[:, 0]
        true_gi = true_reg_values[:, 1]
        pred_fitness = predictions[:, 0]
        pred_gi = predictions[:, 1]

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

    def on_test_epoch_end(self):
        for metric_name, metric_dict in self.test_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        if self.trainer.sanity_checking:
            return

        # Get the stored values
        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Extract values for plotting - use continuous predictions directly
        true_fitness = true_reg_values[:, 0]
        true_gi = true_reg_values[:, 1]
        pred_fitness = predictions[:, 0]
        pred_gi = predictions[:, 1]

        # Create box plots - note the test/ prefix
        fig_fitness = fitness.box_plot(true_fitness, pred_fitness)
        wandb.log({"test/fitness_box_plot": wandb.Image(fig_fitness)})
        plt.close(fig_fitness)

        fig_gi = genetic_interaction_score.box_plot(true_gi, pred_gi)
        wandb.log({"test/gene_interaction_box_plot": wandb.Image(fig_gi)})
        plt.close(fig_gi)

        self.true_reg_values = []
        self.predictions = []

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
