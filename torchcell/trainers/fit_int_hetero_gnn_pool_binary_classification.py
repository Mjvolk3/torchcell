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
    NaNTolerantPrecision,
    NaNTolerantRecall,
)
from torchcell.metrics.nan_tolerant_metrics import (
    NaNTolerantMSE,
    NaNTolerantRMSE,
    NaNTolerantR2Score,
    NaNTolerantPearsonCorrCoef,
)
from torchcell.viz import fitness, genetic_interaction_score
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

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
        class_metrics = MetricCollection(
            {
                "Accuracy": NaNTolerantAccuracy(task="binary"),
                "F1": NaNTolerantF1Score(task="binary"),
                "Precision": NaNTolerantPrecision(task="binary"),
                "Recall": NaNTolerantRecall(task="binary"),
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
    ):
        """Log a wandb table with all prediction forms for comparison."""
        num_bins = logits.shape[1] // 2  # Number of bins per target

        # Convert logits to softmax separately for each label
        fitness_softmax = torch.softmax(logits[:, :num_bins], dim=1)
        gi_softmax = torch.softmax(logits[:, num_bins:], dim=1)

        # Create dynamic column headers
        columns = []
        # Add regression truth columns
        columns.extend(["True Reg (Fitness)", "True Reg (GI)"])

        # Add classification truth columns - grouped by target
        columns.extend([f"True Class Fitness (Bin {i})" for i in range(num_bins)])
        columns.extend([f"True Class GI (Bin {i})" for i in range(num_bins)])

        # Add logits columns - grouped by target
        columns.extend([f"Logits Fitness (Bin {i})" for i in range(num_bins)])
        columns.extend([f"Logits GI (Bin {i})" for i in range(num_bins)])

        # Add softmax columns - grouped by target
        columns.extend([f"Softmax Fitness (Bin {i})" for i in range(num_bins)])
        columns.extend([f"Softmax GI (Bin {i})" for i in range(num_bins)])

        # Add regression prediction columns
        columns.extend(["Pred Reg (Fitness)", "Pred Reg (GI)"])

        # Create table data
        data = []
        for i in range(len(true_reg_values)):
            row = []
            # Add true regression values
            row.extend([true_reg_values[i, 0].item(), true_reg_values[i, 1].item()])

            # Add true class values for fitness
            row.extend([true_class_values[i, j].item() for j in range(num_bins)])
            # Add true class values for GI
            row.extend(
                [true_class_values[i, num_bins + j].item() for j in range(num_bins)]
            )

            # Add logits for fitness
            row.extend([logits[i, j].item() for j in range(num_bins)])
            # Add logits for GI
            row.extend([logits[i, num_bins + j].item() for j in range(num_bins)])

            # Add softmax values for fitness
            row.extend([fitness_softmax[i, j].item() for j in range(num_bins)])
            # Add softmax values for GI
            row.extend([gi_softmax[i, j].item() for j in range(num_bins)])

            # Add predicted regression values
            row.extend([inverse_preds[i, 0].item(), inverse_preds[i, 1].item()])

            data.append(row)

        # Create and log wandb table
        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"{stage}/second_to_last_batch_predictions": table}, commit=False)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        return self.model(x_dict, edge_index_dict, batch_dict)

    def _shared_step(self, batch, batch_idx, stage="train"):
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

        # Extract targets
        fitness = batch["gene"].fitness
        gene_interaction = batch["gene"].gene_interaction
        y = torch.cat([fitness, gene_interaction], dim=1)

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

        # Split logits for each task
        fitness_logits = logits[:, :2]
        gene_int_logits = logits[:, 2:]

        # Update classification metrics
        metrics = getattr(self, f"{stage}_metrics")
        metrics["fitness"](fitness_logits, fitness)
        metrics["gene_interaction"](gene_int_logits, gene_interaction)

        # For regression metrics, transform predictions back to regression space
        # Use detached tensors for metrics
        pred_data = HeteroData()
        pred_data["gene"] = {
            "fitness": fitness_logits.detach(),  # Detach here
            "gene_interaction": gene_int_logits.detach(),  # Detach here
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

        if batch_idx == num_batches - 2:  # Second to last batch
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
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

    def on_validation_epoch_end(self):
        # Log metrics
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        if self.trainer.sanity_checking or (
            self.current_epoch % self.hparams.boxplot_every_n_epochs != 0
        ):
            return

        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Create HeteroData object for predictions
        pred_data = HeteroData()
        pred_data["gene"] = {
            "fitness": torch.softmax(predictions[:, :2], dim=1),
            "gene_interaction": torch.softmax(predictions[:, 2:], dim=1),
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

            fig_gi = genetic_interaction_score.box_plot(true_gi, pred_gi)
            wandb.log({"val/gene_interaction_box_plot": wandb.Image(fig_gi)})

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
        # Log metrics
        for metric_name, metric_dict in self.test_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        if self.trainer.sanity_checking:
            return

        true_reg_values = torch.cat(self.true_reg_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)

        # Create HeteroData object for predictions
        pred_data = HeteroData()
        pred_data["gene"] = {
            "fitness": torch.softmax(predictions[:, :2], dim=1),
            "gene_interaction": torch.softmax(predictions[:, 2:], dim=1),
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
