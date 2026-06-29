"""Lightning regression task for the DiffPool dense cell-integration model."""

# torchcell/trainers/fit_int_cell_diffpool_dense_regression
# [[torchcell.trainers.fit_int_cell_diffpool_dense_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/fit_int_cell_diffpool_dense_regression
# Test file: tests/torchcell/trainers/test_fit_int_cell_diffpool_dense_regression.py

import logging
import os.path as osp
import sys
from typing import Any, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import MetricCollection

import torchcell
from torchcell.metrics.nan_tolerant_metrics import (
    NaNTolerantMAE,
    NaNTolerantMSE,
    NaNTolerantPearsonCorrCoef,
    NaNTolerantRMSE,
    NaNTolerantSpearmanCorrCoef,
)
from torchcell.viz import fitness, genetic_interaction_score

log = logging.getLogger(__name__)

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def log_error_information(
    batch_idx: int,
    y: torch.Tensor,
    y_hat: torch.Tensor,
    x: torch.Tensor,
    adj_dict: dict[str, torch.Tensor],
    mask: torch.Tensor,
    head_loss: torch.Tensor,
    dim_losses: torch.Tensor,
    graph_link_losses: Any,  # dynamic per-graph loss structures, logging only
    graph_entropy_losses: Any,  # dynamic per-graph loss structures, logging only
    graph_pool_attention: Any,  # dynamic per-graph attention tensors, logging only
    graph_cluster_assignments: Any,  # dynamic per-graph assignments, logging only
) -> None:
    """Log and persist batch tensors and loss components when a NaN loss occurs."""
    log.error("NaN loss detected. Logging relevant information and terminating.")
    log.error(f"Batch index: {batch_idx}")
    log.error(f"y: {y}")
    log.error(f"y_hat: {y_hat}")
    log.error(f"x: {x}")
    log.error(f"adj_dict: {adj_dict}")
    log.error(f"mask: {mask}")
    log.error(f"head_loss: {head_loss}")
    log.error(f"dim_losses: {dim_losses}")
    log.error(f"graph_link_losses: {graph_link_losses}")
    log.error(f"graph_entropy_losses: {graph_entropy_losses}")
    log.error(f"graph_pool_attention: {graph_pool_attention}")
    log.error(f"graph_cluster_assignments: {graph_cluster_assignments}")

    with open("nan_loss_debug.log", "w") as f:
        f.write(f"Batch index: {batch_idx}\n")
        f.write(f"y: {y}\n")
        f.write(f"y_hat: {y_hat}\n")
        f.write(f"x: {x}\n")
        f.write(f"adj_dict: {adj_dict}\n")
        f.write(f"mask: {mask}\n")
        f.write(f"head_loss: {head_loss}\n")
        f.write(f"dim_losses: {dim_losses}\n")
        f.write(f"graph_link_losses: {graph_link_losses}\n")
        f.write(f"graph_entropy_losses: {graph_entropy_losses}\n")
        f.write(f"graph_pool_attention: {graph_pool_attention}\n")
        f.write(f"graph_cluster_assignments: {graph_cluster_assignments}\n")


class RegressionTask(L.LightningModule):
    """Lightning module training the model for fitness and gene-interaction regression."""

    # Metric collections are stored as ModuleDicts of MetricCollections at runtime;
    # declare them as MetricCollection so mypy resolves .items()/.compute()/.reset().
    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection

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
        cluster_loss_weight: float = 1.0,
        link_pred_loss_weight: float = 1.0,
        entropy_loss_weight: float = 1.0,
        grad_accumulation_schedule: dict[int, int] | None = None,
    ):
        """Store the model and loss, build per-target metric collections, and set hparams."""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        # loss_func is always supplied at runtime (used unconditionally below);
        # the None default is only an implicit-Optional artifact.
        self.combined_loss = cast(nn.Module, loss_func)
        self.current_accumulation_steps = 1

        metrics = MetricCollection(
            {
                "RMSE": NaNTolerantRMSE(),
                "MSE": NaNTolerantMSE(),
                "MAE": NaNTolerantMAE(),
                "PearsonR": NaNTolerantPearsonCorrCoef(),
                "SpearmanR": NaNTolerantSpearmanCorrCoef(),
            }
        )

        self.train_metrics = cast(
            MetricCollection,
            nn.ModuleDict(
                {
                    "fitness": metrics.clone(prefix="train/fitness/"),
                    "gene_interaction": metrics.clone(prefix="train/gene_interaction/"),
                }
            ),
        )
        self.val_metrics = cast(
            MetricCollection,
            nn.ModuleDict(
                {
                    "fitness": metrics.clone(prefix="val/fitness/"),
                    "gene_interaction": metrics.clone(prefix="val/gene_interaction/"),
                }
            ),
        )
        self.test_metrics = cast(
            MetricCollection,
            nn.ModuleDict(
                {
                    "fitness": metrics.clone(prefix="test/fitness/"),
                    "gene_interaction": metrics.clone(prefix="test/gene_interaction/"),
                }
            ),
        )

        self.true_values: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.last_logged_best_step: int | None = None
        self.automatic_optimization = False

    def setup(self, stage: str | None = None) -> None:
        """Move the model and loss weights onto the active device."""
        self.model = self.model.to(self.device)
        self.combined_loss.weights = self.combined_loss.weights.to(self.device)

    def update_accumulation_steps(self, epoch: int) -> None:
        """Set gradient accumulation steps from the schedule for the given epoch."""
        if self.hparams["grad_accumulation_schedule"] is not None:
            self.current_accumulation_steps = max(
                [
                    steps
                    for e, steps in self.hparams["grad_accumulation_schedule"].items()
                    if int(e) <= epoch
                ],
                default=1,
            )

    def forward(
        self, x: torch.Tensor, adj_dict: dict[str, torch.Tensor], mask: torch.Tensor
    ) -> Any:  # model returns a dynamic multi-output tuple
        """Run the model on dense node features, adjacency, and mask."""
        return self.model(x, adj_dict, mask)

    def on_train_start(self) -> None:
        """Log parameter count and initialize gradient accumulation at train start."""
        parameter_size = sum(p.numel() for p in self.parameters())
        self.log("model/parameters_size", float(parameter_size), on_epoch=True)
        self.update_accumulation_steps(self.current_epoch)

    def _shared_step(
        self, batch: HeteroData, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Process input data for dense format
        x = batch["gene"].x
        adj_dict = {
            "physical_interaction": batch["gene", "physical_interaction", "gene"].adj,
            "regulatory_interaction": batch[
                "gene", "regulatory_interaction", "gene"
            ].adj,
        }
        mask = batch["gene"].mask
        # Fix the dimension issue by squeezing the last dimension
        y = torch.stack(
            [
                batch["gene"].fitness.squeeze(-1),
                batch["gene"].gene_interaction.squeeze(-1),
            ],
            dim=1,
        )

        # Forward pass with dense model
        (
            final_output,
            graph_link_losses,
            graph_entropy_losses,
            graph_pool_attention,
            graph_embed_attention,
            graph_cluster_assignments,
            graph_cluster_outputs,
            individual_predictions,
        ) = self(x, adj_dict, mask)

        # Compute losses
        head_loss, dim_losses = self.combined_loss(final_output, y)

        # Compute cluster losses for each graph
        cluster_losses = {}
        total_cluster_loss = 0
        for graph_name, clusters in graph_cluster_outputs.items():
            cluster_predictions_tensor = torch.stack(clusters)
            expanded_y = y.unsqueeze(0).expand_as(cluster_predictions_tensor)
            cluster_loss, _ = self.combined_loss(cluster_predictions_tensor, expanded_y)
            cluster_losses[graph_name] = cluster_loss / len(clusters)
            total_cluster_loss += cluster_loss

        # Weighted losses
        total_cluster_loss = total_cluster_loss * self.hparams["cluster_loss_weight"]
        total_link_loss = (
            sum(sum(losses) for losses in graph_link_losses.values())
            * self.hparams["link_pred_loss_weight"]
        )
        total_entropy_loss = (
            sum(sum(losses) for losses in graph_entropy_losses.values())
            * self.hparams["entropy_loss_weight"]
        )

        # Total loss
        loss = head_loss + total_cluster_loss + total_link_loss + total_entropy_loss

        if torch.isnan(loss):
            log_error_information(
                batch_idx,
                y,
                final_output,
                x,
                adj_dict,
                mask,
                head_loss,
                dim_losses,
                graph_link_losses,
                graph_entropy_losses,
                graph_pool_attention,
                graph_cluster_assignments,
            )
            sys.exit(1)

        # Batch size for logging
        batch_size = x.size(0)

        # Logging
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)
        self.log(f"{stage}/head_loss", head_loss, batch_size=batch_size, sync_dist=True)
        self.log(
            f"{stage}/cluster_loss",
            total_cluster_loss,
            batch_size=batch_size,
            sync_dist=True,
        )
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
        self.log(
            f"{stage}/link_pred_loss",
            total_link_loss,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{stage}/entropy_loss",
            total_entropy_loss,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Update metrics
        metrics = getattr(self, f"{stage}_metrics")
        metrics["fitness"](final_output[:, 0], y[:, 0])
        metrics["gene_interaction"](final_output[:, 1], y[:, 1])

        if stage in ["val", "test"]:
            self.true_values.append(y.detach())
            self.predictions.append(final_output.detach())

        return loss, final_output, y

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a manual-optimization training step with gradient accumulation."""
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Scale loss by accumulation steps
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

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a validation step and return its loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a test step and return its loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def on_train_epoch_end(self) -> None:
        """Compute, log, and reset training metrics at epoch end."""
        # Compute and log metrics for each metric type
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"train/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()  # Reset metrics after logging

    def on_validation_epoch_end(self) -> None:
        """Compute, log, and reset validation metrics at epoch end."""
        # Compute and log metrics for each metric type
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"val/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()  # Reset metrics after logging

        # Skip plotting during sanity check
        if self.trainer.sanity_checking or (
            self.current_epoch % self.hparams["boxplot_every_n_epochs"] != 0
        ):
            return

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="val")

        # Create and log box plots for both fitness and gene interaction
        fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
        wandb.log({"fitness_binned_values_box_plot": wandb.Image(fig_fitness)})
        plt.close(fig_fitness)

        fig_gi = genetic_interaction_score.box_plot(
            true_values[:, 1], predictions[:, 1]
        )
        wandb.log({"gene_interaction_binned_values_box_plot": wandb.Image(fig_gi)})
        plt.close(fig_gi)

        # Clear the stored values for the next epoch
        self.true_values = []
        self.predictions = []

        # Logging model artifact
        current_global_step = self.global_step
        ckpt = cast(ModelCheckpoint, self.trainer.checkpoint_callback)
        assert ckpt is not None
        best_model_path = ckpt.best_model_path
        if best_model_path and current_global_step != self.last_logged_best_step:
            # Save model as a W&B artifact
            artifact = wandb.Artifact(
                name=f"model-global_step-{current_global_step}",
                type="model",
                description=f"Model on validation epoch end step - {current_global_step}",
                metadata=dict(self.hparams),
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            self.last_logged_best_step = (
                current_global_step  # update the last logged step
            )

    def compute_prediction_stats(
        self, true_values: torch.Tensor, predictions: torch.Tensor, stage: str = "val"
    ) -> None:
        """Bin predictions per target and log per-bin mean/std tables to wandb."""
        # Define the bin edges for each dimension
        bin_edges = {
            "fitness": torch.tensor(
                [-float("inf"), 0] + [i * 0.1 for i in range(1, 13)] + [float("inf")]
            ),
            "gene_interaction": torch.tensor(
                [-float("inf"), -0.2, -0.1, 0, 0.1, 0.2, float("inf")]
            ),
        }

        for dim, dim_name in enumerate(["fitness", "gene_interaction"]):
            # Prepare a table with columns for the range, mean, and standard deviation
            wandb_table = wandb.Table(columns=["Range", "Mean", "StdDev"])

            dim_pred = predictions[:, dim]

            # Calculate mean and std for each bin and add to the table
            for i in range(len(bin_edges[dim_name]) - 1):
                bin_mask = (dim_pred >= bin_edges[dim_name][i]) & (
                    dim_pred < bin_edges[dim_name][i + 1]
                )
                if bin_mask.any():
                    bin_predictions = dim_pred[bin_mask]
                    mean_val = bin_predictions.mean().item()
                    std_val = bin_predictions.std().item()
                    if i == len(bin_edges[dim_name]) - 2:
                        range_str = f"{bin_edges[dim_name][i].item():.2f} - inf"
                    else:
                        range_str = f"{bin_edges[dim_name][i].item():.2f} - {bin_edges[dim_name][i + 1].item():.2f}"
                    wandb_table.add_data(range_str, mean_val, std_val)

            # Log the table to wandb
            wandb.log(
                {
                    f"{stage}/{dim_name}_Prediction_Stats_{self.current_epoch}": wandb_table
                }
            )

    def on_test_epoch_end(self) -> None:
        """Log test metrics, prediction stats, and box plots, then clear buffers."""
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

        # Convert lists to tensors
        true_values = torch.cat(self.true_values, dim=0)
        predictions = torch.cat(self.predictions, dim=0)
        self.compute_prediction_stats(true_values, predictions, stage="test")

        # Create and log box plots for both fitness and gene interaction
        fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
        wandb.log({"test_fitness_binned_values_box_plot": wandb.Image(fig_fitness)})
        plt.close(fig_fitness)

        fig_gi = genetic_interaction_score.box_plot(
            true_values[:, 1], predictions[:, 1]
        )
        wandb.log({"test_gene_interaction_binned_values_box_plot": wandb.Image(fig_gi)})
        plt.close(fig_gi)

        # Clear the stored values
        self.true_values = []
        self.predictions = []

    def configure_optimizers(self) -> Any:  # Lightning accepts optimizer or config dict
        """Build the optimizer and ReduceLROnPlateau scheduler from hparams."""
        optimizer_class = getattr(optim, self.hparams["optimizer_config"]["type"])
        optimizer_params = {
            k: v for k, v in self.hparams["optimizer_config"].items() if k != "type"
        }

        # Replace 'learning_rate' with 'lr' if present
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # Remove 'type' from lr_scheduler_config before passing to ReduceLROnPlateau
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
