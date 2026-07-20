"""Lightning regression trainer for the SAGPool cell-graph interaction model."""

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
from torchcell.metrics.nan_tolerant_metrics import NaNTolerantMSE
from torchcell.viz import fitness, genetic_interaction_score

log = logging.getLogger(__name__)

style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def log_error_information(
    batch_idx: int,
    y: torch.Tensor,
    y_hat: torch.Tensor,
    x: torch.Tensor,
    edge_indices: dict[str, torch.Tensor],
    batch: torch.Tensor,
    head_loss: torch.Tensor,
    dim_losses: torch.Tensor,
    graph_attention_weights: dict[str, Any],
    graph_pool_scores: dict[str, Any],
    graph_intermediate_predictions: dict[str, Any],
    graph_pool_sizes: dict[str, Any],
    graph_node_selections: dict[str, Any],
) -> None:
    """Log batch tensors and per-graph diagnostics when a NaN loss occurs."""
    log.error("NaN loss detected. Logging relevant information and terminating.")
    log.error(f"Batch index: {batch_idx}")
    log.error(f"y: {y}")
    log.error(f"y_hat: {y_hat}")
    log.error(f"x: {x}")
    log.error(f"edge_indices: {edge_indices}")
    log.error(f"batch: {batch}")
    log.error(f"head_loss: {head_loss}")
    log.error(f"dim_losses: {dim_losses}")

    # Log graph-specific information
    for graph_name in graph_attention_weights.keys():
        log.error(f"\n{graph_name} graph information:")
        log.error(f"Attention weights: {graph_attention_weights[graph_name]}")
        log.error(f"Pool scores: {graph_pool_scores[graph_name]}")
        log.error(
            f"Intermediate predictions: {graph_intermediate_predictions[graph_name]}"
        )
        log.error(f"Pool sizes: {graph_pool_sizes[graph_name]}")
        log.error(f"Node selections: {graph_node_selections[graph_name]}")

    # Write to file
    with open("nan_loss_debug.log", "w") as f:
        f.write(f"Batch index: {batch_idx}\n")
        f.write(f"y: {y}\n")
        f.write(f"y_hat: {y_hat}\n")
        f.write(f"x: {x}\n")
        f.write(f"edge_indices: {edge_indices}\n")
        f.write(f"batch: {batch}\n")
        f.write(f"head_loss: {head_loss}\n")
        f.write(f"dim_losses: {dim_losses}\n")

        for graph_name in graph_attention_weights.keys():
            f.write(f"\n{graph_name} graph information:\n")
            f.write(f"Attention weights: {graph_attention_weights[graph_name]}\n")
            f.write(f"Pool scores: {graph_pool_scores[graph_name]}\n")
            f.write(
                f"Intermediate predictions: {graph_intermediate_predictions[graph_name]}\n"
            )
            f.write(f"Pool sizes: {graph_pool_sizes[graph_name]}\n")
            f.write(f"Node selections: {graph_node_selections[graph_name]}\n")


class RegressionTask(L.LightningModule):
    """Lightning task training a SAGPool model on fitness and interaction labels."""

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
        intermediate_loss_weight: float = 0.1,
        grad_accumulation_schedule: dict[int, int] | None = None,
        device: str = "cuda",
    ) -> None:
        """Store the model, loss, optimizer config, and per-stage metrics."""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.combined_loss = loss_func
        self.current_accumulation_steps = 1

        metrics = MetricCollection(
            {
                # "RMSE": NaNTolerantRMSE(),
                "MSE": NaNTolerantMSE()
                # "MAE": NaNTolerantMAE(),
                # "PearsonR": NaNTolerantPearsonCorrCoef(),
                # # "SpearmanR": NaNTolerantSpearmanCorrCoef(),
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

        self.true_values: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.last_logged_best_step: int | None = None
        self.automatic_optimization = False

    # return: model output tuple, dynamic from nn.Module
    def forward(
        self,
        x: torch.Tensor,
        edge_indices: dict[str, torch.Tensor],
        batch: torch.Tensor,
    ) -> Any:
        """Run the wrapped model on node features, edges, and batch index."""
        return self.model(x, edge_indices, batch)

    def _shared_step(
        self, batch: HeteroData, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass, compute losses and metrics for one batch."""
        # Process input data for sparse format
        x = batch["gene"].x
        edge_indices = {
            "physical": batch["gene", "physical_interaction", "gene"].edge_index,
            "regulatory": batch["gene", "regulatory_interaction", "gene"].edge_index,
        }
        batch_index = batch["gene"].batch
        y = torch.stack(
            [
                batch["gene"].fitness.squeeze(-1),
                batch["gene"].gene_interaction.squeeze(-1),
            ],
            dim=1,
        )

        # Forward pass with sparse model - now matches CellSAGPool's return values
        (
            final_output,
            graph_outputs,
            graph_attention_weights,
            graph_pool_scores,
            graph_intermediate_predictions,
            graph_pool_sizes,
            graph_node_selections,
        ) = self(x, edge_indices, batch_index)

        # Compute main loss (loss_func is set before training begins)
        assert self.combined_loss is not None
        head_loss, dim_losses = self.combined_loss(final_output, y)

        # Compute individual graph losses
        graph_losses = sum(
            self.combined_loss(pred, y)[0] for pred in graph_outputs.values()
        )

        # Compute intermediate prediction losses
        intermediate_losses = sum(
            self.combined_loss(pred, y)[0]
            for graph_preds in graph_intermediate_predictions.values()
            for pred in graph_preds
        )

        # Total loss with weights
        loss = (
            head_loss
            + self.hparams["intermediate_loss_weight"] * graph_losses
            + self.hparams["intermediate_loss_weight"] * intermediate_losses
        )

        if torch.isnan(loss):
            log_error_information(
                batch_idx,
                y,
                final_output,
                x,
                edge_indices,
                batch_index,
                head_loss,
                dim_losses,
                graph_attention_weights,
                graph_pool_scores,
                graph_intermediate_predictions,
                graph_pool_sizes,
                graph_node_selections,
            )
            sys.exit(1)

        # Batch size for logging
        batch_size = x.size(0)

        # Logging
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)  # OK
        self.log(
            f"{stage}/head_loss", head_loss, batch_size=batch_size, sync_dist=True
        )  # OK
        self.log(
            f"{stage}/intermediate_loss",
            intermediate_losses,
            batch_size=batch_size,
            sync_dist=True,
        )  # OK
        self.log(
            f"{stage}/graph_losses", graph_losses, batch_size=batch_size, sync_dist=True
        )  # OK
        self.log(
            f"{stage}/fitness_loss",
            dim_losses[0],
            batch_size=batch_size,
            sync_dist=True,
        )  # OK
        self.log(
            f"{stage}/gene_interaction_loss",
            dim_losses[1],
            batch_size=batch_size,
            sync_dist=True,
        )  # OK

        # Log pooling information
        device = next(self.parameters()).device  # or size.device
        for graph_name, pool_sizes in graph_pool_sizes.items():
            for layer_idx, size in enumerate(pool_sizes):
                mean_size = torch.mean(size.to(dtype=torch.float32, device=device))
                self.log(
                    f"{stage}/{graph_name}/layer_{layer_idx}_pool_size",
                    mean_size,
                    batch_size=batch_size,
                    sync_dist=True,
                )  # OK after device

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
        """Run the shared step for one validation batch and return the loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run the shared step for one test batch and return the loss."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def on_train_epoch_end(self) -> None:
        """Compute, log, and reset the training metrics at epoch end."""
        # Compute and log metrics for each metric type
        for metric_name, metric_module in self.train_metrics.items():
            metric_dict = cast(MetricCollection, metric_module)
            computed_metrics = metric_dict.compute()
            for name, value in computed_metrics.items():
                self.log(f"train/{metric_name}/{name}", value, sync_dist=True)
            metric_dict.reset()  # Reset metrics after logging

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and emit prediction plots at epoch end."""
        # Compute and log metrics for each metric type
        for metric_name, metric_module in self.val_metrics.items():
            metric_dict = cast(MetricCollection, metric_module)
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
        """Build and log box plots of predictions binned by true value."""
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
        """Log test metrics and emit prediction box plots at epoch end."""
        # ModuleDict.compute/reset shadowed by Tensor.compute in torch stubs
        self.log_dict(self.test_metrics.compute(), sync_dist=True)  # type: ignore[operator]
        self.test_metrics.reset()  # type: ignore[operator]

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
        """Build the optimizer and LR scheduler from the hyperparameter config."""
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
