import lightning as L
import torch
import torch.nn as nn
import wandb
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
import matplotlib.pyplot as plt
from typing import Optional
import logging
from torchcell.viz.visual_graph_degen import VisGraphDegen
from torchcell.viz import genetic_interaction_score
from torchcell.timestamp import timestamp
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)


class RegressionTask(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        cell_graph: torch.Tensor,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        batch_size: int = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        plot_sample_ceiling: int = 1000,
        plot_every_n_epochs: int = 10,
        loss_func: nn.Module = None,
        grad_accumulation_schedule: Optional[dict[int, int]] = None,
        device: str = "cuda",
        forward_transform: Optional[nn.Module] = None,
        inverse_transform: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.cell_graph = cell_graph
        self.inverse_transform = inverse_transform
        self.current_accumulation_steps = 1
        self.loss_func = loss_func if loss_func else nn.MSELoss()

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )

        # Create metrics for each stage
        for stage in ["train", "val", "test"]:
            metrics_dict = reg_metrics.clone(prefix=f"{stage}/gene_interaction/")
            setattr(self, f"{stage}_metrics", metrics_dict)

            # Add metrics operating in transformed space
            transformed_metrics = reg_metrics.clone(
                prefix=f"{stage}/transformed/gene_interaction/"
            )
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

        # Separate accumulators for train and validation samples
        self.train_samples = {"true_values": [], "predictions": []}
        self.val_samples = {"true_values": [], "predictions": []}
        self.automatic_optimization = False

    def forward(self, batch):
        batch_device = batch["gene"].x.device
        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != batch_device
        ):
            self.cell_graph = self.cell_graph.to(batch_device)
            self._cell_graph_device = batch_device

        # Return all outputs from the model
        return self.model(self.cell_graph, batch)

    def _ensure_no_unused_params_loss(self):
        """Add a dummy loss to ensure all parameters are used in backward pass."""
        dummy_loss = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is None:
                dummy_loss = dummy_loss + 0.0 * param.sum()
        return dummy_loss

    def _shared_step(self, batch, batch_idx, stage="train"):
        # Get model outputs
        predictions, representations = self(batch)

        # Ensure predictions has correct shape (batch_size, 1)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Use transformed values for loss computation
        gene_interaction_vals = batch["gene"].gene_interaction

        # Handle tensor shape
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)

        # Get original values for metrics and visualization
        gene_interaction_orig = (
            batch["gene"].gene_interaction_original
            if "gene_interaction_original" in batch["gene"]
            else gene_interaction_vals
        )

        # Handle tensor shape
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)

        # Get z_p from representations
        z_p = representations.get("z_p")

        # Calculate loss
        if self.loss_func is not None:
            if z_p is not None:
                loss_output = self.loss_func(predictions, gene_interaction_vals, z_p)
            else:
                loss_output = self.loss_func(predictions, gene_interaction_vals)

            # Handle if loss_func returns a tuple
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # First element is the loss
                loss_dict = loss_output[1] if len(loss_output) > 1 else {}

                # Log additional loss components if available
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            self.log(
                                f"{stage}/{key}",
                                value,
                                batch_size=batch_size,
                                sync_dist=True,
                            )
            else:
                loss = loss_output
        else:
            # Default to MSE if no loss function is provided
            loss = nn.MSELoss()(predictions, gene_interaction_vals)

        # Add dummy loss for unused parameters
        dummy_loss = self._ensure_no_unused_params_loss()
        loss = loss + dummy_loss

        # Log the loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)

        # Log z_p norm if available
        if z_p is not None:
            z_p_norm = z_p.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/z_p_norm", z_p_norm, batch_size=batch_size, sync_dist=True
            )

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            # Ensure 1D vectors for metrics update - reshape using view(-1)
            # This is crucial for preventing IndexError
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions
            temp_data = HeteroData()
            temp_data["gene"] = {"gene_interaction": predictions.clone().squeeze()}

            # Apply the inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_gene_int = inv_data["gene"]["gene_interaction"]

            # Handle tensor shape
            if isinstance(inv_gene_int, torch.Tensor):
                if inv_gene_int.dim() == 0:
                    inv_predictions = inv_gene_int.unsqueeze(0).unsqueeze(0)
                elif inv_gene_int.dim() == 1:
                    inv_predictions = inv_gene_int.unsqueeze(1)
                else:
                    inv_predictions = inv_gene_int

        # Update metrics with original scale values
        mask = ~torch.isnan(gene_interaction_orig)
        if mask.sum() > 0:
            metrics = getattr(self, f"{stage}_metrics")
            # Ensure 1D vectors for metrics update - reshape using view(-1)
            metrics.update(
                inv_predictions[mask].view(-1), gene_interaction_orig[mask].view(-1)
            )

        # Collect samples for visualization
        if (
            stage == "train"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            current_count = sum(t.size(0) for t in self.train_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.train_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.train_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )
                    if z_p is not None and "latents" in self.train_samples:
                        self.train_samples["latents"]["z_p"].append(z_p[idx].detach())
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())
                    if z_p is not None and "latents" in self.train_samples:
                        self.train_samples["latents"]["z_p"].append(z_p.detach())
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot
            self.val_samples["true_values"].append(gene_interaction_orig.detach())
            self.val_samples["predictions"].append(inv_predictions.detach())
            if z_p is not None and "latents" in self.val_samples:
                self.val_samples["latents"]["z_p"].append(z_p.detach())

        return loss, predictions, gene_interaction_orig

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

    def _compute_metrics_safely(self, metrics_dict):
        results = {}
        for metric_name, metric in metrics_dict.items():
            try:
                results[metric_name] = metric.compute()
            except ValueError as e:
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

    def _plot_samples(self, samples, stage: str) -> None:
        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)

        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]

        # Log scatter plot
        mask = ~torch.isnan(true_values)
        true_vals_clean = true_values[mask].squeeze().cpu().numpy()
        pred_vals_clean = predictions[mask].squeeze().cpu().numpy()

        if len(true_vals_clean) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(true_vals_clean, pred_vals_clean, alpha=0.5)
            ax.set_xlabel("True Gene Interaction")
            ax.set_ylabel("Predicted Gene Interaction")
            ax.set_title(
                f"Gene Interaction Predictions ({stage}, Epoch {self.current_epoch})"
            )

            # Add identity line
            min_val = min(true_vals_clean.min(), pred_vals_clean.min())
            max_val = max(true_vals_clean.max(), pred_vals_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--")

            # Calculate and display correlation
            if len(true_vals_clean) > 1:
                correlation = np.corrcoef(true_vals_clean, pred_vals_clean)[0, 1]
                ax.text(
                    0.05,
                    0.95,
                    f"Correlation: {correlation:.4f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                )

            wandb.log({f"{stage}_scatter_plot": wandb.Image(fig)})
            plt.close(fig)

        # Log genetic interaction box plot
        if torch.any(~torch.isnan(true_values)):
            fig_gi = genetic_interaction_score.box_plot(
                true_values.squeeze().cpu(), predictions.squeeze().cpu()
            )
            wandb.log({f"{stage}/gene_interaction_box_plot": wandb.Image(fig_gi)})
            plt.close(fig_gi)

    def on_train_epoch_end(self):
        # Log training metrics
        computed_metrics = self._compute_metrics_safely(self.train_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.train_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_transformed_metrics.reset()

        # Plot training samples
        if (
            self.current_epoch + 1
        ) % self.hparams.plot_every_n_epochs == 0 and self.train_samples["true_values"]:
            self._plot_samples(self.train_samples, "train_sample")
            # Reset the sample containers
            self.train_samples = {"true_values": [], "predictions": []}

    def on_train_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": []}

    def on_validation_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.val_samples = {"true_values": [], "predictions": []}

    def on_validation_epoch_end(self):
        # Log validation metrics
        computed_metrics = self._compute_metrics_safely(self.val_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(self.val_transformed_metrics)
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_transformed_metrics.reset()

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            # Reset the sample containers
            self.val_samples = {"true_values": [], "predictions": []}

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
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
                "monitor": "val/gene_interaction/MSE",
                "interval": "epoch",
                "frequency": 1,
            },
        }
