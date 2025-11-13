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
from torchcell.viz.visual_regression import Visualization
from torchcell.timestamp import timestamp
from torch_geometric.data import HeteroData
from torchcell.losses.logcosh import LogCoshLoss
from torchcell.losses.diffusion_loss import DiffusionLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR

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
        inverse_transform: Optional[nn.Module] = None,
        execution_mode: str = "training",  # "training" or "dataloader_profiling"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.execution_mode = execution_mode
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func

        # Initialize gradient accumulation
        self.current_accumulation_steps = 1
        if self.hparams.grad_accumulation_schedule is not None:
            # Get the accumulation steps for epoch 0
            self.current_accumulation_steps = (
                self.hparams.grad_accumulation_schedule.get(0, 1)
            )

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

        # Separate accumulators for train, validation, and test samples
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}
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
        # DataLoader profiling mode: Skip model forward, create dummy loss
        if self.execution_mode == "dataloader_profiling":
            # Execute all batch preparation (moving to device happens in forward())
            batch_device = batch["gene"].x.device
            # Ensure cell_graph is on correct device
            if not hasattr(self, "_cell_graph_device") or self._cell_graph_device != batch_device:
                self.cell_graph = self.cell_graph.to(batch_device)
                self._cell_graph_device = batch_device

            # Create trivial loss that touches ALL model parameters (required for DDP)
            # This ensures no "unused parameters" error in DDP mode
            loss = torch.zeros((), device=batch_device, requires_grad=True)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss = loss + (param * 0.0).sum()

            # Log minimal metrics
            batch_size = batch["gene"].x.size(0)
            self.log(f"{stage}/dataloader_profile_loss", loss, batch_size=batch_size, sync_dist=True)
            self.log(f"{stage}/dataloader_profile_batch_size", float(batch_size), batch_size=batch_size, sync_dist=True)

            return loss, None, None

        # Normal training/validation/test execution
        # Get model outputs
        predictions, representations = self(batch)

        # Ensure predictions has correct shape (batch_size, 1)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Get target values - now in COO format
        # For gene interaction dataset, phenotype_values directly contains the values
        gene_interaction_vals = batch["gene"].phenotype_values

        # Handle tensor shape
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)

        # For original values, check if there's a phenotype_values_original
        if hasattr(batch["gene"], "phenotype_values_original"):
            gene_interaction_orig = batch["gene"].phenotype_values_original
        else:
            gene_interaction_orig = gene_interaction_vals

        # Handle tensor shape
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)

        # Get z_p from representations
        z_p = representations.get("z_p")

        # Calculate loss based on loss function type
        if self.loss_func is None:
            raise ValueError("No loss function provided")

        if isinstance(self.loss_func, LogCoshLoss):
            # For LogCoshLoss, just pass predictions and targets
            loss = self.loss_func(predictions, gene_interaction_vals)
        else:
            # For ICLoss or other custom losses that might use z_p
            # Check if loss function accepts epoch parameter (for MleDistSupCR and MleWassSupCR)
            if z_p is not None:
                if isinstance(self.loss_func, (MleDistSupCR, MleWassSupCR)):
                    loss_output = self.loss_func(
                        predictions,
                        gene_interaction_vals,
                        z_p,
                        epoch=self.current_epoch,
                    )
                else:
                    loss_output = self.loss_func(
                        predictions, gene_interaction_vals, z_p
                    )
            else:
                loss_output = self.loss_func(predictions, gene_interaction_vals)

            # Handle if loss_func returns a tuple (for ICLoss)
            if isinstance(loss_output, tuple):
                loss = loss_output[0]  # First element is the loss
                loss_dict = loss_output[1] if len(loss_output) > 1 else {}

                # Log additional loss components if available
                if isinstance(loss_dict, dict):
                    for key, value in loss_dict.items():
                        if isinstance(value, torch.Tensor):
                            # Handle multi-dimensional tensors
                            if value.numel() == 1:
                                # Single element tensor - log as scalar
                                self.log(
                                    f"{stage}/{key}",
                                    value.item(),
                                    batch_size=batch_size,
                                    sync_dist=True,
                                )
                            elif value.numel() > 1:
                                # Multi-element tensor - log each element separately
                                for i in range(value.numel()):
                                    self.log(
                                        f"{stage}/{key}_{i}",
                                        value[i].item(),
                                        batch_size=batch_size,
                                        sync_dist=True,
                                    )
                            # Skip empty tensors
                        elif isinstance(value, (int, float)):
                            # Handle scalar values
                            self.log(
                                f"{stage}/{key}",
                                value,
                                batch_size=batch_size,
                                sync_dist=True,
                            )
            else:
                loss = loss_output

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

        # Log gate weights if available
        if "gate_weights" in representations:
            gate_weights = representations["gate_weights"]
            # Average gate weights across batch
            avg_gate_weights = gate_weights.mean(dim=0)
            self.log(
                f"{stage}/gate_weight_global",
                avg_gate_weights[0],
                batch_size=batch_size,
                sync_dist=True,
            )
            # Only log local weight if it exists (when local predictor is enabled)
            if avg_gate_weights.size(0) > 1:
                self.log(
                    f"{stage}/gate_weight_local",
                    avg_gate_weights[1],
                    batch_size=batch_size,
                    sync_dist=True,
                )

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions in COO format
            temp_data = HeteroData()

            # Create COO format data for predictions
            batch_size = predictions.size(0)
            device = predictions.device
            temp_data["gene"].phenotype_values = predictions.squeeze()
            temp_data["gene"].phenotype_type_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            temp_data["gene"].phenotype_sample_indices = torch.arange(
                batch_size, device=device
            )
            temp_data["gene"].phenotype_types = ["gene_interaction"]

            # Apply the inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_gene_int = inv_data["gene"]["phenotype_values"]

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
                    if z_p is not None:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "z_p" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["z_p"] = []
                        self.train_samples["latents"]["z_p"].append(z_p[idx].detach())
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())
                    if z_p is not None:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "z_p" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["z_p"] = []
                        self.train_samples["latents"]["z_p"].append(z_p.detach())
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot, respecting ceiling
            current_count = sum(t.size(0) for t in self.val_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.val_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions[idx].detach())
                    if z_p is not None:
                        if "latents" not in self.val_samples:
                            self.val_samples["latents"] = {}
                        if "z_p" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["z_p"] = []
                        self.val_samples["latents"]["z_p"].append(z_p[idx].detach())
                else:
                    self.val_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions.detach())
                    if z_p is not None:
                        if "latents" not in self.val_samples:
                            self.val_samples["latents"] = {}
                        if "z_p" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["z_p"] = []
                        self.val_samples["latents"]["z_p"].append(z_p.detach())
        elif stage == "test":
            # For test, always collect samples (no epoch check since test runs once)
            self.test_samples["true_values"].append(gene_interaction_orig.detach())
            self.test_samples["predictions"].append(inv_predictions.detach())
            if z_p is not None:
                if "latents" not in self.test_samples:
                    self.test_samples["latents"] = {}
                if "z_p" not in self.test_samples["latents"]:
                    self.test_samples["latents"]["z_p"] = []
                self.test_samples["latents"]["z_p"].append(z_p.detach())

        return loss, predictions, gene_interaction_orig

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Model profiling mode: Skip optimizer step to isolate model compute
        if self.execution_mode == "model_profiling":
            return loss

        # Normal training: Run optimizer
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
        # Log effective batch size when using gradient accumulation
        if self.hparams.grad_accumulation_schedule is not None:
            # Get world size for DDP
            world_size = 1
            if hasattr(self.trainer, "strategy") and hasattr(
                self.trainer.strategy, "_strategy_name"
            ):
                if self.trainer.strategy._strategy_name == "ddp":
                    import torch.distributed as dist

                    if dist.is_initialized():
                        world_size = dist.get_world_size()

            effective_batch_size = (
                batch["gene"].x.size(0) * self.current_accumulation_steps * world_size
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                batch_size=batch["gene"].x.size(0),
                sync_dist=True,
            )
        # print(f"Loss: {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "val")

        # Defragment GPU memory every 50 batches to prevent OOM from fragmentation
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

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
        if not samples["true_values"]:
            return

        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)

        # Process latents if they exist
        latents = {}
        if "latents" in samples and samples["latents"]:
            for k, v in samples["latents"].items():
                if v:  # Check if the list is not empty
                    latents[k] = torch.cat(v, dim=0)

        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
            for key in latents:
                latents[key] = latents[key][idx]

        # Use Visualization for enhanced plotting
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )

        loss_name = (
            self.loss_func.__class__.__name__ if self.loss_func is not None else "Loss"
        )

        # Ensure data is in the correct format for visualization
        # For gene interactions, we only need a single dimension
        if true_values.dim() == 1:
            true_values = true_values.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        # For hetero_cell_bipartite_dango_gi, we use z_p latents
        z_p_latents = {}
        if "z_p" in latents:
            z_p_latents["z_p"] = latents["z_p"]

        # Use our updated visualize_model_outputs method
        vis.visualize_model_outputs(
            predictions,
            true_values,
            z_p_latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )

        # Log oversmoothing metrics on latent spaces if available
        if "z_p" in latents:
            smoothness = VisGraphDegen.compute_smoothness(latents["z_p"])
            wandb.log({f"{stage}/oversmoothing_z_p": smoothness.item()})

        # Log genetic interaction box plot
        if torch.any(~torch.isnan(true_values)):
            # For gene interactions, values are in the first dimension
            fig_gi = genetic_interaction_score.box_plot(
                true_values[:, 0].cpu(), predictions[:, 0].cpu()
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
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

        # Step the scheduler when using manual optimization
        sch = self.lr_schedulers()
        if sch is not None:
            # Lightning returns a list of schedulers even if there's only one
            if isinstance(sch, list) and len(sch) > 0:
                sch[0].step()
            else:
                sch.step()

        # CRITICAL: Clear GPU memory at end of training epoch
        # This ensures validation starts with maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_start(self):
        # Update gradient accumulation steps based on current epoch
        if self.hparams.grad_accumulation_schedule is not None:
            for epoch_threshold in sorted(
                self.hparams.grad_accumulation_schedule.keys()
            ):
                # Convert epoch_threshold to int if it's a string
                epoch_threshold_int = (
                    int(epoch_threshold)
                    if isinstance(epoch_threshold, str)
                    else epoch_threshold
                )
                if self.current_epoch >= epoch_threshold_int:
                    self.current_accumulation_steps = (
                        self.hparams.grad_accumulation_schedule[epoch_threshold]
                    )
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_start(self):
        # CRITICAL: Aggressively clear GPU memory before validation starts
        # This prevents OOM when transitioning from training to validation
        # Training state (optimizer, gradients, cached activations) can fragment memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all pending operations complete before validation
            torch.cuda.synchronize()

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_start(self):
        # Always clear sample containers for test (test runs only once)
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

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
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_end(self):
        # Log test metrics
        computed_metrics = self._compute_metrics_safely(self.test_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.test_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_transformed_metrics.reset()

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            # Reset the sample containers
            self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hparams.optimizer_config.items() if k != "type"
        }
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # If no lr_scheduler_config is provided, return just the optimizer
        if self.hparams.lr_scheduler_config is None:
            return optimizer

        # Handle different scheduler types
        scheduler_type = self.hparams.lr_scheduler_config.get(
            "type", "ReduceLROnPlateau"
        )
        scheduler_params = {
            k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"
        }

        if scheduler_type == "CosineAnnealingWarmupRestarts":
            # Import the custom scheduler
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Default to ReduceLROnPlateau
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


class DiffusionRegressionTask(L.LightningModule):
    """Standalone regression task specifically for diffusion models.

    This task handles the unique training requirements of diffusion models,
    including diffusion loss during training and MSE evaluation during validation/testing.
    All visualization and metric tracking functionality is preserved from RegressionTask.
    """

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
        inverse_transform: Optional[nn.Module] = None,
        execution_mode: str = "training",  # "training" or "dataloader_profiling"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.execution_mode = execution_mode
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func

        # Initialize gradient accumulation
        self.current_accumulation_steps = 1
        if self.hparams.grad_accumulation_schedule is not None:
            self.current_accumulation_steps = (
                self.hparams.grad_accumulation_schedule.get(0, 1)
            )

        # Setup metrics
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

        # Sample accumulators for visualization
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

        # Diffusion-specific tracking
        self.train_diffusion_loss = []
        self.val_mse_during_inference = []

        # Manual optimization for gradient accumulation
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
        # DataLoader profiling mode: Skip model forward, create dummy loss
        if self.execution_mode == "dataloader_profiling":
            # Execute all batch preparation (moving to device happens in forward())
            batch_device = batch["gene"].x.device
            # Ensure cell_graph is on correct device
            if not hasattr(self, "_cell_graph_device") or self._cell_graph_device != batch_device:
                self.cell_graph = self.cell_graph.to(batch_device)
                self._cell_graph_device = batch_device

            # Create trivial loss that touches ALL model parameters (required for DDP)
            # This ensures no "unused parameters" error in DDP mode
            loss = torch.zeros((), device=batch_device, requires_grad=True)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss = loss + (param * 0.0).sum()

            # Log minimal metrics
            batch_size = batch["gene"].x.size(0)
            self.log(f"{stage}/dataloader_profile_loss", loss, batch_size=batch_size, sync_dist=True)
            self.log(f"{stage}/dataloader_profile_batch_size", float(batch_size), batch_size=batch_size, sync_dist=True)

            return loss, None, None

        # Normal training/validation/test execution
        # Get model outputs
        predictions, representations = self(batch)

        # Ensure predictions has correct shape (batch_size, 1)
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Get target values - now in COO format
        gene_interaction_vals = batch["gene"].phenotype_values

        # Handle tensor shape
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)

        # For original values, check if there's a phenotype_values_original
        if hasattr(batch["gene"], "phenotype_values_original"):
            gene_interaction_orig = batch["gene"].phenotype_values_original
        else:
            gene_interaction_orig = gene_interaction_vals

        # Handle tensor shape
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)

        # Get z_p from representations for conditioning
        z_p = representations.get("z_p")

        # Compute loss based on training/evaluation stage
        if stage == "train":
            # During training, use diffusion loss (noise prediction)
            loss_output = self.loss_func(predictions, gene_interaction_vals, z_p)

            # Handle tuple return from DiffusionLoss
            if isinstance(loss_output, tuple):
                loss = loss_output[0]
                loss_dict = loss_output[1] if len(loss_output) > 1 else {}

                # Log diffusion-specific metrics
                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        self.log(
                            f"{stage}/{key}",
                            value.item() if value.numel() == 1 else value.mean().item(),
                            batch_size=batch_size,
                            sync_dist=True,
                        )
            else:
                loss = loss_output
                loss_dict = {}

            # Track diffusion loss separately
            self.train_diffusion_loss.append(loss.detach())

        else:
            # During validation/test, evaluate using MSE on inference samples
            mse_loss = nn.functional.mse_loss(predictions, gene_interaction_vals)
            loss = mse_loss
            loss_dict = {}

            # Log inference MSE separately
            self.log(
                f"{stage}/inference_mse",
                mse_loss.item(),
                batch_size=batch_size,
                sync_dist=True,
            )

            if stage == "val":
                self.val_mse_during_inference.append(mse_loss.detach())

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

        # Log gate weights if available (for Dango-like architectures)
        if "gate_weights" in representations:
            gate_weights = representations["gate_weights"]
            avg_gate_weights = gate_weights.mean(dim=0)
            self.log(
                f"{stage}/gate_weight_global",
                avg_gate_weights[0],
                batch_size=batch_size,
                sync_dist=True,
            )
            # Only log local weight if it exists (when local predictor is enabled)
            if avg_gate_weights.size(0) > 1:
                self.log(
                    f"{stage}/gate_weight_local",
                    avg_gate_weights[1],
                    batch_size=batch_size,
                    sync_dist=True,
                )

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions in COO format
            temp_data = HeteroData()

            # Create COO format data for predictions
            batch_size = predictions.size(0)
            device = predictions.device
            temp_data["gene"].phenotype_values = predictions.squeeze()
            temp_data["gene"].phenotype_type_indices = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            temp_data["gene"].phenotype_sample_indices = torch.arange(
                batch_size, device=device
            )
            temp_data["gene"].phenotype_types = ["gene_interaction"]

            # Apply the inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_gene_int = inv_data["gene"]["phenotype_values"]

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
                    if z_p is not None:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "z_p" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["z_p"] = []
                        self.train_samples["latents"]["z_p"].append(z_p[idx].detach())
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())
                    if z_p is not None:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "z_p" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["z_p"] = []
                        self.train_samples["latents"]["z_p"].append(z_p.detach())
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot, respecting ceiling
            current_count = sum(t.size(0) for t in self.val_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.val_samples["true_values"].append(
                        gene_interaction_orig[idx].detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions[idx].detach())
                    if z_p is not None:
                        if "latents" not in self.val_samples:
                            self.val_samples["latents"] = {}
                        if "z_p" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["z_p"] = []
                        self.val_samples["latents"]["z_p"].append(z_p[idx].detach())
                else:
                    self.val_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.val_samples["predictions"].append(inv_predictions.detach())
                    if z_p is not None:
                        if "latents" not in self.val_samples:
                            self.val_samples["latents"] = {}
                        if "z_p" not in self.val_samples["latents"]:
                            self.val_samples["latents"]["z_p"] = []
                        self.val_samples["latents"]["z_p"].append(z_p.detach())
        elif stage == "test":
            # For test, always collect samples (no test runs once)
            self.test_samples["true_values"].append(gene_interaction_orig.detach())
            self.test_samples["predictions"].append(inv_predictions.detach())
            if z_p is not None:
                if "latents" not in self.test_samples:
                    self.test_samples["latents"] = {}
                if "z_p" not in self.test_samples["latents"]:
                    self.test_samples["latents"]["z_p"] = []
                self.test_samples["latents"]["z_p"].append(z_p.detach())

        return loss, predictions, gene_interaction_orig

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Model profiling mode: Skip optimizer step to isolate model compute
        if self.execution_mode == "model_profiling":
            return loss

        # Normal training: Run optimizer
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
        # Log effective batch size when using gradient accumulation
        if self.hparams.grad_accumulation_schedule is not None:
            # Get world size for DDP
            world_size = 1
            if hasattr(self.trainer, "strategy") and hasattr(
                self.trainer.strategy, "_strategy_name"
            ):
                if self.trainer.strategy._strategy_name == "ddp":
                    import torch.distributed as dist

                    if dist.is_initialized():
                        world_size = dist.get_world_size()

            effective_batch_size = (
                batch["gene"].x.size(0) * self.current_accumulation_steps * world_size
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                batch_size=batch["gene"].x.size(0),
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx, "val")

        # Defragment GPU memory every 50 batches to prevent OOM from fragmentation
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

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
        if not samples["true_values"]:
            return

        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)

        # Process latents if they exist
        latents = {}
        if "latents" in samples and samples["latents"]:
            for k, v in samples["latents"].items():
                if v:  # Check if the list is not empty
                    latents[k] = torch.cat(v, dim=0)

        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
            for key in latents:
                latents[key] = latents[key][idx]

        # Use Visualization for enhanced plotting
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )

        loss_name = (
            self.loss_func.__class__.__name__ if self.loss_func is not None else "Loss"
        )

        # Ensure data is in the correct format for visualization
        if true_values.dim() == 1:
            true_values = true_values.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)

        # For diffusion models, we use z_p latents
        z_p_latents = {}
        if "z_p" in latents:
            z_p_latents["z_p"] = latents["z_p"]

        # Use our updated visualize_model_outputs method
        vis.visualize_model_outputs(
            predictions,
            true_values,
            z_p_latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )

        # Log oversmoothing metrics on latent spaces if available
        if "z_p" in latents:
            smoothness = VisGraphDegen.compute_smoothness(latents["z_p"])
            wandb.log({f"{stage}/oversmoothing_z_p": smoothness.item()})

        # Log genetic interaction box plot
        if torch.any(~torch.isnan(true_values)):
            fig_gi = genetic_interaction_score.box_plot(
                true_values[:, 0].cpu(), predictions[:, 0].cpu()
            )
            wandb.log({f"{stage}/gene_interaction_box_plot": wandb.Image(fig_gi)})
            plt.close(fig_gi)

    def on_train_epoch_start(self):
        # Update gradient accumulation steps based on current epoch
        if self.hparams.grad_accumulation_schedule is not None:
            for epoch_threshold in sorted(
                self.hparams.grad_accumulation_schedule.keys()
            ):
                # Convert epoch_threshold to int if it's a string
                epoch_threshold_int = (
                    int(epoch_threshold)
                    if isinstance(epoch_threshold, str)
                    else epoch_threshold
                )
                if self.current_epoch >= epoch_threshold_int:
                    self.current_accumulation_steps = (
                        self.hparams.grad_accumulation_schedule[epoch_threshold]
                    )
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_start(self):
        # CRITICAL: Aggressively clear GPU memory before validation starts
        # This prevents OOM when transitioning from training to validation
        # Training state (optimizer, gradients, cached activations) can fragment memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all pending operations complete before validation
            torch.cuda.synchronize()

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_start(self):
        # Always clear sample containers for test (test runs only once)
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

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

        # Log average diffusion loss if available
        if self.train_diffusion_loss:
            avg_diffusion_loss = torch.stack(self.train_diffusion_loss).mean()
            self.log("train/avg_diffusion_loss", avg_diffusion_loss, sync_dist=True)
            self.train_diffusion_loss = []

        # Plot training samples
        if (
            self.current_epoch + 1
        ) % self.hparams.plot_every_n_epochs == 0 and self.train_samples["true_values"]:
            self._plot_samples(self.train_samples, "train_sample")
            # Reset the sample containers
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

        # Step the scheduler when using manual optimization
        sch = self.lr_schedulers()
        if sch is not None:
            if isinstance(sch, list) and len(sch) > 0:
                sch[0].step()
            else:
                sch.step()

        # CRITICAL: Clear GPU memory at end of training epoch
        # This ensures validation starts with maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_start(self):
        # Update gradient accumulation steps based on current epoch
        if self.hparams.grad_accumulation_schedule is not None:
            for epoch_threshold in sorted(self.hparams.grad_accumulation_schedule.keys()):
                epoch_threshold_int = (
                    int(epoch_threshold)
                    if isinstance(epoch_threshold, str)
                    else epoch_threshold
                )
                if self.current_epoch >= epoch_threshold_int:
                    self.current_accumulation_steps = (
                        self.hparams.grad_accumulation_schedule[epoch_threshold]
                    )
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

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

        # Log average inference MSE if available
        if self.val_mse_during_inference:
            avg_inference_mse = torch.stack(self.val_mse_during_inference).mean()
            self.log("val/avg_inference_mse", avg_inference_mse, sync_dist=True)
            self.val_mse_during_inference = []

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            # Reset the sample containers
            self.val_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_test_epoch_end(self):
        # Log test metrics
        computed_metrics = self._compute_metrics_safely(self.test_metrics)
        for name, value in computed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_metrics.reset()

        # Compute and log transformed metrics
        transformed_metrics = self._compute_metrics_safely(
            self.test_transformed_metrics
        )
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_transformed_metrics.reset()

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            # Reset the sample containers
            self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

    def configure_optimizers(self):
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hparams.optimizer_config.items() if k != "type"
        }
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # If no lr_scheduler_config is provided, return just the optimizer
        if self.hparams.lr_scheduler_config is None:
            return optimizer

        # Handle different scheduler types
        scheduler_type = self.hparams.lr_scheduler_config.get(
            "type", "ReduceLROnPlateau"
        )
        scheduler_params = {
            k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"
        }

        if scheduler_type == "CosineAnnealingWarmupRestarts":
            # Import the custom scheduler
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Default to ReduceLROnPlateau
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
