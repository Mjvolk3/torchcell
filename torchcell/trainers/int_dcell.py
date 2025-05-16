# torchcell/trainers/int_dcell
# [[torchcell.trainers.int_dcell]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/int_dcell
# Test file: tests/torchcell/trainers/test_int_dcell.py


import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
from typing import Dict, Optional, Tuple, Union, List

from torchcell.losses.dcell import DCellLoss
from torchcell.timestamp import timestamp
from torchcell.viz import genetic_interaction_score
from torchcell.viz.visual_regression import Visualization
from torchcell.viz.visual_graph_degen import VisGraphDegen

log = logging.getLogger(__name__)


class RegressionTask(LightningModule):
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
        self.save_hyperparameters(ignore=["model", "loss_func"])
        self.model = model
        self.cell_graph = cell_graph
        self.inverse_transform = inverse_transform
        self.forward_transform = forward_transform
        self.current_accumulation_steps = 1
        self.loss_func = loss_func
        
        # Get device - use the one passed in if model parameters aren't initialized yet
        try:
            # Try to get device from model parameters
            self.device = next(model.parameters()).device
        except StopIteration:
            # If model has no parameters yet, use the device passed in
            self.device = torch.device(device)
            log.warning(f"Model has no parameters yet, using specified device: {self.device}")
        
        log.info(f"Initializing RegressionTask with device: {self.device}")

        # Create metrics on the correct device
        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True).to(self.device),
                "RMSE": MeanSquaredError(squared=False).to(self.device),
                "Pearson": PearsonCorrCoef().to(self.device),
            }
        )

        # Create metrics for each stage - ensuring they're on the right device
        for stage in ["train", "val", "test"]:
            metrics_dict = reg_metrics.clone(prefix=f"{stage}/gene_interaction/").to(self.device)
            setattr(self, f"{stage}_metrics", metrics_dict)

            # Add metrics operating in transformed space
            transformed_metrics = reg_metrics.clone(
                prefix=f"{stage}/transformed/gene_interaction/"
            ).to(self.device)
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

        # Separate accumulators for train and validation samples
        self.train_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}
        self.val_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}
        self.automatic_optimization = False

    def forward(self, batch):
        """
        Forward pass through the model
        
        Args:
            batch: HeteroData batch containing perturbation information
            
        Returns:
            Tuple of (predictions, representations)
        """
        # Get model device to ensure consistency
        model_device = next(self.model.parameters()).device
        
        # Move batch to model's device
        batch = batch.to(model_device)
        
        # Always ensure cell_graph is on the model's device
        self.cell_graph = self.cell_graph.to(model_device)
        self._cell_graph_device = model_device
            
        # Return all outputs from the model
        # DCellModel returns (predictions, outputs_dict)
        predictions, outputs_dict = self.model(self.cell_graph, batch)
        
        # Return the model outputs in a standardized format
        return predictions, {"subsystem_outputs": outputs_dict.get("subsystem_outputs", {})}

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
        
        # Ensure predictions has correct shape (batch_size, 1) for gene interactions
        if predictions.dim() == 0:
            predictions = predictions.unsqueeze(0).unsqueeze(0)  # Make it [1, 1]
        elif predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)  # Make it [batch_size, 1]

        batch_size = predictions.size(0)

        # Get target values
        gene_interaction_vals = batch["gene"].phenotype_values
        
        # Handle tensor shape - ensure it's [batch_size, 1] for consistency
        if gene_interaction_vals.dim() == 0:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_vals.dim() == 1:
            gene_interaction_vals = gene_interaction_vals.unsqueeze(1)
        elif gene_interaction_vals.dim() > 1 and gene_interaction_vals.size(1) > 1:
            # If we somehow have multiple dimensions, keep only the first one
            gene_interaction_vals = gene_interaction_vals[:, 0:1]

        # Get original values for metrics and visualization
        gene_interaction_orig = (
            batch["gene"].phenotype_values_original
            if hasattr(batch["gene"], "phenotype_values_original")
            else gene_interaction_vals
        )

        # Handle tensor shape - ensure it's [batch_size, 1] for consistency
        if gene_interaction_orig.dim() == 0:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(0).unsqueeze(0)
        elif gene_interaction_orig.dim() == 1:
            gene_interaction_orig = gene_interaction_orig.unsqueeze(1)
        elif gene_interaction_orig.dim() > 1 and gene_interaction_orig.size(1) > 1:
            # If we somehow have multiple dimensions, keep only the first one
            gene_interaction_orig = gene_interaction_orig[:, 0:1]

        # Get subsystem outputs from representations
        subsystem_outputs = representations.get("subsystem_outputs", {})

        # Calculate loss
        if self.loss_func is None:
            raise ValueError("No loss function provided")

        # For DCellLoss, we need to pass the outputs dictionary for auxiliary losses
        if isinstance(self.loss_func, DCellLoss):
            # Extract reconstructions and prepare adjacency matrices
            # Call the loss function with current epoch for dynamic weighting
            _, outputs_dict = self.model(self.cell_graph, batch)
            
            # Pass to the loss function
            loss_output = self.loss_func(
                predictions, 
                outputs_dict,
                gene_interaction_vals
            )
            
            # Handle the tuple return format (loss, loss_dict)
            loss = loss_output[0]  # First element is the loss
            loss_dict = loss_output[1] if len(loss_output) > 1 else {}
            
            # Log additional loss components
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
            # For other loss functions (fallback)
            loss = self.loss_func(predictions, gene_interaction_vals)
            if isinstance(loss, tuple):
                loss = loss[0]  # Take first element if tuple

        # Add dummy loss for unused parameters
        dummy_loss = self._ensure_no_unused_params_loss()
        loss = loss + dummy_loss

        # Log the loss
        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            # Ensure tensors are on the same device as metrics
            preds_device = predictions[mask].view(-1).to(self.device)
            target_device = gene_interaction_vals[mask].view(-1).to(self.device)
            transformed_metrics.update(preds_device, target_device)

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
            # Ensure tensors are on the same device as metrics
            inv_preds_device = inv_predictions[mask].view(-1).to(self.device)
            orig_target_device = gene_interaction_orig[mask].view(-1).to(self.device)
            metrics.update(inv_preds_device, orig_target_device)

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
                    # Note: We're not collecting subsystem outputs here as they're big and complex
                    # We can enable this if needed for visualization
                    # self.train_samples["latents"]["subsystem_outputs"] = {...}
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach())
                    # Same note as above
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot
            self.val_samples["true_values"].append(gene_interaction_orig.detach())
            self.val_samples["predictions"].append(inv_predictions.detach())
            # Same note as above

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
            batch_size=batch["gene"].phenotype_values.size(0),
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
        if not samples["true_values"]:
            return
            
        true_values = torch.cat(samples["true_values"], dim=0)
        predictions = torch.cat(samples["predictions"], dim=0)
        
        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
        
        # Use Visualization for enhanced plotting
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )
        
        loss_name = (
            self.loss_func.__class__.__name__
            if self.loss_func is not None
            else "Loss"
        )
        
        # Ensure data is in the correct format for visualization
        # For gene interactions, we only need a single dimension
        if true_values.dim() == 1:
            true_values = true_values.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        
        # Skip UMAP visualization by passing empty latents dictionary
        z_p_latents = {}  # Empty dictionary to skip UMAP visualization
        
        # Use our updated visualize_model_outputs method which now properly handles single target case
        vis.visualize_model_outputs(
            predictions,
            true_values,
            z_p_latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )
        
        # Log genetic interaction box plot - for gene interactions, always use the first column
        if torch.any(~torch.isnan(true_values)):
            # For DCell model, genetic interaction values are in the first dimension (index 0)
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
            self.train_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}

    def on_train_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}

    def on_validation_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.val_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}

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
            self.val_samples = {"true_values": [], "predictions": [], "latents": {"subsystem_outputs": []}}

    def configure_optimizers(self):
        """
        Set up optimizers for training.
        
        For DCellModel, we need to pre-register parameters by running a forward pass.
        """
        log.info("Setting up optimizer and initializing DCellModel parameters")
        
        # Initialize the model parameters by doing a forward pass
        # This must happen BEFORE creating the optimizer
        with torch.no_grad():
            # Get the cell graph and move it to the right device
            cell_graph_device = self.cell_graph.to(self.device)
            
            # Create a minimal dummy batch for initialization
            from torch_geometric.data import HeteroData, Data
            dummy_batch = HeteroData()
            dummy_batch["gene"] = Data()
            dummy_batch["gene_ontology"] = Data()
            
            # Need to set term_ids and other critical fields that the model will check for
            # Use .x to set node features (this is standard in PyG)
            dummy_batch["gene"].x = torch.zeros(1, 1, device=self.device)
            dummy_batch["gene_ontology"].x = torch.zeros(1, 1, device=self.device)
            
            # Add critical fields that are checked in the forward pass
            dummy_batch["gene_ontology"].mutant_state = torch.zeros(1, 3, device=self.device)
            
            # Add basic batch info
            dummy_batch.num_graphs = 1
            
            # Initialize all model parameters with a forward pass
            # This is necessary because DCellModel creates parameters dynamically
            try:
                self.model(cell_graph_device, dummy_batch)
                log.info("Model parameters initialized successfully")
            except Exception as e:
                log.warning(f"Error during model initialization: {e}")
                # Add fallback dummy parameters to allow optimizer creation
                self.model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1, device=self.device)))
                log.info("Added dummy parameter to allow optimizer creation")
        
        # Check if we have parameters after initialization
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters: {param_count:,}")
        
        if param_count == 0:
            log.warning("No parameters found after initialization, adding dummy parameter")
            # Add a dummy parameter as fallback
            self.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1, device=self.device)))
        
        # Now create the optimizer with the initialized parameters
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
        optimizer_params = {k: v for k, v in self.hparams.optimizer_config.items() if k != "type"}
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        
        optimizer = optimizer_class(self.parameters(), **optimizer_params)
        
        # Set up scheduler
        scheduler_params = {k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"}
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
        
        # Return configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/gene_interaction/MSE",
                "interval": "epoch",
                "frequency": 1,
            },
        }
