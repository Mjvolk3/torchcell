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
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.forward_transform = forward_transform
        self.current_accumulation_steps = 1
        self.loss_func = loss_func

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )

        # Create metrics for each stage - fixed to create metrics first then move them
        for stage in ["train", "val", "test"]:
            # First create the metrics
            metrics_dict = reg_metrics.clone(prefix=f"{stage}/gene_interaction/")
            transformed_metrics = reg_metrics.clone(
                prefix=f"{stage}/transformed/gene_interaction/"
            )

            # Store the metrics (will move to device in _shared_step when needed)
            setattr(self, f"{stage}_metrics", metrics_dict)
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

        # Separate accumulators for train, validation, and test samples
        self.train_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.val_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.test_samples = {"true_values": [], "predictions": [], "latents": {}}
        self.automatic_optimization = False

    def forward(self, batch):
        """Forward pass through the model"""
        # Get model device to ensure consistency
        model_device = next(self.model.parameters()).device

        # Move batch to model's device if it isn't already
        batch_device = batch.device if hasattr(batch, "device") else None

        if batch_device is None or batch_device != model_device:
            # Handle PyG batches where device isn't a direct attribute
            if hasattr(batch, "gene") and hasattr(
                batch["gene"], "perturbation_indices"
            ):
                batch_device = batch["gene"].perturbation_indices.device

            # Move the batch if needed
            if batch_device != model_device:
                batch = batch.to(model_device)

        # Always ensure cell_graph is on the model's device
        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != model_device
        ):
            self.cell_graph = self.cell_graph.to(model_device)
            self._cell_graph_device = model_device

        # Return all outputs from the model
        predictions, outputs_dict = self.model(self.cell_graph, batch)

        # Return the full outputs_dict to avoid redundant forward passes
        return predictions, outputs_dict

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

        # Get target values and ensure they are on same device as predictions
        device = predictions.device
        gene_interaction_vals = batch["gene"].phenotype_values.to(device)

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
            batch["gene"].phenotype_values_original.to(device)
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
            # Use the outputs_dict from representations (already computed in line 132)
            # This avoids redundant forward pass
            outputs_dict = representations

            # Pass to the loss function
            loss_output = self.loss_func(
                predictions, outputs_dict, gene_interaction_vals
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

        # Get metrics and make sure they're on the correct device
        transformed_metrics = getattr(self, f"{stage}_transformed_metrics").to(device)
        metrics = getattr(self, f"{stage}_metrics").to(device)

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            # Ensure all inputs to metrics are on the same device
            pred_transformed = predictions[mask].view(-1).to(device)
            target_transformed = gene_interaction_vals[mask].view(-1).to(device)
            transformed_metrics.update(pred_transformed, target_transformed)

        # Handle inverse transform if available
        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions
            temp_data = HeteroData()
            temp_data["gene"] = {"gene_interaction": predictions.clone().squeeze()}
            temp_data = temp_data.to(device)  # Ensure it's on the right device

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
            # Ensure all inputs to metrics are on the same device
            pred_orig = inv_predictions[mask].view(-1).to(device)
            target_orig = gene_interaction_orig[mask].view(-1).to(device)
            metrics.update(pred_orig, target_orig)

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
                        gene_interaction_orig[idx].detach().float()
                    )
                    self.train_samples["predictions"].append(
                        inv_predictions[idx].detach().float()
                    )
                    # Collect subsystem outputs if available
                    if subsystem_outputs:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "subsystem_outputs" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["subsystem_outputs"] = []
                        # Select just the final layer or a representative subset
                        if "root" in subsystem_outputs:
                            self.train_samples["latents"]["subsystem_outputs"].append(
                                subsystem_outputs["root"][idx].detach().float()
                            )
                else:
                    self.train_samples["true_values"].append(
                        gene_interaction_orig.detach().float()
                    )
                    self.train_samples["predictions"].append(inv_predictions.detach().float())
                    # Collect subsystem outputs if available
                    if subsystem_outputs:
                        if "latents" not in self.train_samples:
                            self.train_samples["latents"] = {}
                        if "subsystem_outputs" not in self.train_samples["latents"]:
                            self.train_samples["latents"]["subsystem_outputs"] = []
                        # Select just the final layer or a representative subset
                        if "root" in subsystem_outputs:
                            self.train_samples["latents"]["subsystem_outputs"].append(
                                subsystem_outputs["root"].detach().float()
                            )
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot
            self.val_samples["true_values"].append(gene_interaction_orig.detach().float())
            self.val_samples["predictions"].append(inv_predictions.detach().float())
            # Collect subsystem outputs if available
            if subsystem_outputs:
                if "latents" not in self.val_samples:
                    self.val_samples["latents"] = {}
                if "subsystem_outputs" not in self.val_samples["latents"]:
                    self.val_samples["latents"]["subsystem_outputs"] = []
                # Select just the final layer or a representative subset  
                if "root" in subsystem_outputs:
                    self.val_samples["latents"]["subsystem_outputs"].append(
                        subsystem_outputs["root"].detach().float()
                    )
        elif stage == "test":
            # For test, always collect samples (no epoch check since test runs once)
            self.test_samples["true_values"].append(gene_interaction_orig.detach().float())
            self.test_samples["predictions"].append(inv_predictions.detach().float())
            # Collect subsystem outputs if available
            if subsystem_outputs:
                if "latents" not in self.test_samples:
                    self.test_samples["latents"] = {}
                if "subsystem_outputs" not in self.test_samples["latents"]:
                    self.test_samples["latents"]["subsystem_outputs"] = []
                # Select just the final layer or a representative subset
                if "root" in subsystem_outputs:
                    self.test_samples["latents"]["subsystem_outputs"].append(
                        subsystem_outputs["root"].detach().float()
                    )

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

        # For DCell, we use subsystem_outputs as latents if available
        z_p_latents = {}
        if "subsystem_outputs" in latents:
            z_p_latents["subsystem_outputs"] = latents["subsystem_outputs"]

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
        
        # Log oversmoothing metrics on latent spaces if available
        if "subsystem_outputs" in latents:
            smoothness = VisGraphDegen.compute_smoothness(latents["subsystem_outputs"])
            wandb.log({f"{stage}/oversmoothing_subsystem": smoothness.item()})

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
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_train_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {"true_values": [], "predictions": [], "latents": {}}

    def on_validation_epoch_start(self):
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
        transformed_metrics = self._compute_metrics_safely(self.test_transformed_metrics)
        for name, value in transformed_metrics.items():
            self.log(name, value, sync_dist=True)
        self.test_transformed_metrics.reset()

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            # Reset the sample containers
            self.test_samples = {"true_values": [], "predictions": [], "latents": {}}

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
            
            # Add batch information for gene nodes (required for DCellOpt)
            dummy_batch["gene"].batch = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Add critical fields for DCellOpt model
            # The model pre-computes indices expecting the full dataset structure
            # Use the actual go_gene_strata_state structure from cell_graph
            if hasattr(cell_graph_device["gene_ontology"], "go_gene_strata_state"):
                # Clone the structure but zero out the states (column 3)
                dummy_state = cell_graph_device["gene_ontology"].go_gene_strata_state.clone()
                dummy_state[:, 3] = 0  # Zero out all states for dummy batch
                dummy_batch["gene_ontology"].go_gene_strata_state = dummy_state
                # Set pointer to indicate one sample with full structure
                dummy_batch["gene_ontology"].go_gene_strata_state_ptr = torch.tensor(
                    [0, len(dummy_state)], dtype=torch.long, device=self.device
                )
            else:
                # Fallback for models that don't use go_gene_strata_state
                # Create a minimal dummy state
                dummy_state = torch.tensor([[0, 0, 0, 0]], dtype=torch.long, device=self.device)
                dummy_batch["gene_ontology"].go_gene_strata_state = dummy_state
                dummy_batch["gene_ontology"].go_gene_strata_state_ptr = torch.tensor(
                    [0, 1], dtype=torch.long, device=self.device
                )
            
            # Add mutant_state for backward compatibility with original DCell
            dummy_batch["gene_ontology"].mutant_state = torch.zeros(
                1, 3, device=self.device
            )
            
            # Add phenotype values for loss calculation
            dummy_batch["gene"].phenotype_values = torch.zeros(1, device=self.device)
            
            # Add perturbation indices
            dummy_batch["gene"].perturbation_indices = torch.zeros(1, dtype=torch.long, device=self.device)

            # Add basic batch info
            dummy_batch.num_graphs = 1

            # Initialize all model parameters with a forward pass
            # This is necessary because DCellModel creates parameters dynamically
            try:
                # Always run the dummy forward pass to initialize dynamic parameters
                # DCell/DCellOpt creates subsystem modules during first forward pass
                self.model(cell_graph_device, dummy_batch)
                log.info("Model parameters initialized successfully")
            except Exception as e:
                log.warning(f"Error during model initialization: {e}")
                # Add fallback dummy parameters to allow optimizer creation
                self.model.register_parameter(
                    "dummy", torch.nn.Parameter(torch.zeros(1, device=self.device))
                )
                log.info("Added dummy parameter to allow optimizer creation")

        # Check if we have parameters after initialization
        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Total trainable parameters: {param_count:,}")

        if param_count == 0:
            log.warning(
                "No parameters found after initialization, adding dummy parameter"
            )
            # Add a dummy parameter as fallback
            self.register_parameter(
                "dummy", torch.nn.Parameter(torch.zeros(1, device=self.device))
            )

        # Now create the optimizer with the initialized parameters
        optimizer_class = getattr(torch.optim, self.hparams.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hparams.optimizer_config.items() if k != "type"
        }
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")

        optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # Set up scheduler
        scheduler_params = {
            k: v for k, v in self.hparams.lr_scheduler_config.items() if k != "type"
        }
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
