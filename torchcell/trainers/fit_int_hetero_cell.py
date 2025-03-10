import lightning as L
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection, MeanSquaredError, PearsonCorrCoef
import matplotlib.pyplot as plt
from typing import Optional
import logging
from torchcell.viz.visual_regression import Visualization
from torchcell.timestamp import timestamp
from torchcell.viz.visual_graph_degen import VisGraphDegen
from torchcell.viz import genetic_interaction_score
from torchcell.viz import fitness
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
        plot_every_n_epochs: int = 10,  # New parameter for plotting frequency
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
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.current_accumulation_steps = 1
        self.loss_func = loss_func

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )

        # Create per-label metrics as you already have
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

            # Add new combined metrics with Pearson
            combined_metrics = MetricCollection(
                {
                    "MSE": MeanSquaredError(squared=True),
                    "RMSE": MeanSquaredError(squared=False),
                    "Pearson": PearsonCorrCoef(),  # Add Pearson to combined metrics
                }
            )
            setattr(
                self,
                f"{stage}_combined_metrics",
                combined_metrics.clone(prefix=f"{stage}/combined/"),
            )
            # Add new metrics operating in transformed space
            transformed_metrics = nn.ModuleDict(
                {
                    "fitness": reg_metrics.clone(
                        prefix=f"{stage}/transformed/fitness/"
                    ),
                    "gene_interaction": reg_metrics.clone(
                        prefix=f"{stage}/transformed/gene_interaction/"
                    ),
                }
            )
            setattr(self, f"{stage}_transformed_metrics", transformed_metrics)

            transformed_combined = reg_metrics.clone(
                prefix=f"{stage}/transformed/combined/"
            )
            setattr(self, f"{stage}_transformed_combined", transformed_combined)

        # Separate accumulators for train and validation samples.
        self.train_samples = {
            "true_values": [],
            "predictions": [],
            "latents": {"z_p": []},
        }
        self.val_samples = {
            "true_values": [],
            "predictions": [],
            "latents": {"z_p": []},
        }
        self.automatic_optimization = False

    def forward(self, batch):
        batch_device = batch["gene"].x.device
        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != batch_device
        ):
            self.cell_graph = self.cell_graph.to(batch_device)
            self._cell_graph_device = batch_device
        return self.model(self.cell_graph, batch)

    def _shared_step(self, batch, batch_idx, stage="train"):
        predictions, representations = self(batch)
        batch_size = predictions.size(0)

        # Use transformed values for loss computation
        fitness_vals = batch["gene"].fitness.view(-1, 1)
        gene_interaction_vals = batch["gene"].gene_interaction.view(-1, 1)
        targets = torch.cat([fitness_vals, gene_interaction_vals], dim=1)

        # Get original values for metrics and visualization
        fitness_orig = (
            batch["gene"].fitness_original.view(-1, 1)
            if "fitness_original" in batch["gene"]
            else fitness_vals
        )
        gene_interaction_orig = (
            batch["gene"].gene_interaction_original.view(-1, 1)
            if "gene_interaction_original" in batch["gene"]
            else gene_interaction_vals
        )
        orig_targets = torch.cat([fitness_orig, gene_interaction_orig], dim=1)

        loss, loss_dict = self.loss_func(predictions, targets, representations["z_p"])

        # Log loss components
        for key, value in [
            ("loss", loss),
            ("fitness_loss", loss_dict["mse_dim_losses"][0]),
            ("gene_interaction_loss", loss_dict["mse_dim_losses"][1]),
            ("mse_loss", loss_dict["mse_loss"]),
            ("dist_loss", loss_dict["dist_loss"]),
            ("supcr_loss", loss_dict["supcr_loss"]),
            ("total_loss", loss_dict["total_loss"]),
        ]:
            self.log(f"{stage}/{key}", value, batch_size=batch_size, sync_dist=True)

        # Additional loss component logging
        for key in [
            "weighted_mse",
            "weighted_dist",
            "weighted_supcr",
            "total_weighted",
            "norm_weighted_mse",
            "norm_weighted_dist",
            "norm_weighted_supcr",
            "norm_unweighted_mse",
            "norm_unweighted_dist",
            "norm_unweighted_supcr",
        ]:
            if key in loss_dict:
                self.log(
                    f"{stage}/{key}",
                    loss_dict[key],
                    batch_size=batch_size,
                    sync_dist=True,
                )

        # Log z_p norm
        if "z_p" in representations:
            z_p_norm = representations["z_p"].norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/z_p_norm", z_p_norm, batch_size=batch_size, sync_dist=True
            )

        transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
        for key, col in zip(["fitness", "gene_interaction"], [0, 1]):
            mask = ~torch.isnan(targets[:, col])
            if mask.sum() > 0:
                metric_collection = transformed_metrics[key]
                metric_collection.update(predictions[mask, col], targets[mask, col])

        # Update combined transformed metrics
        combined_mask = ~torch.isnan(targets).any(dim=1)
        if combined_mask.sum() > 0:
            transformed_combined = getattr(self, f"{stage}_transformed_combined")
            transformed_combined.update(
                predictions[combined_mask].reshape(-1),
                targets[combined_mask].reshape(-1),
            )

        inv_predictions = predictions.clone()
        if hasattr(self, "inverse_transform") and self.inverse_transform is not None:
            # Create a temp HeteroData object with predictions
            temp_data = HeteroData()
            temp_data["gene"] = {
                "fitness": predictions[:, 0].clone(),
                "gene_interaction": predictions[:, 1].clone(),
            }

            # Apply the proper inverse transform
            inv_data = self.inverse_transform(temp_data)

            # Extract the inversed predictions
            inv_predictions = torch.stack(
                [inv_data["gene"]["fitness"], inv_data["gene"]["gene_interaction"]],
                dim=1,
            )

        # Update metrics - important for val/fitness/MSE
        for key, col in zip(["fitness", "gene_interaction"], [0, 1]):
            mask = ~torch.isnan(orig_targets[:, col])
            if mask.sum() > 0:
                metric_collection = getattr(self, f"{stage}_metrics")[key]
                metric_collection.update(
                    inv_predictions[mask, col], orig_targets[mask, col]
                )

        # Update combined metrics (across all dimensions)
        combined_mask = ~torch.isnan(orig_targets).any(dim=1)
        if combined_mask.sum() > 0:
            combined_metrics = getattr(self, f"{stage}_combined_metrics")
            # Reshape to flatten all predictions and targets
            combined_metrics.update(
                inv_predictions[combined_mask].reshape(-1),
                orig_targets[combined_mask].reshape(-1),
            )

        if (
            stage == "train"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            current_count = sum(t.size(0) for t in self.train_samples["true_values"])
            if current_count < self.hparams.plot_sample_ceiling:
                remaining = self.hparams.plot_sample_ceiling - current_count
                if batch_size > remaining:
                    idx = torch.randperm(batch_size)[:remaining]
                    self.train_samples["true_values"].append(orig_targets[idx].detach())
                    self.train_samples["predictions"].append(
                        inv_predictions[idx].detach()
                    )
                    if "z_p" in representations:
                        self.train_samples["latents"]["z_p"].append(
                            representations["z_p"][idx].detach()
                        )
                else:
                    self.train_samples["true_values"].append(orig_targets.detach())
                    self.train_samples["predictions"].append(inv_predictions.detach())
                    if "z_p" in representations:
                        self.train_samples["latents"]["z_p"].append(
                            representations["z_p"].detach()
                        )
        elif (
            stage == "val"
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
        ):
            # Only collect validation samples on epochs we'll plot
            self.val_samples["true_values"].append(orig_targets.detach())
            self.val_samples["predictions"].append(inv_predictions.detach())
            if "z_p" in representations:
                self.val_samples["latents"]["z_p"].append(
                    representations["z_p"].detach()
                )

        return loss, predictions, orig_targets

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
        latents = {k: torch.cat(v, dim=0) for k, v in samples["latents"].items()}
        max_samples = self.hparams.plot_sample_ceiling
        if true_values.size(0) > max_samples:
            idx = torch.randperm(true_values.size(0))[:max_samples]
            true_values = true_values[idx]
            predictions = predictions[idx]
            for key in latents:
                latents[key] = latents[key][idx]
        vis = Visualization(
            base_dir=self.trainer.default_root_dir, max_points=max_samples
        )
        loss_name = (
            self.hparams.loss_func.__class__.__name__
            if self.hparams.loss_func is not None
            else "Loss"
        )
        vis.visualize_model_outputs(
            predictions,
            true_values,
            latents,
            loss_name,
            self.current_epoch,
            None,
            stage=stage,
        )
        # Log oversmoothing metrics on latent spaces.
        if "z_p" in latents:
            smoothness_zp = VisGraphDegen.compute_smoothness(latents["z_p"])
            wandb.log({f"{stage}/oversmoothing_zp": smoothness_zp.item()})
        # Log additional box plots for fitness and gene interaction scores.
        if true_values.dim() > 1 and predictions.dim() > 1:
            if torch.any(~torch.isnan(true_values[:, 0])):
                fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
                wandb.log({f"{stage}/fitness_box_plot": wandb.Image(fig_fitness)})
                plt.close(fig_fitness)
            if torch.any(~torch.isnan(true_values[:, 1])):
                fig_gi = genetic_interaction_score.box_plot(
                    true_values[:, 1], predictions[:, 1]
                )
                wandb.log({f"{stage}/gene_interaction_box_plot": wandb.Image(fig_gi)})
                plt.close(fig_gi)

    def on_train_epoch_end(self):
        # Log training metrics for individual labels
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        # Compute and log combined metrics
        combined_metrics = self._compute_metrics_safely(self.train_combined_metrics)
        for name, value in combined_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_combined_metrics.reset()

        # Compute and log transformed metrics for individual labels
        for metric_name, metric_dict in self.train_transformed_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        # Compute and log transformed combined metrics
        transformed_combined_metrics = self._compute_metrics_safely(
            self.train_transformed_combined
        )
        for name, value in transformed_combined_metrics.items():
            self.log(name, value, sync_dist=True)
        self.train_transformed_combined.reset()

        # Plot training samples only on the final epoch
        if (
            self.current_epoch + 1
        ) % self.hparams.plot_every_n_epochs == 0 and self.train_samples["true_values"]:
            self._plot_samples(self.train_samples, "train_sample")
            # Reset the sample containers
            self.train_samples = {
                "true_values": [],
                "predictions": [],
                "latents": {"z_p": []},
            }

    def on_train_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.train_samples = {
                "true_values": [],
                "predictions": [],
                "latents": {"z_p": []},
            }

    def on_validation_epoch_start(self):
        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0:
            self.val_samples = {
                "true_values": [],
                "predictions": [],
                "latents": {"z_p": []},
            }

    def on_validation_epoch_end(self):
        # Log validation metrics for individual labels
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        # Compute and log combined metrics
        combined_metrics = self._compute_metrics_safely(self.val_combined_metrics)
        for name, value in combined_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_combined_metrics.reset()

        # Compute and log transformed metrics for individual labels
        for metric_name, metric_dict in self.val_transformed_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()

        # Compute and log transformed combined metrics
        transformed_combined_metrics = self._compute_metrics_safely(
            self.val_transformed_combined
        )
        for name, value in transformed_combined_metrics.items():
            self.log(name, value, sync_dist=True)
        self.val_transformed_combined.reset()

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % self.hparams.plot_every_n_epochs == 0
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            # Reset the sample containers
            self.val_samples = {
                "true_values": [],
                "predictions": [],
                "latents": {"z_p": []},
            }

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
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
