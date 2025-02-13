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
        self.loss_func = loss_func

        reg_metrics = MetricCollection(
            {
                "MSE": MeanSquaredError(squared=True),
                "RMSE": MeanSquaredError(squared=False),
                "Pearson": PearsonCorrCoef(),
            }
        )
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

        self.true_values = []
        self.predictions = []
        self.latents = {"z_p": [], "z_i": []}
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
        fitness_vals = batch["gene"].fitness.view(-1, 1)
        gene_interaction_vals = batch["gene"].gene_interaction.view(-1, 1)
        targets = torch.cat([fitness_vals, gene_interaction_vals], dim=1)
        loss, loss_dict = self.loss_func(
            predictions,
            targets,
            representations["z_w"],
            representations["z_p"],
            representations["z_i"],
        )
        # Log all loss components with stage prefix.
        keys = [
            ("loss", loss),
            ("fitness_loss", loss_dict["mse_dim_losses"][0]),
            ("gene_interaction_loss", loss_dict["mse_dim_losses"][1]),
            ("mse_loss", loss_dict["mse_loss"]),
            ("dist_loss", loss_dict["dist_loss"]),
            ("supcr_loss", loss_dict["supcr_loss"]),
            ("cell_loss", loss_dict["cell_loss"]),
            ("total_loss", loss_dict["total_loss"]),
        ]
        for key, value in keys:
            self.log(f"{stage}/{key}", value, batch_size=batch_size, sync_dist=True)
        extra_keys = [
            "weighted_mse",
            "weighted_dist",
            "weighted_supcr",
            "weighted_cell",
            "total_weighted",
            "norm_weighted_mse",
            "norm_weighted_dist",
            "norm_weighted_supcr",
            "norm_weighted_cell",
            "norm_unweighted_mse",
            "norm_unweighted_dist",
            "norm_unweighted_supcr",
            "norm_unweighted_cell",
        ]
        for key in extra_keys:
            if key in loss_dict:
                self.log(
                    f"{stage}/{key}",
                    loss_dict[key],
                    batch_size=batch_size,
                    sync_dist=True,
                )
        if stage in ["val", "test"]:
            self.true_values.append(targets.detach())
            self.predictions.append(predictions.detach())
            if "z_p" in representations:
                self.latents["z_p"].append(representations["z_p"].detach())
            if "z_i" in representations:
                self.latents["z_i"].append(representations["z_i"].detach())
        return loss, predictions, targets

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

    def on_train_epoch_end(self):
        for metric_name, metric_dict in self.train_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()
        # Save visual artifacts only on the final training epoch.
        if self.current_epoch == self.trainer.max_epochs - 1:
            if self.true_values and self.predictions:
                true_values = torch.cat(self.true_values, dim=0)
                predictions = torch.cat(self.predictions, dim=0)
                latents = {k: torch.cat(v, dim=0) for k, v in self.latents.items()}
                vis = Visualization(
                    base_dir=self.trainer.default_root_dir,
                    max_points=self.hparams.get("plot_sample_ceiling", 1000),
                )
                loss_name = (
                    self.hparams.loss_func.__class__.__name__
                    if self.hparams.loss_func is not None
                    else "Loss"
                )
                ts = timestamp()
                vis.visualize_model_outputs(
                    predictions,
                    true_values,
                    latents,
                    loss_name,
                    self.current_epoch,
                    ts,
                    stage="train",
                )
                vis.log_sample_metrics(predictions, true_values, stage="train")
        self.true_values = []
        self.predictions = []
        self.latents = {"z_p": [], "z_i": []}

    def on_validation_epoch_end(self):
        for metric_name, metric_dict in self.val_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()
        if (
            not self.trainer.sanity_checking
            and self.current_epoch == self.trainer.max_epochs - 1
        ):
            if self.true_values and self.predictions:
                true_values = torch.cat(self.true_values, dim=0)
                predictions = torch.cat(self.predictions, dim=0)
                latents = {k: torch.cat(v, dim=0) for k, v in self.latents.items()}
                vis = Visualization(
                    base_dir=self.trainer.default_root_dir,
                    max_points=self.hparams.get("plot_sample_ceiling", 1000),
                )
                loss_name = (
                    self.hparams.loss_func.__class__.__name__
                    if self.hparams.loss_func is not None
                    else "Loss"
                )
                ts = timestamp()
                vis.visualize_model_outputs(
                    predictions,
                    true_values,
                    latents,
                    loss_name,
                    self.current_epoch,
                    ts,
                    stage="val",
                )
                vis.log_sample_metrics(predictions, true_values, stage="val")
            # Also log fitness and gene interaction box plots.
            true_values = (
                torch.cat(self.true_values, dim=0) if self.true_values else None
            )
            predictions = (
                torch.cat(self.predictions, dim=0) if self.predictions else None
            )
            if true_values is not None and predictions is not None:
                from torchcell.viz import fitness, genetic_interaction_score

                if torch.any(~torch.isnan(true_values[:, 0])):
                    fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
                    wandb.log({"val/fitness_box_plot": wandb.Image(fig_fitness)})
                    plt.close(fig_fitness)
                if torch.any(~torch.isnan(true_values[:, 1])):
                    fig_gi = genetic_interaction_score.box_plot(
                        true_values[:, 1], predictions[:, 1]
                    )
                    wandb.log({"val/gene_interaction_box_plot": wandb.Image(fig_gi)})
                    plt.close(fig_gi)
        self.true_values = []
        self.predictions = []
        self.latents = {"z_p": [], "z_i": []}

    def on_test_epoch_end(self):
        for metric_name, metric_dict in self.test_metrics.items():
            computed_metrics = self._compute_metrics_safely(metric_dict)
            for name, value in computed_metrics.items():
                self.log(name, value, sync_dist=True)
            metric_dict.reset()
        if not self.trainer.sanity_checking and self.true_values and self.predictions:
            true_values = torch.cat(self.true_values, dim=0)
            predictions = torch.cat(self.predictions, dim=0)
            latents = {k: torch.cat(v, dim=0) for k, v in self.latents.items()}
            vis = Visualization(
                base_dir=self.trainer.default_root_dir,
                max_points=self.hparams.get("plot_sample_ceiling", 1000),
            )
            loss_name = (
                self.hparams.loss_func.__class__.__name__
                if self.hparams.loss_func is not None
                else "Loss"
            )
            ts = timestamp()
            vis.visualize_model_outputs(
                predictions,
                true_values,
                latents,
                loss_name,
                self.current_epoch,
                ts,
                stage="test",
            )
            vis.log_sample_metrics(predictions, true_values, stage="test")
            from torchcell.viz import fitness, genetic_interaction_score

            if torch.any(~torch.isnan(true_values[:, 0])):
                fig_fitness = fitness.box_plot(true_values[:, 0], predictions[:, 0])
                wandb.log({"test/fitness_box_plot": wandb.Image(fig_fitness)})
                plt.close(fig_fitness)
            if torch.any(~torch.isnan(true_values[:, 1])):
                fig_gi = genetic_interaction_score.box_plot(
                    true_values[:, 1], predictions[:, 1]
                )
                wandb.log({"test/gene_interaction_box_plot": wandb.Image(fig_gi)})
                plt.close(fig_gi)
        self.true_values = []
        self.predictions = []
        self.latents = {"z_p": [], "z_i": []}

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
