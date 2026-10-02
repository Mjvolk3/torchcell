"""Lightning regression tasks for heterogeneous-cell gene interaction models.

``RegressionTask`` trains a model whose forward returns ``(predictions,
representations)`` for one ``gene_interaction`` value per genotype.
``DiffusionRegressionTask`` is the same task for a diffusion decoder: it differs only
in the stage loss (the diffusion loss in training, the MSE of the sampled predictions
at validation and test) and in not scoring training-mode predictions, which the
diffusion model returns as all-zero placeholders. Everything else, one
``_shared_step`` included, is shared (issue #614).
"""

import logging
import re
from typing import Any, Protocol, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch_geometric.data import HeteroData
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef

from torchcell.losses.diffusion_loss import DiffusionLoss
from torchcell.losses.isomorphic_cell_loss import ICLoss
from torchcell.losses.mle_dist_supcr import MleDistSupCR
from torchcell.losses.mle_wasserstein import MleWassSupCR
from torchcell.losses.point_dist_graph_reg import PointDistGraphReg
from torchcell.viz import genetic_interaction_score
from torchcell.viz.visual_graph_degen import VisGraphDegen
from torchcell.viz.visual_regression import Visualization

log = logging.getLogger(__name__)

# The metric a ReduceLROnPlateau scheduler is stepped on, once per validation epoch.
PLATEAU_MONITOR = "val/gene_interaction/MSE"
SCHEDULER_TYPES = (
    "CosineAnnealingWarmupRestarts",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
)
# Losses called (predictions, targets, z_p); the MLE losses also take the epoch.
Z_P_LOSSES = (ICLoss, DiffusionLoss, MleDistSupCR, MleWassSupCR)
EPOCH_LOSSES = (MleDistSupCR, MleWassSupCR)


class _HParams(Protocol):
    """Typed view of the hyperparameters saved by ``save_hyperparameters``."""

    optimizer_config: dict[str, Any]
    lr_scheduler_config: dict[str, Any] | None
    clip_grad_norm: bool
    clip_grad_norm_max_norm: float
    plot_sample_ceiling: int
    plot_every_n_epochs: int


def scheduler_type(config: dict[str, Any]) -> str:
    """Return the scheduler ``type`` of an lr scheduler config, refusing a missing or
    unknown one by name (a misspelled type used to become a plateau scheduler).
    """
    if "type" not in config:
        raise ValueError(
            "lr_scheduler_config has no 'type'; expected one of "
            f"{list(SCHEDULER_TYPES)} (pass lr_scheduler_config=None for no scheduler)"
        )
    kind = config["type"]
    if kind not in SCHEDULER_TYPES:
        raise ValueError(
            f"lr_scheduler_config type {kind!r} is not one of {list(SCHEDULER_TYPES)}"
        )
    return str(kind)


def normalize_accumulation_schedule(
    schedule: dict[Any, Any] | None,
) -> dict[int, int] | None:
    """Return the schedule with integer epoch keys in ascending order.

    A key is an integer epoch >= 0, given as an ``int`` or as a string of digits:
    ``wandb.config`` returns every mapping key as a string, so ``{0: 16}`` in a config
    reaches the task as ``{"0": 16}``. Anything else is refused by name, notably
    ``"0:16"``, which is what YAML reads from the flow mapping ``{0:16}`` written
    without a space. A value is a positive integer step count.
    """
    if schedule is None:
        return None
    normalized: dict[int, int] = {}
    for key, steps in schedule.items():
        is_int = isinstance(key, int) and not isinstance(key, bool) and key >= 0
        is_digits = isinstance(key, str) and re.fullmatch(r"[0-9]+", key) is not None
        if not (is_int or is_digits):
            hint = (
                "; a YAML flow mapping needs a space after the colon: write {0: 16}, "
                "not {0:16}"
                if isinstance(key, str) and ":" in key
                else ""
            )
            raise ValueError(
                f"grad_accumulation_schedule key {key!r} is not an integer epoch{hint}"
            )
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(
                f"grad_accumulation_schedule[{key!r}] = {steps!r} is not a positive "
                "integer number of accumulation steps"
            )
        epoch = int(key)
        if epoch in normalized:
            raise ValueError(
                f"grad_accumulation_schedule names epoch {epoch} twice: {schedule!r}"
            )
        normalized[epoch] = steps
    return dict(sorted(normalized.items()))


def accumulation_steps(schedule: dict[int, int], epoch: int) -> int:
    """Steps of the last threshold reached (``epoch >= threshold``); 1 before any."""
    steps = 1
    for threshold, value in schedule.items():
        if epoch >= threshold:
            steps = value
    return steps


def _as_column(values: torch.Tensor) -> torch.Tensor:
    """A 0-dim tensor becomes [1, 1], a vector [n, 1]; a 2-dim tensor is kept."""
    if values.dim() == 0:
        return values.unsqueeze(0).unsqueeze(0)
    if values.dim() == 1:
        return values.unsqueeze(1)
    return values


def _empty_samples() -> dict[str, Any]:
    return {"true_values": [], "predictions": [], "latents": {}}


class RegressionTask(L.LightningModule):
    """Lightning task training a model to predict gene interaction scores."""

    # Whether train-stage predictions are scored (metrics and plot buffers).
    scores_train_predictions = True

    train_metrics: MetricCollection
    val_metrics: MetricCollection
    test_metrics: MetricCollection
    train_transformed_metrics: MetricCollection
    val_transformed_metrics: MetricCollection
    test_transformed_metrics: MetricCollection
    _cell_graph_device: torch.device

    @property
    def hp(self) -> _HParams:
        """Return ``self.hparams`` as a precisely typed view (runtime no-op cast)."""
        return cast(_HParams, self.hparams)

    def __init__(
        self,
        model: nn.Module,
        cell_graph: torch.Tensor,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any] | None,
        batch_size: int | None = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        plot_sample_ceiling: int = 1000,
        plot_every_n_epochs: int = 10,
        loss_func: nn.Module | None = None,
        grad_accumulation_schedule: dict[Any, int] | None = None,
        device: str = "cuda",
        inverse_transform: nn.Module | None = None,
        execution_mode: str = "training",  # "training" or "dataloader_profiling"
    ) -> None:
        """Store the model, cell graph, loss, and set up metrics and accumulators.

        Args:
            model: The regression model to train.
            cell_graph: Reference cell graph (cloned for pin_memory safety).
            optimizer_config: Optimizer configuration dict.
            lr_scheduler_config: Learning-rate scheduler configuration dict with a
                ``type`` from ``SCHEDULER_TYPES``, or None for no scheduler.
            batch_size: Not used and not saved as a hyperparameter: every logged
                batch size is the batch's genotype count. Accepted because the
                experiment scripts and older checkpoints pass it.
            clip_grad_norm: Whether to clip gradient norms.
            clip_grad_norm_max_norm: Max norm for gradient clipping.
            plot_sample_ceiling: Max number of samples to accumulate for plots.
            plot_every_n_epochs: Epoch interval for rendering plots.
            loss_func: Loss module used during training.
            grad_accumulation_schedule: Optional epoch-to-steps accumulation map,
                normalized by ``normalize_accumulation_schedule``.
            device: Not used and not saved as a hyperparameter (Lightning places the
                task); accepted for the same reason as ``batch_size``.
            inverse_transform: Optional module mapping predictions back to raw space.
            execution_mode: "training" or "dataloader_profiling".
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "batch_size", "device"])
        if lr_scheduler_config is not None:
            scheduler_type(lr_scheduler_config)
        self.model = model
        self.execution_mode = execution_mode
        # Clone cell_graph to avoid modifying the dataset's original cell_graph
        # This is necessary for pin_memory compatibility in DataLoader
        self.cell_graph = cell_graph.clone()
        self.inverse_transform = inverse_transform
        self.loss_func = loss_func

        self.grad_accumulation_schedule = normalize_accumulation_schedule(
            grad_accumulation_schedule
        )
        self.current_accumulation_steps = (
            1
            if self.grad_accumulation_schedule is None
            else accumulation_steps(self.grad_accumulation_schedule, 0)
        )
        self._announced_unscored_train = False

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
        self.train_samples: dict[str, Any] = _empty_samples()
        self.val_samples: dict[str, Any] = _empty_samples()
        self.test_samples: dict[str, Any] = _empty_samples()
        self.automatic_optimization = False

    def _get_batch_size(self, batch: HeteroData) -> int:
        """The number of genotypes in a collated batch: its ``num_graphs``.

        Both collaters the callers use (PyG's and ``LazyCollater``) write
        ``num_graphs``; node rows (``gene.x``) and perturbed-gene counts are not
        genotype counts (issues #567, #596), so a batch without it is refused.
        """
        if not hasattr(batch, "num_graphs"):
            raise ValueError(
                f"cannot size a {type(batch).__name__} batch: its genotype count is "
                "the collated batch's num_graphs, which this batch does not carry"
            )
        return int(batch.num_graphs)

    # return: model output tuple, dynamic from nn.Module
    def forward(self, batch: HeteroData) -> Any:
        """Run the model on a batch and return predictions and any auxiliary outputs."""
        self._place_cell_graph(batch)
        # Return all outputs from the model
        return self.model(self.cell_graph, batch)

    def _batch_device(self, batch: HeteroData) -> torch.device:
        """The device of the batch's first gene tensor, else of the model."""
        device: torch.device
        if hasattr(batch["gene"], "x"):
            device = batch["gene"].x.device
        elif hasattr(batch["gene"], "perturbation_indices"):
            device = batch["gene"].perturbation_indices.device
        elif hasattr(batch["gene"], "phenotype_values"):
            device = batch["gene"].phenotype_values.device
        else:
            device = next(self.model.parameters()).device
        return device

    def _place_cell_graph(self, batch: HeteroData) -> torch.device:
        """Move the cell graph to the batch's device (cached) and return the device."""
        batch_device = self._batch_device(batch)
        if (
            not hasattr(self, "_cell_graph_device")
            or self._cell_graph_device != batch_device
        ):
            self.cell_graph = self.cell_graph.to(batch_device)
            self._cell_graph_device = batch_device
        return batch_device

    def _ensure_no_unused_params_loss(self) -> torch.Tensor | int:
        """Add a dummy loss to ensure all parameters are used in backward pass."""
        dummy_loss: torch.Tensor | int = 0
        for param in self.model.parameters():
            if param.requires_grad and param.grad is None:
                dummy_loss = dummy_loss + 0.0 * param.sum()
        return dummy_loss

    def _call_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        representations: dict[str, Any],
    ) -> Any:
        """Call ``loss_func`` with the arguments its class takes.

        ``PointDistGraphReg`` gets the representations and the epoch; ``ICLoss`` and
        ``DiffusionLoss`` get ``z_p``; the MLE losses get ``z_p`` and the epoch; any
        other loss (``LogCoshLoss``, ``nn.MSELoss``, ...) gets ``(predictions,
        targets)``. A loss that needs ``z_p`` from a model that returns none is
        refused by name.
        """
        if self.loss_func is None:
            raise ValueError("No loss function provided")
        if isinstance(self.loss_func, PointDistGraphReg):
            return self.loss_func(
                predictions, targets, representations, epoch=self.current_epoch
            )
        if isinstance(self.loss_func, Z_P_LOSSES):
            z_p = representations.get("z_p")
            if z_p is None:
                raise ValueError(
                    f"{type(self.loss_func).__name__} needs representations['z_p'], "
                    "which the model did not return"
                )
            if isinstance(self.loss_func, EPOCH_LOSSES):
                return self.loss_func(
                    predictions, targets, z_p, epoch=self.current_epoch
                )
            return self.loss_func(predictions, targets, z_p)
        return self.loss_func(predictions, targets)

    def _log_loss_components(
        self, stage: str, components: Any, batch_size: int
    ) -> None:
        """Log a loss's component dict: a one-element tensor as a scalar, a longer
        tensor element by element (``{key}_{i}``), a number as is. Empty tensors,
        other value types and a non-dict tail log nothing.
        """
        if not isinstance(components, dict):
            return
        for key, value in components.items():
            if isinstance(value, torch.Tensor):
                flat = value.reshape(-1)
                if flat.numel() == 1:
                    self.log(
                        f"{stage}/{key}",
                        flat[0].item(),
                        batch_size=batch_size,
                        sync_dist=True,
                    )
                else:
                    for i in range(flat.numel()):
                        self.log(
                            f"{stage}/{key}_{i}",
                            flat[i].item(),
                            batch_size=batch_size,
                            sync_dist=True,
                        )
            elif isinstance(value, (int, float)):
                self.log(f"{stage}/{key}", value, batch_size=batch_size, sync_dist=True)

    def _stage_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        representations: dict[str, Any],
        stage: str,
        batch_size: int,
    ) -> torch.Tensor:
        """The loss of ``loss_func`` at every stage, its components logged."""
        output = self._call_loss(predictions, targets, representations)
        if isinstance(output, tuple):
            self._log_loss_components(
                stage, output[1] if len(output) > 1 else {}, batch_size
            )
            loss: torch.Tensor = output[0]
            return loss
        bare: torch.Tensor = output
        return bare

    def _shared_step(
        self, batch: HeteroData, batch_idx: int, stage: str = "train"
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size = self._get_batch_size(batch)
        # DataLoader profiling mode: Skip model forward, create dummy loss
        if self.execution_mode == "dataloader_profiling":
            batch_device = self._place_cell_graph(batch)

            # Create trivial loss that touches ALL model parameters (required for DDP)
            # This ensures no "unused parameters" error in DDP mode
            loss = torch.zeros((), device=batch_device, requires_grad=True)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss = loss + (param * 0.0).sum()

            self.log(
                f"{stage}/dataloader_profile_loss",
                loss,
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log(
                f"{stage}/dataloader_profile_batch_size",
                float(batch_size),
                batch_size=batch_size,
                sync_dist=True,
            )
            return loss, None, None

        has_original = hasattr(batch["gene"], "phenotype_values_original")
        if has_original and self.inverse_transform is None:
            raise ValueError(
                "batch carries gene.phenotype_values_original but the task has no "
                "inverse_transform; model-unit predictions would be scored against "
                "original-unit targets"
            )

        predictions, representations = self(batch)
        predictions = _as_column(predictions)
        if predictions.size(0) != batch_size:
            raise ValueError(
                f"the model returned {predictions.size(0)} prediction rows for a "
                f"batch of {batch_size} genotypes"
            )

        # Targets in COO format: one gene_interaction value per genotype
        gene_interaction_vals = _as_column(batch["gene"].phenotype_values)
        gene_interaction_orig = (
            _as_column(batch["gene"].phenotype_values_original)
            if has_original
            else gene_interaction_vals
        )
        z_p = representations.get("z_p")

        loss = self._stage_loss(
            predictions, gene_interaction_vals, representations, stage, batch_size
        )

        # Add graph regularization loss if present (for transformer models)
        # IMPORTANT: Skip if using PointDistGraphReg, as it already includes graph_reg in total
        if "graph_reg_loss" in representations and not isinstance(
            self.loss_func, PointDistGraphReg
        ):
            graph_reg_loss = representations["graph_reg_loss"]
            loss = loss + graph_reg_loss
            self.log(
                f"{stage}/graph_reg_loss",
                graph_reg_loss,
                batch_size=batch_size,
                sync_dist=True,
            )

        # Add dummy loss for unused parameters
        dummy_loss = self._ensure_no_unused_params_loss()
        loss = loss + dummy_loss

        self.log(f"{stage}/loss", loss, batch_size=batch_size, sync_dist=True)

        if z_p is not None:
            z_p_norm = z_p.norm(p=2, dim=-1).mean()
            self.log(
                f"{stage}/z_p_norm", z_p_norm, batch_size=batch_size, sync_dist=True
            )

        if "gate_weights" in representations:
            # Average gate weights across batch
            avg_gate_weights = representations["gate_weights"].mean(dim=0)
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

        if stage == "train" and not self.scores_train_predictions:
            if not self._announced_unscored_train:
                log.info(
                    "%s does not score train-stage predictions: "
                    "train/gene_interaction/* and train/transformed/gene_interaction/* "
                    "are not logged and no train samples are plotted",
                    type(self).__name__,
                )
                self._announced_unscored_train = True
            return loss, predictions, gene_interaction_orig

        # Update transformed metrics
        mask = ~torch.isnan(gene_interaction_vals)
        if mask.sum() > 0:
            transformed_metrics = getattr(self, f"{stage}_transformed_metrics")
            transformed_metrics.update(
                predictions[mask].view(-1), gene_interaction_vals[mask].view(-1)
            )

        inv_predictions = self._invert(predictions)

        # Update metrics with original scale values
        mask = ~torch.isnan(gene_interaction_orig)
        if mask.sum() > 0:
            metrics = getattr(self, f"{stage}_metrics")
            metrics.update(
                inv_predictions[mask].view(-1), gene_interaction_orig[mask].view(-1)
            )

        # Collect samples for visualization
        plot_epoch = (self.current_epoch + 1) % self.hp.plot_every_n_epochs == 0
        if stage in ("train", "val") and plot_epoch:
            self._keep_samples(
                getattr(self, f"{stage}_samples"),
                gene_interaction_orig,
                inv_predictions,
                z_p,
                self.hp.plot_sample_ceiling,
            )
        elif stage == "test":
            # Test runs once: keep every batch (the ceiling applies when plotting)
            self._keep_samples(
                self.test_samples, gene_interaction_orig, inv_predictions, z_p, None
            )

        return loss, predictions, gene_interaction_orig

    def _invert(self, predictions: torch.Tensor) -> torch.Tensor:
        """Predictions in original units through ``inverse_transform`` (a copy if none)."""
        if self.inverse_transform is None:
            return predictions.clone()
        # A one-type COO object holding the predictions
        n = predictions.size(0)
        device = predictions.device
        temp_data = HeteroData()
        temp_data["gene"].phenotype_values = predictions.squeeze()
        temp_data["gene"].phenotype_type_indices = torch.zeros(
            n, dtype=torch.long, device=device
        )
        temp_data["gene"].phenotype_sample_indices = torch.arange(n, device=device)
        temp_data["gene"].phenotype_types = ["gene_interaction"]

        inv_gene_int = self.inverse_transform(temp_data)["gene"]["phenotype_values"]
        # A non-tensor would leave the original-scale metrics computed on the
        # transformed predictions, so it is refused rather than skipped (as #534).
        if not isinstance(inv_gene_int, torch.Tensor):
            raise TypeError(
                f"inverse_transform {type(self.inverse_transform).__name__} "
                "returned phenotype_values of type "
                f"{type(inv_gene_int).__name__}; expected torch.Tensor"
            )
        return _as_column(inv_gene_int)

    @staticmethod
    def _keep_samples(
        buffer: dict[str, Any],
        true_values: torch.Tensor,
        predictions: torch.Tensor,
        z_p: torch.Tensor | None,
        ceiling: int | None,
    ) -> None:
        """Append detached rows to a plot buffer, a random subset up to ``ceiling``."""
        rows = true_values.size(0)
        if ceiling is not None:
            current = sum(t.size(0) for t in buffer["true_values"])
            if current >= ceiling:
                return
            if rows > ceiling - current:
                idx = torch.randperm(rows)[: ceiling - current]
                true_values, predictions = true_values[idx], predictions[idx]
                z_p = None if z_p is None else z_p[idx]
        buffer["true_values"].append(true_values.detach())
        buffer["predictions"].append(predictions.detach())
        if z_p is not None:
            buffer["latents"].setdefault("z_p", []).append(z_p.detach())

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a manual-optimization training step and log loss and metrics."""
        loss, _, _ = self._shared_step(batch, batch_idx, "train")

        # Model profiling mode: Skip optimizer step to isolate model compute
        if self.execution_mode == "model_profiling":
            return loss

        batch_size = self._get_batch_size(batch)

        # Normal training: Run optimizer
        if self.grad_accumulation_schedule is not None:
            loss = loss / self.current_accumulation_steps
        opt = cast(LightningOptimizer, self.optimizers())
        self.manual_backward(loss)
        if (
            self.grad_accumulation_schedule is None
            or (batch_idx + 1) % self.current_accumulation_steps == 0
        ):
            if self.hp.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.hp.clip_grad_norm_max_norm
                )
            opt.step()
            opt.zero_grad()
        self.log(
            "learning_rate",
            opt.param_groups[0]["lr"],
            batch_size=batch_size,
            sync_dist=True,
        )
        if self.grad_accumulation_schedule is not None:
            # Genotypes behind one optimizer step: this rank's batch, times the
            # accumulation steps, times the ranks (equal per-rank batches assumed).
            effective_batch_size = (
                batch_size * self.current_accumulation_steps * self.trainer.world_size
            )
            self.log(
                "effective_batch_size",
                effective_batch_size,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a validation step, accumulating predictions and logging metrics."""
        loss, _, _ = self._shared_step(batch, batch_idx, "val")

        # Defragment GPU memory every 50 batches to prevent OOM from fragmentation
        if batch_idx > 0 and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        return loss

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Run a test step, accumulating predictions and logging metrics."""
        loss, _, _ = self._shared_step(batch, batch_idx, "test")
        return loss

    def _log_metrics(self, collection: MetricCollection) -> dict[str, torch.Tensor]:
        """Compute, log (``sync_dist=True``) and reset a metric collection.

        A metric error propagates. An epoch with no update logs what torchmetrics
        computes for it (NaN, with its warning).
        """
        computed: dict[str, torch.Tensor] = collection.compute()
        for name, value in computed.items():
            self.log(name, value, sync_dist=True)
        collection.reset()
        return computed

    def _plot_samples(self, samples: dict[str, Any], stage: str) -> None:
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

        max_samples = self.hp.plot_sample_ceiling
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

    def _active_scheduler(self) -> LRScheduler | ReduceLROnPlateau | None:
        """The configured scheduler (the first if Lightning returns a list)."""
        sch = self.lr_schedulers()
        if isinstance(sch, list):
            return sch[0] if sch else None
        return sch

    def on_train_epoch_end(self) -> None:
        """Log epoch-level training metrics, plot, and step an epoch scheduler.

        Under manual optimization Lightning steps no scheduler, so this hook steps a
        warmup or cosine scheduler once per epoch. A ReduceLROnPlateau scheduler is
        not stepped here: it needs its metric and is stepped in
        ``on_validation_epoch_end``.
        """
        if self.scores_train_predictions:
            self._log_metrics(self.train_metrics)
            self._log_metrics(self.train_transformed_metrics)

        # Plot training samples
        if (
            self.current_epoch + 1
        ) % self.hp.plot_every_n_epochs == 0 and self.train_samples["true_values"]:
            self._plot_samples(self.train_samples, "train_sample")
            self.train_samples = _empty_samples()

        sch = self._active_scheduler()
        if sch is not None and not isinstance(sch, ReduceLROnPlateau):
            sch.step()

        # CRITICAL: Clear GPU memory at end of training epoch
        # This ensures validation starts with maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_epoch_start(self) -> None:
        """Update gradient accumulation and reset training sample accumulators."""
        if self.grad_accumulation_schedule is not None:
            self.current_accumulation_steps = accumulation_steps(
                self.grad_accumulation_schedule, self.current_epoch
            )
            print(
                f"Epoch {self.current_epoch}: Using gradient accumulation steps = {self.current_accumulation_steps}"
            )

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hp.plot_every_n_epochs == 0:
            self.train_samples = _empty_samples()

    def on_validation_epoch_start(self) -> None:
        """Reset validation sample accumulators at the start of the epoch."""
        # CRITICAL: Aggressively clear GPU memory before validation starts
        # This prevents OOM when transitioning from training to validation
        # Training state (optimizer, gradients, cached activations) can fragment memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Synchronize to ensure all pending operations complete before validation
            torch.cuda.synchronize()

        # Clear sample containers at the start of epochs where we'll collect samples
        if (self.current_epoch + 1) % self.hp.plot_every_n_epochs == 0:
            self.val_samples = _empty_samples()

    def on_test_epoch_start(self) -> None:
        """Reset test sample accumulators at the start of the epoch."""
        # Always clear sample containers for test (test runs only once)
        self.test_samples = _empty_samples()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics, step a plateau scheduler, plot, and reset.

        A ReduceLROnPlateau scheduler is stepped here, once per validation epoch
        outside the sanity check, on ``PLATEAU_MONITOR`` (``val/gene_interaction/MSE``)
        as just computed: validation runs inside the training epoch, so this is where
        the monitor is fresh, and an epoch without validation does not step it. A
        validation epoch whose collection received no rows is refused rather than
        stepped on NaN.
        """
        val_rows = self.val_metrics["MSE"].update_count
        computed = self._log_metrics(self.val_metrics)
        sch = self._active_scheduler()
        if not self.trainer.sanity_checking and isinstance(sch, ReduceLROnPlateau):
            if val_rows == 0:
                raise ValueError(
                    f"ReduceLROnPlateau monitors {PLATEAU_MONITOR}, which received no "
                    "validation rows this epoch (every target NaN or no batch); "
                    "refusing to step it on NaN"
                )
            sch.step(computed[PLATEAU_MONITOR])
        self._log_metrics(self.val_transformed_metrics)

        # Plot validation samples
        if (
            not self.trainer.sanity_checking
            and (self.current_epoch + 1) % self.hp.plot_every_n_epochs == 0
            and self.val_samples["true_values"]
        ):
            self._plot_samples(self.val_samples, "val_sample")
            self.val_samples = _empty_samples()

    def on_test_epoch_end(self) -> None:
        """Log test metrics, render plots, and reset accumulators."""
        self._log_metrics(self.test_metrics)
        self._log_metrics(self.test_transformed_metrics)

        # Plot test samples
        if self.test_samples["true_values"]:
            self._plot_samples(self.test_samples, "test_sample")
            self.test_samples = _empty_samples()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the optimizer and learning-rate scheduler from config."""
        optimizer_class = getattr(torch.optim, self.hp.optimizer_config["type"])
        optimizer_params = {
            k: v for k, v in self.hp.optimizer_config.items() if k != "type"
        }
        if "learning_rate" in optimizer_params:
            optimizer_params["lr"] = optimizer_params.pop("learning_rate")
        optimizer: torch.optim.Optimizer = optimizer_class(
            self.parameters(), **optimizer_params
        )

        # If no lr_scheduler_config is provided, return just the optimizer
        if self.hp.lr_scheduler_config is None:
            return optimizer

        kind = scheduler_type(self.hp.lr_scheduler_config)
        scheduler_params = {
            k: v for k, v in self.hp.lr_scheduler_config.items() if k != "type"
        }

        scheduler: LRScheduler | ReduceLROnPlateau
        if kind == "CosineAnnealingWarmupRestarts":
            from torchcell.scheduler.cosine_annealing_warmup import (
                CosineAnnealingWarmupRestarts,
            )

            scheduler = CosineAnnealingWarmupRestarts(optimizer, **scheduler_params)
        elif kind == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
        else:
            # The task steps it on PLATEAU_MONITOR in on_validation_epoch_end; the
            # monitor is declared here too, for the record.
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": PLATEAU_MONITOR,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DiffusionRegressionTask(RegressionTask):
    """``RegressionTask`` for a diffusion decoder (006 ``hetero_cell_bipartite_dango_diff_gi``).

    ``loss_func`` must be a ``DiffusionLoss`` (or None for evaluation only): any other
    loss would train on the placeholders. Two differences, both from the model's
    contract: in training mode
    ``GeneInteractionDiff`` returns all-zero placeholders instead of predictions (its
    diffusion loss trains the decoder from the targets and ``z_p``), so train-stage
    predictions are not scored; and validation and test score the MSE of the sampled
    predictions, logged as ``{stage}/inference_mse``, instead of calling the
    diffusion loss, which ignores the predictions. ``train/avg_diffusion_loss`` and
    ``val/avg_inference_mse`` are the epoch means of those per-batch losses.
    """

    scores_train_predictions = False

    def __init__(
        self,
        model: nn.Module,
        cell_graph: torch.Tensor,
        optimizer_config: dict[str, Any],
        lr_scheduler_config: dict[str, Any] | None,
        batch_size: int | None = None,
        clip_grad_norm: bool = False,
        clip_grad_norm_max_norm: float = 0.1,
        plot_sample_ceiling: int = 1000,
        plot_every_n_epochs: int = 10,
        loss_func: nn.Module | None = None,
        grad_accumulation_schedule: dict[Any, int] | None = None,
        device: str = "cuda",
        inverse_transform: nn.Module | None = None,
        execution_mode: str = "training",
    ) -> None:
        """Set up the task as ``RegressionTask`` does, plus the two loss buffers."""
        super().__init__(
            model=model,
            cell_graph=cell_graph,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            batch_size=batch_size,
            clip_grad_norm=clip_grad_norm,
            clip_grad_norm_max_norm=clip_grad_norm_max_norm,
            plot_sample_ceiling=plot_sample_ceiling,
            plot_every_n_epochs=plot_every_n_epochs,
            loss_func=loss_func,
            grad_accumulation_schedule=grad_accumulation_schedule,
            device=device,
            inverse_transform=inverse_transform,
            execution_mode=execution_mode,
        )
        if loss_func is not None and not isinstance(loss_func, DiffusionLoss):
            raise ValueError(
                f"DiffusionRegressionTask trains with a DiffusionLoss, got "
                f"{type(loss_func).__name__}: in training mode the diffusion model "
                "returns all-zero placeholder predictions, so any other loss would "
                "train on them"
            )
        self.train_diffusion_loss: list[torch.Tensor] = []
        self.val_mse_during_inference: list[torch.Tensor] = []

    def _stage_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        representations: dict[str, Any],
        stage: str,
        batch_size: int,
    ) -> torch.Tensor:
        """Train: the ``RegressionTask`` loss (the diffusion loss), kept for the epoch
        mean. Validation and test: the MSE of the sampled predictions.
        """
        if stage == "train":
            loss = super()._stage_loss(
                predictions, targets, representations, stage, batch_size
            )
            self.train_diffusion_loss.append(loss.detach())
            return loss
        mse_loss = F.mse_loss(predictions, targets)
        self.log(
            f"{stage}/inference_mse",
            mse_loss.item(),
            batch_size=batch_size,
            sync_dist=True,
        )
        if stage == "val":
            self.val_mse_during_inference.append(mse_loss.detach())
        return mse_loss

    def on_train_epoch_end(self) -> None:
        """Log the epoch's mean diffusion loss, then end the epoch as the base task."""
        if self.train_diffusion_loss:
            avg_diffusion_loss = torch.stack(self.train_diffusion_loss).mean()
            self.log("train/avg_diffusion_loss", avg_diffusion_loss, sync_dist=True)
            self.train_diffusion_loss = []
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Log the epoch's mean inference MSE, then end the epoch as the base task."""
        if self.val_mse_during_inference:
            avg_inference_mse = torch.stack(self.val_mse_during_inference).mean()
            self.log("val/avg_inference_mse", avg_inference_mse, sync_dist=True)
            self.val_mse_during_inference = []
        super().on_validation_epoch_end()
