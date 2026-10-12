# torchcell/trainers/compute_accounting.py
# [[torchcell.trainers.compute_accounting]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/trainers/compute_accounting
# Test file: tests/torchcell/trainers/test_compute_accounting.py
"""Log the three scaling-law quantities of a training run: N, D and C.

A scaling law relates held-out loss to parameters N, training data D and compute C, and
a curve is only as good as the bookkeeping behind each axis. This callback writes that
bookkeeping to W&B for every run, so the numbers exist when a campaign is read rather
than being reconstructed afterwards from wall clocks and guesses.

- **N**: ``compute/params_total`` and ``compute/params_non_embedding``. Kaplan et al. fit
  on non-embedding parameters because an embedding table scales with the vocabulary (here
  the genome) rather than with the capacity the loss responds to. A model exposes
  ``num_parameters`` as a dict; the components named in ``embedding_keys`` are subtracted.
- **D**: ``compute/records_per_epoch`` (the distinct training records the optimizer sees
  in one pass), ``compute/records_seen`` (cumulative, records times epochs) and
  ``compute/epochs_completed``. Which unit the loss averages over, records or entry
  rows, is the caller's statement in ``unit``; it is logged as a config key so a sweep
  cannot mix units unnoticed.
- **C**: measured, not computed from 6ND. ``torch.utils.flop_counter.FlopCounterMode``
  wraps SEVERAL real training steps (forward and backward, under manual or automatic
  optimization; the batch indices in ``measure_steps``, the second to the sixth by
  default) and reports the FLOPs each executed. ``compute/flops_per_step`` is the mean
  over the measured steps on rank 0, with ``compute/flops_per_step_min`` and ``_max``
  for the spread and ``compute/flops_measured_steps`` for how many were measured;
  ``compute/flops_per_record`` divides the summed FLOPs by the summed records of those
  steps; ``compute/flops_per_epoch`` multiplies the mean by the steps per epoch and the
  world size; ``compute/flops_cumulative`` accumulates across epochs and across
  checkpoint resumes (the callback state is saved with the checkpoint).
  ``compute/gpu_hours_cumulative`` is the wall-clock complement, elapsed training time
  times the world size. Every epoch-end row also carries ``epoch`` so the keys join the
  validation rows in a W&B history query (a row without it is invisible to a query that
  asks for ``epoch``).

Caveats, so the numbers are read correctly. FlopCounterMode counts the matmul family
(mm, addmm, bmm, baddbmm, convolution, scaled-dot-product attention); elementwise
kernels, softmax, layer norm and the scatter and gather of message passing are not
counted, so the figure is a lower bound on executed FLOPs and is comparable ACROSS
MODEL SIZES of one architecture, which is what a scaling curve needs. The measured
steps are the first few of the run, so on a per-entry model whose rows per batch vary
the min-to-max spread says how much one batch's composition moves the count. GPU hours
include loader stalls, FLOPs do not, and the ratio of the two is the utilization a run
achieved.

    from torchcell.trainers.compute_accounting import ComputeAccounting
    callbacks.append(ComputeAccounting(unit="entry_rows", embedding_keys=("gene_embedding",
                                       "embedding_preprocessor")))
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.utils.flop_counter import FlopCounterMode

__all__ = ["ComputeAccounting", "count_parameters"]


def count_parameters(
    model: torch.nn.Module, embedding_keys: Sequence[str]
) -> dict[str, int]:
    """``{"total", "non_embedding", "embedding"}`` trainable-parameter counts.

    Uses the model's ``num_parameters`` dict when it has one (the torchcell models do),
    subtracting the components named in ``embedding_keys``; otherwise counts every
    trainable parameter as non-embedding and reports embedding 0.
    """
    counts = getattr(model, "num_parameters", None)
    if isinstance(counts, dict) and "total" in counts:
        total = int(counts["total"])
        emb = int(sum(counts.get(k, 0) for k in embedding_keys))
    else:
        total = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        emb = 0
    return {"total": total, "non_embedding": total - emb, "embedding": emb}


class ComputeAccounting(Callback):
    """Log N, D and C for a run; see the module docstring for the keys and caveats."""

    def __init__(
        self,
        unit: str = "records",
        embedding_keys: Sequence[str] = ("gene_embedding", "embedding_preprocessor"),
        model_attr: str = "model",
        measure_steps: Sequence[int] = (1, 2, 3, 4, 5),
    ) -> None:
        """``unit`` names what the loss averages over; ``model_attr`` is the attribute of
        the LightningModule holding the network whose parameters are counted;
        ``measure_steps`` are the training batch indices whose FLOPs are counted (the
        first batch of a resumed run can carry one-time work, so the default starts at
        the second). The measurement closes at the end of the first epoch with whatever
        subset of those indices the epoch held.
        """
        super().__init__()
        self.unit = unit
        self.embedding_keys = tuple(embedding_keys)
        self.model_attr = model_attr
        self.measure_steps = tuple(int(s) for s in measure_steps)
        self.flops_per_step: float = 0.0
        self.flops_per_step_min: float = 0.0
        self.flops_per_step_max: float = 0.0
        self.flops_per_record: float = 0.0
        self.flops_measured_steps: int = 0
        self.flops_cumulative: float = 0.0
        self.records_seen: int = 0
        self.gpu_hours_cumulative: float = 0.0
        self.epochs_completed: int = 0
        self._counter: Any = None
        self._step_flops: list[float] = []
        self._step_records: list[int] = []
        self._measured = False
        self._epoch_start: float | None = None

    # --- state across checkpoint resumes
    def state_dict(self) -> dict[str, Any]:
        """Cumulative counters, saved with the checkpoint so a resume continues them."""
        return {
            "flops_per_step": self.flops_per_step,
            "flops_per_step_min": self.flops_per_step_min,
            "flops_per_step_max": self.flops_per_step_max,
            "flops_per_record": self.flops_per_record,
            "flops_measured_steps": self.flops_measured_steps,
            "flops_cumulative": self.flops_cumulative,
            "records_seen": self.records_seen,
            "gpu_hours_cumulative": self.gpu_hours_cumulative,
            "epochs_completed": self.epochs_completed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the counters; the step cost is measured again on the new job."""
        for k, v in state_dict.items():
            setattr(self, k, v)
        # A resumed run measures again: the step cost is a property of the model and
        # batch, and the new job may run on different hardware or batch size.
        self._measured = False
        self._step_flops = []
        self._step_records = []

    # --- helpers
    @staticmethod
    def _world_size(trainer: Trainer) -> int:
        return int(getattr(trainer, "world_size", 1) or 1)

    @staticmethod
    def _records_in_batch(batch: Any) -> int:
        n = getattr(batch, "num_graphs", None)
        if n is not None:
            return int(n)
        if (
            isinstance(batch, (list, tuple))
            and len(batch)
            and hasattr(batch[0], "shape")
        ):
            return int(batch[0].shape[0])
        if hasattr(batch, "shape"):
            return int(batch.shape[0])
        return 0

    def _records_per_epoch(self, trainer: Trainer, steps: int) -> int:
        """Distinct training records one epoch covers: the data module's train index
        (torchcell's ``CellDataModule``), else the train dataset's length, else the
        mean measured batch size times the steps.
        """
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "index") and hasattr(dm.index, "train"):
            return len(dm.index.train)
        if dm is not None and hasattr(dm, "train_dataset"):
            return len(dm.train_dataset)
        loader = getattr(trainer, "train_dataloader", None)
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "__len__"):
            return len(dataset)
        if not self._step_records:
            return 0
        return int(round(steps * sum(self._step_records) / len(self._step_records)))

    def _close_measurement(self) -> None:
        """Reduce the measured steps to the logged per-step and per-record figures."""
        if self._measured or not self._step_flops:
            return
        self.flops_measured_steps = len(self._step_flops)
        self.flops_per_step = float(sum(self._step_flops) / len(self._step_flops))
        self.flops_per_step_min = float(min(self._step_flops))
        self.flops_per_step_max = float(max(self._step_flops))
        records = sum(self._step_records)
        self.flops_per_record = (
            float(sum(self._step_flops) / records) if records else 0.0
        )
        self._measured = True

    def _log(self, pl_module: LightningModule, values: dict[str, float]) -> None:
        logger = pl_module.logger
        if logger is not None:
            logger.log_metrics(values, step=pl_module.global_step)

    # --- hooks
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log N: total, non-embedding and embedding parameters, and the world size."""
        net = getattr(pl_module, self.model_attr, pl_module)
        params = count_parameters(net, self.embedding_keys)
        if trainer.is_global_zero:
            self._log(
                pl_module,
                {
                    "compute/params_total": float(params["total"]),
                    "compute/params_non_embedding": float(params["non_embedding"]),
                    "compute/params_embedding": float(params["embedding"]),
                    "compute/world_size": float(self._world_size(trainer)),
                },
            )
            if logger := pl_module.logger:
                logger.log_hyperparams(
                    {
                        "compute_unit": self.unit,
                        "params_non_embedding": params["non_embedding"],
                    }
                )

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Start the epoch wall clock."""
        self._epoch_start = time.time()

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        """Enter the FLOP counter around a measured training step."""
        if self._measured or batch_idx not in self.measure_steps:
            return
        self._step_records.append(self._records_in_batch(batch))
        self._counter = FlopCounterMode(display=False)
        self._counter.__enter__()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Exit the FLOP counter and record the step's FLOPs; close the measurement
        once the last requested step has been counted.
        """
        if self._counter is None:
            return
        self._counter.__exit__(None, None, None)
        self._step_flops.append(float(self._counter.get_total_flops()))
        self._counter = None
        if batch_idx >= max(self.measure_steps):
            self._close_measurement()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log C and D for the epoch and the cumulative totals."""
        # An epoch shorter than the last requested step closes with what it measured.
        self._close_measurement()
        world = self._world_size(trainer)
        steps = int(trainer.num_training_batches)
        records_per_epoch = self._records_per_epoch(trainer, steps)
        flops_epoch = self.flops_per_step * steps * world
        self.flops_cumulative += flops_epoch
        self.records_seen += records_per_epoch
        self.epochs_completed += 1
        if self._epoch_start is not None:
            self.gpu_hours_cumulative += (
                (time.time() - self._epoch_start) / 3600 * world
            )
        if trainer.is_global_zero:
            self._log(
                pl_module,
                {
                    "epoch": float(trainer.current_epoch),
                    "compute/flops_per_step": self.flops_per_step,
                    "compute/flops_per_step_min": self.flops_per_step_min,
                    "compute/flops_per_step_max": self.flops_per_step_max,
                    "compute/flops_measured_steps": float(self.flops_measured_steps),
                    "compute/flops_per_record": self.flops_per_record,
                    "compute/flops_per_epoch": flops_epoch,
                    "compute/flops_cumulative": self.flops_cumulative,
                    "compute/records_per_epoch": float(records_per_epoch),
                    "compute/records_seen": float(self.records_seen),
                    "compute/epochs_completed": float(self.epochs_completed),
                    "compute/gpu_hours_cumulative": self.gpu_hours_cumulative,
                    "compute/steps_per_epoch": float(steps),
                },
            )
