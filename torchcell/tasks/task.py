# torchcell/tasks/task.py
# [[torchcell.tasks.task]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/tasks/task.py
"""The benchmark task object, its splits, the locked test part and the metrics."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel
from scipy.stats import rankdata

from torchcell.tasks.splits import TEST, TRAIN, VAL

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


class TestSplitLocked(RuntimeError):
    """Raised when the test part of a split is read before it is unlocked."""

    __test__ = False  # not a pytest class


class TestSplitWarning(UserWarning):
    """Emitted every time the test part of a split is unlocked."""

    __test__ = False  # not a pytest class


class TaskSpec(BaseModel):
    """What one benchmark task is: the dataset, the genome, the label and the metric.

    Attributes:
        name: The task name passed to ``load_task``.
        title: A one-line title for people.
        dataset_slug: Directory name of the dataset's store.
        dataset_module: Module that defines the dataset class.
        dataset_class: Name of the dataset class.
        assembly_set: Id of the genome assembly the records are written against.
        kind: ``single_gene`` (a sample deletes one gene) or ``gene_pair`` (two).
        label: Field of a record's phenotype that holds the label.
        n_targets_note: How many targets a sample has and what one target is.
        headline_metric: The metric results are reported by.
    """

    name: str
    title: str
    dataset_slug: str
    dataset_module: str
    dataset_class: str
    assembly_set: str
    kind: Literal["single_gene", "gene_pair"]
    label: str
    n_targets_note: str
    headline_metric: Literal["pearson"] = "pearson"


class Split:
    """One named split of a task: row indices of train and validation, test locked.

    ``train`` and ``val`` are arrays of row indices into the task's samples. ``test``
    raises :class:`TestSplitLocked`. ``unlock_test`` returns the test indices and warns:
    the test part is for one final evaluation of a model that was chosen on validation,
    and a script that calls it says so in ``reason``.
    """

    def __init__(self, task: str, name: str, parts: npt.NDArray[np.str_]) -> None:
        """Hold the part of every sample under one split.

        Args:
            task: Name of the task (a string, not the task object).
            name: Name of the split.
            parts: ``"train"``, ``"val"`` or ``"test"`` for every sample of the task.
        """
        self.task = task
        self.name = name
        self._parts = parts

    @property
    def train(self) -> IntArray:
        """Row indices of the training samples."""
        return np.flatnonzero(self._parts == TRAIN)

    @property
    def val(self) -> IntArray:
        """Row indices of the validation samples."""
        return np.flatnonzero(self._parts == VAL)

    @property
    def test(self) -> IntArray:
        """Locked. Call ``unlock_test(reason=...)`` for the test indices."""
        raise TestSplitLocked(
            f"the test part of {self.task}/{self.name} is locked. Select models and "
            "hyperparameters on .val. For the one final evaluation call "
            "split.unlock_test(reason='...')."
        )

    def unlock_test(self, reason: str) -> IntArray:
        """Row indices of the test samples, with a warning.

        Args:
            reason: Why this script reads the test part (for example ``"final
                evaluation of the model selected on validation"``). Required and not
                empty; it is repeated in the warning.

        Returns:
            Row indices of the test samples.

        Raises:
            ValueError: ``reason`` is empty.
        """
        if not reason.strip():
            raise ValueError("unlock_test needs a non-empty reason")
        warnings.warn(
            f"TEST SPLIT UNLOCKED for {self.task}/{self.name} ({reason}). Do not fit, "
            "tune, select or early-stop on these samples; report one evaluation.",
            TestSplitWarning,
            stacklevel=2,
        )
        return np.flatnonzero(self._parts == TEST)

    def sizes(self) -> dict[str, int]:
        """Number of samples in train, validation and test."""
        return {part: int((self._parts == part).sum()) for part in (TRAIN, VAL, TEST)}


class BenchmarkTask:
    """A processed benchmark task: samples, genes, labels, genome and pinned splits.

    Attributes:
        spec: The task definition.
        samples: One row per benchmark sample, in label order. Columns include
            ``record_index`` (the row of the source dataset) and ``key`` (the locus
            tag, or two joined by ``|``).
        genes: Locus tags of the perturbed genes, shape ``(n_samples, 1)`` for a
            single-gene task and ``(n_samples, 2)`` for a gene-pair task.
        y: Labels, shape ``(n_samples, n_targets)``; ``nan`` where a target was not
            measured for a sample.
        targets: Name of each label column.
        excluded: Records of the source dataset that are not benchmark samples, each
            with its ``exclusion_reason``. Nothing is dropped without a row here.
        genome: The genome the records are written against.
    """

    def __init__(
        self,
        spec: TaskSpec,
        samples: pd.DataFrame,
        genes: npt.NDArray[np.str_],
        y: FloatArray,
        targets: list[str],
        excluded: pd.DataFrame,
        genome: Any,
        splits: pd.DataFrame | None = None,
    ) -> None:
        """Assemble a task; ``splits`` has one column of parts per split name."""
        self.spec = spec
        self.samples = samples
        self.genes = genes
        self.y = y
        self.targets = targets
        self.excluded = excluded
        self.genome = genome
        self._splits = splits

    def __len__(self) -> int:
        """Number of benchmark samples."""
        return len(self.samples)

    @property
    def split_names(self) -> list[str]:
        """Names of the pinned splits."""
        if self._splits is None:
            return []
        return [str(c) for c in self._splits.columns]

    def split(self, name: str) -> Split:
        """The pinned split ``name``."""
        if self._splits is None or name not in self._splits.columns:
            raise KeyError(
                f"{self.spec.name} has no split {name!r}: {self.split_names}"
            )
        return Split(self.spec.name, name, self._splits[name].to_numpy(dtype=str))

    def score(self, y_true: FloatArray, y_pred: FloatArray) -> dict[str, float]:
        """Mean over targets of each metric; see :func:`regression_metrics`.

        A target whose metric is ``nan`` (for example the correlation of a constant
        prediction) is left out of that metric's mean.

        Args:
            y_true: True labels, ``(n_samples, n_targets)``.
            y_pred: Predictions of the same shape.

        Returns:
            ``pearson``, ``spearman``, ``r2``, ``mse`` and ``mae``, each one number.
        """
        return {
            k: float(np.nanmean(v))
            for k, v in regression_metrics(y_true, y_pred).items()
        }


def regression_metrics(y_true: FloatArray, y_pred: FloatArray) -> dict[str, FloatArray]:
    """Pearson, Spearman, R2, MSE and MAE of each target column.

    A sample whose true value is ``nan`` is left out of that target. A target with no
    measured sample gets ``nan`` for every metric. A target whose true values do not
    vary gets ``nan`` for R2 and the correlations, and a target with fewer than three
    measured samples or a constant prediction gets ``nan`` for the correlations. A prediction is treated as constant when its
    standard deviation is below 1e-9 of its largest magnitude (or of 1), so the
    rounding noise of a constant model is not scored as a correlation.

    Args:
        y_true: True labels, ``(n_samples, n_targets)`` or ``(n_samples,)``.
        y_pred: Predictions of the same shape.

    Returns:
        Metric name to an array with one value per target.
    """
    truth = np.asarray(y_true, dtype=np.float64).reshape(len(y_true), -1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(len(y_pred), -1)
    if truth.shape != pred.shape:
        raise ValueError(f"shape mismatch: {truth.shape} and {pred.shape}")
    n_targets = truth.shape[1]
    out: dict[str, FloatArray] = {
        name: np.full(n_targets, np.nan, dtype=np.float64)
        for name in ("pearson", "spearman", "r2", "mse", "mae")
    }
    for j in range(n_targets):
        keep = ~np.isnan(truth[:, j])
        t, p = truth[keep, j], pred[keep, j]
        if len(t) == 0:
            continue
        error = t - p
        out["mse"][j] = float(np.mean(error**2))
        out["mae"][j] = float(np.mean(np.abs(error)))
        variance = float(np.sum((t - t.mean()) ** 2))
        if variance > 0:
            out["r2"][j] = 1.0 - float(np.sum(error**2)) / variance
        spread = float(np.std(p))
        if (
            len(t) >= 3
            and variance > 0
            and spread > 1e-9 * max(1.0, float(np.abs(p).max()))
        ):
            out["pearson"][j] = float(np.corrcoef(t, p)[0, 1])
            out["spearman"][j] = float(np.corrcoef(rankdata(t), rankdata(p))[0, 1])
    return out
