# torchcell/benchmark/grading.py
# [[torchcell.benchmark.grading]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/grading.py
# Test file: tests/torchcell/benchmark/test_grading.py

"""The grader: metrics computed from submitted predictions against the stored labels.

Five metrics per target per split: Pearson r, Spearman rho (Pearson on average ranks,
so ties share a rank), mean squared error, mean absolute error, and R2
(``1 - SS_res / SS_tot``). A multi-target dataset reports the macro average (the
unweighted mean of the per-target values) beside the per-target values. numpy only.

A dataset whose task is ``binary`` has 0/1 labels and is scored with two metrics
instead: the area under the ROC curve (AUROC) and the area under the precision-recall
curve as average precision (AUPRC). A prediction is then a score, larger meaning more
likely 1, and only the order of the scores matters.

The grader assumes a validated submission: :func:`score` raises if the prediction keys
are not exactly the expected keys, and :func:`metric_set` raises on a constant vector,
because a correlation with a constant is undefined. Validation rejects a constant
prediction vector before grading, and the bundle builder rejects constant labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from torchcell.benchmark.submission import SCORED_SPLITS, Split

Task = Literal["regression", "binary"]
RegressionMetricName = Literal["pearson", "spearman", "mse", "mae", "r2"]
BinaryMetricName = Literal["auroc", "auprc"]
MetricName = RegressionMetricName | BinaryMetricName
METRIC_NAMES: tuple[MetricName, ...] = get_args(RegressionMetricName)
BINARY_METRIC_NAMES: tuple[MetricName, ...] = get_args(BinaryMetricName)
TASK_METRICS: dict[Task, tuple[MetricName, ...]] = {
    "regression": METRIC_NAMES,
    "binary": BINARY_METRIC_NAMES,
}
HIGHER_IS_BETTER: dict[MetricName, bool] = {
    "pearson": True,
    "spearman": True,
    "mse": False,
    "mae": False,
    "r2": True,
    "auroc": True,
    "auprc": True,
}

PairKey = tuple[str, str]  # (record_id, target)
FloatArray = npt.NDArray[np.float64]


class MetricSet(BaseModel):
    """The five regression metrics for one target on one split (or their macro average)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pearson: float
    spearman: float
    mse: float
    mae: float
    r2: float


class BinaryMetricSet(BaseModel):
    """The two metrics of a binary target on one split (or their macro average).

    Both read the predictions as scores, larger meaning more likely positive, and
    depend only on their order, so any monotone rescaling of a submission scores the
    same.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auroc: float
    auprc: float


class SplitScores(BaseModel):
    """Scores of one split: the macro average and every target's own metrics."""

    model_config = ConfigDict(frozen=True)

    n_records: int
    macro: MetricSet | BinaryMetricSet
    per_target: dict[str, MetricSet] | dict[str, BinaryMetricSet]


def average_ranks(values: FloatArray) -> FloatArray:
    """Ranks starting at 1, with tied values sharing the mean of their positions."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start
        while (
            stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]
        ):
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
        start = stop + 1
    return ranks


def _pearson(x: FloatArray, y: FloatArray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    return float((xc @ yc) / np.sqrt((xc @ xc) * (yc @ yc)))


def metric_set(prediction: FloatArray, truth: FloatArray) -> MetricSet:
    """All five metrics for one prediction vector against its label vector."""
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("prediction and truth must be 1-D arrays of equal length")
    if len(truth) < 2:
        raise ValueError("at least two records are needed to score a target")
    if np.ptp(prediction) == 0 or np.ptp(truth) == 0:
        raise ValueError("a constant vector has no defined correlation")
    residual = prediction - truth
    ss_res = float(residual @ residual)
    centered = truth - truth.mean()
    return MetricSet(
        pearson=_pearson(prediction, truth),
        spearman=_pearson(average_ranks(prediction), average_ranks(truth)),
        mse=ss_res / len(truth),
        mae=float(np.abs(residual).mean()),
        r2=1.0 - ss_res / float(centered @ centered),
    )


def binary_metric_set(prediction: FloatArray, truth: FloatArray) -> BinaryMetricSet:
    """AUROC and AUPRC of one score vector against its 0/1 label vector.

    AUROC is the Mann-Whitney statistic on average ranks, so tied scores count half.
    AUPRC is average precision: the sum over distinct score thresholds, from the
    highest down, of the recall gained at the threshold times the precision there;
    records with a tied score enter together.
    """
    if prediction.shape != truth.shape or prediction.ndim != 1:
        raise ValueError("prediction and truth must be 1-D arrays of equal length")
    if not np.isin(truth, (0.0, 1.0)).all():
        raise ValueError("binary labels must be 0 or 1")
    n_positive = int(truth.sum())
    n_negative = len(truth) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("a binary target needs both classes to be scored")
    if np.ptp(prediction) == 0:
        raise ValueError("a constant score vector ranks nothing")
    ranks = average_ranks(prediction)
    auroc = (float(ranks[truth == 1.0].sum()) - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )
    order = np.argsort(-prediction, kind="mergesort")
    sorted_scores = prediction[order]
    true_positives = np.cumsum(truth[order])
    # The last index of each run of tied scores is a threshold.
    thresholds = np.flatnonzero(np.diff(sorted_scores, append=-np.inf) != 0)
    recall = true_positives[thresholds] / n_positive
    precision = true_positives[thresholds] / (thresholds + 1)
    auprc = float(np.sum(np.diff(recall, prepend=0.0) * precision))
    return BinaryMetricSet(auroc=auroc, auprc=auprc)


def macro_average[M: (MetricSet, BinaryMetricSet)](per_target: Mapping[str, M]) -> M:
    """The unweighted mean of each metric over the targets."""
    kind = type(next(iter(per_target.values())))
    return kind(
        **{
            name: float(np.mean([getattr(m, name) for m in per_target.values()]))
            for name in kind.model_fields
        }
    )


def score(
    predictions: Mapping[PairKey, float],
    labels: Mapping[PairKey, float],
    expected: Mapping[PairKey, Split],
    task: Task = "regression",
) -> dict[Split, SplitScores]:
    """Score a validated submission on each scored split.

    ``expected`` maps every (record_id, target) pair the dataset scores to its split;
    ``predictions`` and ``labels`` must hold exactly those pairs. ``task`` selects the
    metrics: the five regression metrics, or AUROC and AUPRC for 0/1 labels.
    """
    if predictions.keys() != expected.keys() or labels.keys() != expected.keys():
        raise ValueError("predictions and labels must cover exactly the expected pairs")
    result: dict[Split, SplitScores] = {}
    for split in SCORED_SPLITS:
        by_target: dict[str, list[PairKey]] = {}
        records: set[str] = set()
        for key, key_split in expected.items():
            if key_split == split:
                by_target.setdefault(key[1], []).append(key)
                records.add(key[0])
        vectors = {
            target: (
                np.array([predictions[k] for k in keys], dtype=np.float64),
                np.array([labels[k] for k in keys], dtype=np.float64),
            )
            for target, keys in sorted(by_target.items())
        }
        if task == "binary":
            binary = {t: binary_metric_set(p, y) for t, (p, y) in vectors.items()}
            result[split] = SplitScores(
                n_records=len(records), macro=macro_average(binary), per_target=binary
            )
        else:
            regression = {t: metric_set(p, y) for t, (p, y) in vectors.items()}
            result[split] = SplitScores(
                n_records=len(records),
                macro=macro_average(regression),
                per_target=regression,
            )
    return result
