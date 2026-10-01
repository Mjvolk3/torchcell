# torchcell/benchmark/grading.py
# [[torchcell.benchmark.grading]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/grading.py
# Test file: tests/torchcell/benchmark/test_grading.py

"""The grader: metrics computed from submitted predictions against the stored labels.

Five metrics per target per split: Pearson r, Spearman rho (Pearson on average ranks,
so ties share a rank), mean squared error, mean absolute error, and R2
(``1 - SS_res / SS_tot``). A multi-target dataset reports the macro average (the
unweighted mean of the per-target values) beside the per-target values. numpy only.

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

MetricName = Literal["pearson", "spearman", "mse", "mae", "r2"]
METRIC_NAMES: tuple[MetricName, ...] = get_args(MetricName)
HIGHER_IS_BETTER: dict[MetricName, bool] = {
    "pearson": True,
    "spearman": True,
    "mse": False,
    "mae": False,
    "r2": True,
}

PairKey = tuple[str, str]  # (record_id, target)
FloatArray = npt.NDArray[np.float64]


class MetricSet(BaseModel):
    """The five metrics for one target on one split (or their macro average)."""

    model_config = ConfigDict(frozen=True)

    pearson: float
    spearman: float
    mse: float
    mae: float
    r2: float


class SplitScores(BaseModel):
    """Scores of one split: the macro average and every target's own metrics."""

    model_config = ConfigDict(frozen=True)

    n_records: int
    macro: MetricSet
    per_target: dict[str, MetricSet]


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


def macro_average(per_target: Mapping[str, MetricSet]) -> MetricSet:
    """The unweighted mean of each metric over the targets."""
    return MetricSet(
        **{
            name: float(np.mean([getattr(m, name) for m in per_target.values()]))
            for name in METRIC_NAMES
        }
    )


def score(
    predictions: Mapping[PairKey, float],
    labels: Mapping[PairKey, float],
    expected: Mapping[PairKey, Split],
) -> dict[Split, SplitScores]:
    """Score a validated submission on each scored split.

    ``expected`` maps every (record_id, target) pair the dataset scores to its split;
    ``predictions`` and ``labels`` must hold exactly those pairs.
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
        per_target = {
            target: metric_set(
                np.array([predictions[k] for k in keys], dtype=np.float64),
                np.array([labels[k] for k in keys], dtype=np.float64),
            )
            for target, keys in sorted(by_target.items())
        }
        result[split] = SplitScores(
            n_records=len(records),
            macro=macro_average(per_target),
            per_target=per_target,
        )
    return result
