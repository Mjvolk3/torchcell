# tests/torchcell/benchmark/test_grading.py
# [[tests.torchcell.benchmark.test_grading]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_grading.py
"""``torchcell.benchmark.grading`` against values worked by hand.

Truth 1, 2, 3, 4 and prediction 1, 2, 3, 5: the residuals are 0, 0, 0, 1, so MSE and
MAE are 0.25; SS_tot is 5, so R2 is 1 - 1/5 = 0.8; the centered products give Pearson
6.5 / sqrt(8.75 * 5) = 0.98270763; the order is unchanged, so Spearman is 1. Prediction
1, 1, 2, 3 ties the first two records at rank 1.5, giving Spearman
4.5 / sqrt(4.5 * 5) = 0.94868330.
"""

import math

import numpy as np
import pytest

from torchcell.benchmark.grading import (
    HIGHER_IS_BETTER,
    METRIC_NAMES,
    MetricSet,
    average_ranks,
    macro_average,
    metric_set,
    score,
)
from torchcell.benchmark.submission import Split

TRUTH = np.array([1.0, 2.0, 3.0, 4.0])


def test_metric_names_and_directions() -> None:
    assert METRIC_NAMES == ("pearson", "spearman", "mse", "mae", "r2")
    assert HIGHER_IS_BETTER == {
        "pearson": True,
        "spearman": True,
        "mse": False,
        "mae": False,
        "r2": True,
    }


def test_average_ranks_share_tied_positions() -> None:
    assert average_ranks(np.array([10.0, 20.0, 20.0, 30.0])).tolist() == [
        1,
        2.5,
        2.5,
        4,
    ]
    assert average_ranks(np.array([3.0, 1.0, 2.0])).tolist() == [3, 1, 2]
    assert average_ranks(np.array([5.0, 5.0, 5.0])).tolist() == [2, 2, 2]


def test_perfect_prediction() -> None:
    assert metric_set(TRUTH.copy(), TRUTH) == MetricSet(
        pearson=1.0, spearman=1.0, mse=0.0, mae=0.0, r2=1.0
    )


def test_hand_worked_metrics() -> None:
    metrics = metric_set(np.array([1.0, 2.0, 3.0, 5.0]), TRUTH)
    assert metrics.pearson == pytest.approx(6.5 / math.sqrt(8.75 * 5), abs=1e-12)
    assert metrics.spearman == pytest.approx(1.0, abs=1e-12)
    assert metrics.mse == 0.25
    assert metrics.mae == 0.25
    assert metrics.r2 == pytest.approx(0.8, abs=1e-12)


def test_spearman_with_ties_and_negative_r2() -> None:
    tied = metric_set(np.array([1.0, 1.0, 2.0, 3.0]), TRUTH)
    assert tied.spearman == pytest.approx(4.5 / math.sqrt(4.5 * 5), abs=1e-12)
    reversed_prediction = metric_set(np.array([4.0, 3.0, 2.0, 1.0]), TRUTH)
    assert reversed_prediction.pearson == pytest.approx(-1.0, abs=1e-12)
    assert reversed_prediction.spearman == pytest.approx(-1.0, abs=1e-12)
    # residuals -3, -1, 1, 3: SS_res 20 against SS_tot 5
    assert reversed_prediction.r2 == pytest.approx(-3.0, abs=1e-12)
    assert reversed_prediction.mse == 5.0


@pytest.mark.parametrize(
    ("prediction", "truth", "message"),
    [
        ([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], "constant vector"),
        ([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0], "constant vector"),
        ([1.0], [1.0], "at least two records"),
        ([1.0, 2.0], [1.0, 2.0, 3.0], "equal length"),
    ],
)
def test_metric_set_refuses_undefined_inputs(
    prediction: list[float], truth: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric_set(np.array(prediction), np.array(truth))


def test_macro_average_is_the_unweighted_mean() -> None:
    a = MetricSet(pearson=1.0, spearman=0.5, mse=0.0, mae=0.2, r2=1.0)
    b = MetricSet(pearson=0.0, spearman=1.0, mse=2.0, mae=0.4, r2=-1.0)
    assert macro_average({"a": a, "b": b}).model_dump() == {
        "pearson": 0.5,
        "spearman": 0.75,
        "mse": 1.0,
        "mae": pytest.approx(0.3),
        "r2": 0.0,
    }


def test_score_splits_targets_and_counts_records() -> None:
    expected = {
        ("v1", "x"): Split.VAL,
        ("v2", "x"): Split.VAL,
        ("v3", "x"): Split.VAL,
        ("v1", "y"): Split.VAL,
        ("v2", "y"): Split.VAL,
        ("s1", "x"): Split.TEST,
        ("s2", "x"): Split.TEST,
        ("s1", "y"): Split.TEST,
        ("s2", "y"): Split.TEST,
    }
    labels = {
        ("v1", "x"): 1.0,
        ("v2", "x"): 2.0,
        ("v3", "x"): 3.0,
        ("v1", "y"): 0.0,
        ("v2", "y"): 1.0,
        ("s1", "x"): 1.0,
        ("s2", "x"): 2.0,
        ("s1", "y"): 5.0,
        ("s2", "y"): 7.0,
    }
    predictions = {**labels, ("v1", "y"): 1.0, ("v2", "y"): 0.0}
    scores = score(predictions, labels, expected)
    val, test = scores[Split.VAL], scores[Split.TEST]
    # v3 has no label for y, so x is scored on three records and y on two.
    assert val.n_records == 3
    assert list(val.per_target) == ["x", "y"]
    assert val.per_target["x"].pearson == pytest.approx(1.0)
    assert val.per_target["y"].pearson == pytest.approx(-1.0)
    assert val.macro.pearson == pytest.approx(0.0)
    assert val.per_target["y"].mse == 1.0
    assert val.macro.mse == 0.5
    assert test.n_records == 2
    assert test.macro == MetricSet(pearson=1.0, spearman=1.0, mse=0.0, mae=0.0, r2=1.0)


def test_score_requires_exact_coverage() -> None:
    expected = {("v1", "x"): Split.VAL, ("v2", "x"): Split.VAL}
    labels = {("v1", "x"): 1.0, ("v2", "x"): 2.0}
    with pytest.raises(ValueError, match="exactly the expected pairs"):
        score({("v1", "x"): 1.0}, labels, expected)
    with pytest.raises(ValueError, match="exactly the expected pairs"):
        score({**labels, ("v9", "x"): 1.0}, labels, expected)
