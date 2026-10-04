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
    BINARY_METRIC_NAMES,
    HIGHER_IS_BETTER,
    METRIC_NAMES,
    TASK_METRICS,
    BinaryMetricSet,
    MetricSet,
    average_ranks,
    binary_metric_set,
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
        "auroc": True,
        "auprc": True,
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
    per_target = {t: m.model_dump() for t, m in val.per_target.items()}
    macro = val.macro.model_dump()
    assert per_target["x"]["pearson"] == pytest.approx(1.0)
    assert per_target["y"]["pearson"] == pytest.approx(-1.0)
    assert macro["pearson"] == pytest.approx(0.0)
    assert per_target["y"]["mse"] == 1.0
    assert macro["mse"] == 0.5
    assert test.n_records == 2
    assert test.macro == MetricSet(pearson=1.0, spearman=1.0, mse=0.0, mae=0.0, r2=1.0)


def test_score_requires_exact_coverage() -> None:
    expected = {("v1", "x"): Split.VAL, ("v2", "x"): Split.VAL}
    labels = {("v1", "x"): 1.0, ("v2", "x"): 2.0}
    with pytest.raises(ValueError, match="exactly the expected pairs"):
        score({("v1", "x"): 1.0}, labels, expected)
    with pytest.raises(ValueError, match="exactly the expected pairs"):
        score({**labels, ("v9", "x"): 1.0}, labels, expected)


# ------------------------------------------------------------------- binary task

BINARY_TRUTH = np.array([1.0, 0.0, 1.0, 0.0])


def test_binary_metric_names() -> None:
    assert BINARY_METRIC_NAMES == ("auroc", "auprc")
    assert TASK_METRICS == {
        "regression": ("pearson", "spearman", "mse", "mae", "r2"),
        "binary": ("auroc", "auprc"),
    }
    assert HIGHER_IS_BETTER["auroc"] and HIGHER_IS_BETTER["auprc"]


def test_binary_hand_worked_metrics() -> None:
    # Ranked 1, 0, 1, 0. Three of the four positive-negative pairs are in order, so
    # AUROC is 3/4. Precision is 1 at the first positive and 2/3 at the second, each
    # gaining half the recall, so AUPRC is 1/2 + 1/3.
    metrics = binary_metric_set(np.array([0.9, 0.8, 0.7, 0.1]), BINARY_TRUTH)
    assert metrics.auroc == pytest.approx(0.75)
    assert metrics.auprc == pytest.approx(5 / 6)


def test_binary_perfect_and_inverted_rankings() -> None:
    perfect = binary_metric_set(np.array([0.9, 0.2, 0.8, 0.1]), BINARY_TRUTH)
    assert perfect == BinaryMetricSet(auroc=1.0, auprc=1.0)
    inverted = binary_metric_set(np.array([0.1, 0.8, 0.2, 0.9]), BINARY_TRUTH)
    assert inverted.auroc == 0.0
    # Positives arrive third and fourth: precision 1/3 then 2/4.
    assert inverted.auprc == pytest.approx(0.5 * (1 / 3) + 0.5 * (2 / 4))


def test_binary_ties_count_half_and_enter_together() -> None:
    # Three records share the top score: two positives and one negative. Each positive
    # ties one negative (half) and beats the other, so AUROC is 3/4. The tied block is
    # one threshold with precision 2/3 and all the recall.
    metrics = binary_metric_set(np.array([0.5, 0.5, 0.5, 0.1]), BINARY_TRUTH)
    assert metrics.auroc == pytest.approx(0.75)
    assert metrics.auprc == pytest.approx(2 / 3)


def test_binary_metrics_depend_only_on_the_order() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert binary_metric_set(scores, BINARY_TRUTH) == binary_metric_set(
        1000 * scores - 3, BINARY_TRUTH
    )


@pytest.mark.parametrize(
    ("prediction", "truth", "message"),
    [
        ([0.1, 0.2], [0.0, 2.0], "0 or 1"),
        ([0.1, 0.2], [1.0, 1.0], "both classes"),
        ([0.1, 0.2], [0.0, 0.0], "both classes"),
        ([0.5, 0.5], [0.0, 1.0], "constant score"),
        ([0.1, 0.2, 0.3], [0.0, 1.0], "equal length"),
    ],
)
def test_binary_metric_set_refuses_undefined_inputs(
    prediction: list[float], truth: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        binary_metric_set(np.array(prediction), np.array(truth))


def test_score_binary_task() -> None:
    expected = {
        ("v1", "e"): Split.VAL,
        ("v2", "e"): Split.VAL,
        ("v3", "e"): Split.VAL,
        ("v4", "e"): Split.VAL,
        ("s1", "e"): Split.TEST,
        ("s2", "e"): Split.TEST,
    }
    labels = dict(zip(expected, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
    predictions = dict(zip(expected, [0.9, 0.8, 0.7, 0.1, 2.0, -1.0]))
    scores = score(predictions, labels, expected, "binary")
    val, test = scores[Split.VAL], scores[Split.TEST]
    assert val.n_records == 4 and test.n_records == 2
    assert val.macro == val.per_target["e"]
    assert isinstance(val.macro, BinaryMetricSet)
    assert val.macro.auroc == pytest.approx(0.75)
    assert val.macro.auprc == pytest.approx(5 / 6)
    assert test.macro == BinaryMetricSet(auroc=1.0, auprc=1.0)
    assert set(val.model_dump(mode="json")["macro"]) == {"auroc", "auprc"}
    # The same scores read as a regression give the five regression metrics.
    assert isinstance(score(predictions, labels, expected)[Split.VAL].macro, MetricSet)
