# tests/torchcell/tasks/test_task.py
# [[tests.torchcell.tasks.test_task]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/tasks/test_task.py
"""Pure tests for ``torchcell.tasks.task``: the split lock, the task object, the metrics.

Hand-computed metrics. Column 0: truth ``[1, 2, 3, 4]``, prediction ``[1, 3, 2, 4]``;
errors ``[0, -1, 1, 0]`` give MSE 0.5 and MAE 0.5; the truth's sum of squares about its
mean is 5, so R2 = 1 - 2/5 = 0.6; both deviation vectors have sum of squares 5 and
cross product 4, so Pearson = 0.8, and the ranks equal the values, so Spearman = 0.8.
Column 1: truth ``[nan, 1, 2, 3]``, prediction ``[100, 1, 2, 4]``; the nan row is left
out, errors ``[0, 0, -1]`` give MSE = MAE = 1/3; R2 = 1 - 1/2 = 0.5; deviations
``[-1, 0, 1]`` and ``[-4/3, -1/3, 5/3]`` give Pearson = 3 / sqrt(2 * 42/9) =
9 / sqrt(84); the ranks agree, so Spearman = 1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from torchcell.tasks.task import (
    BenchmarkTask,
    Split,
    TaskSpec,
    TestSplitLocked,
    TestSplitWarning,
    regression_metrics,
)

PARTS = np.array(["train", "val", "test", "train", "test", "val", "train"], dtype=str)


def make_spec() -> TaskSpec:
    return TaskSpec(
        name="toy-task",
        title="Toy task",
        dataset_slug="toy",
        dataset_module="toy.module",
        dataset_class="ToyDataset",
        assembly_set="toy_set",
        kind="single_gene",
        label="fitness",
        n_targets_note="1 target",
    )


def make_task(splits: pd.DataFrame | None) -> BenchmarkTask:
    samples = pd.DataFrame({"record_index": [0, 1, 2], "key": ["b1", "b2", "b3"]})
    return BenchmarkTask(
        spec=make_spec(),
        samples=samples,
        genes=np.array([["b1"], ["b2"], ["b3"]], dtype=str),
        y=np.array([[0.1], [0.2], [0.3]]),
        targets=["fitness"],
        excluded=pd.DataFrame(),
        genome=None,
        splits=splits,
    )


def test_task_spec_default_metric() -> None:
    spec = make_spec()
    assert spec.headline_metric == "pearson"
    assert spec.kind == "single_gene"


def test_split_train_and_val_exact() -> None:
    split = Split("toy-task", "random_s0", PARTS)
    np.testing.assert_array_equal(split.train, np.array([0, 3, 6]))
    np.testing.assert_array_equal(split.val, np.array([1, 5]))
    assert split.task == "toy-task"
    assert split.name == "random_s0"


def test_split_sizes_exact() -> None:
    assert Split("t", "s", PARTS).sizes() == {"train": 3, "val": 2, "test": 2}


def test_split_test_is_locked() -> None:
    split = Split("toy-task", "random_s0", PARTS)
    with pytest.raises(TestSplitLocked) as info:
        _ = split.test
    assert str(info.value) == (
        "the test part of toy-task/random_s0 is locked. Select models and "
        "hyperparameters on .val. For the one final evaluation call "
        "split.unlock_test(reason='...')."
    )
    assert issubclass(TestSplitLocked, RuntimeError)


def test_unlock_test_warns_and_returns_exact_indices() -> None:
    split = Split("toy-task", "random_s0", PARTS)
    with pytest.warns(TestSplitWarning) as record:
        test = split.unlock_test(reason="final evaluation")
    np.testing.assert_array_equal(test, np.array([2, 4]))
    assert len(record) == 1
    assert str(record[0].message) == (
        "TEST SPLIT UNLOCKED for toy-task/random_s0 (final evaluation). Do not fit, "
        "tune, select or early-stop on these samples; report one evaluation."
    )
    assert issubclass(TestSplitWarning, UserWarning)


@pytest.mark.parametrize("reason", ["", "   ", "\t\n"])
def test_unlock_test_rejects_empty_reason(reason: str) -> None:
    split = Split("toy-task", "random_s0", PARTS)
    with pytest.raises(ValueError, match=r"^unlock_test needs a non-empty reason$"):
        split.unlock_test(reason=reason)


def test_task_split_names_and_split() -> None:
    splits = pd.DataFrame(
        {"random_s0": ["train", "val", "test"], "position_s0": ["test", "train", "val"]}
    )
    task = make_task(splits)
    assert len(task) == 3
    assert task.split_names == ["random_s0", "position_s0"]
    split = task.split("position_s0")
    assert split.task == "toy-task"
    assert split.name == "position_s0"
    np.testing.assert_array_equal(split.train, np.array([1]))
    np.testing.assert_array_equal(split.val, np.array([2]))


def test_task_unknown_split_raises_key_error() -> None:
    task = make_task(pd.DataFrame({"random_s0": ["train", "val", "test"]}))
    with pytest.raises(KeyError) as info:
        task.split("nope")
    assert info.value.args[0] == "toy-task has no split 'nope': ['random_s0']"


def test_task_without_splits() -> None:
    task = make_task(None)
    assert task.split_names == []
    with pytest.raises(KeyError) as info:
        task.split("random_s0")
    assert info.value.args[0] == "toy-task has no split 'random_s0': []"


def test_regression_metrics_hand_computed() -> None:
    y_true = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    y_pred = np.array([[1.0, 100.0], [3.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    out = regression_metrics(y_true, y_pred)
    assert sorted(out) == ["mae", "mse", "pearson", "r2", "spearman"]
    np.testing.assert_allclose(out["mse"], [0.5, 1 / 3])
    np.testing.assert_allclose(out["mae"], [0.5, 1 / 3])
    np.testing.assert_allclose(out["r2"], [0.6, 0.5])
    np.testing.assert_allclose(out["pearson"], [0.8, 9 / math.sqrt(84)])
    np.testing.assert_allclose(out["spearman"], [0.8, 1.0])


def test_regression_metrics_one_dimensional_input() -> None:
    out = regression_metrics(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 3, 2, 4]))
    np.testing.assert_allclose(out["pearson"], [0.8])
    np.testing.assert_allclose(out["mse"], [0.5])


def test_regression_metrics_constant_prediction_gives_nan_correlation() -> None:
    out = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
    assert np.isnan(out["pearson"][0])
    assert np.isnan(out["spearman"][0])
    # Errors [-1, 0, 1]: MSE 2/3, MAE 2/3, R2 = 1 - 2/2 = 0.
    np.testing.assert_allclose(out["mse"], [2 / 3])
    np.testing.assert_allclose(out["mae"], [2 / 3])
    np.testing.assert_allclose(out["r2"], [0.0])


def test_regression_metrics_constant_truth_gives_nan_r2_and_correlation() -> None:
    out = regression_metrics(np.array([5.0, 5.0, 5.0]), np.array([4.0, 5.0, 7.0]))
    assert np.isnan(out["r2"][0])
    assert np.isnan(out["pearson"][0])
    # Errors [1, 0, -2]: MSE 5/3, MAE 1.
    np.testing.assert_allclose(out["mse"], [5 / 3])
    np.testing.assert_allclose(out["mae"], [1.0])


def test_regression_metrics_two_measured_samples_give_nan_correlation() -> None:
    out = regression_metrics(np.array([np.nan, 1.0, 3.0]), np.array([0.0, 2.0, 3.0]))
    assert np.isnan(out["pearson"][0])
    assert np.isnan(out["spearman"][0])
    # Measured errors [-1, 0]: MSE 0.5, MAE 0.5; truth sum of squares 2, R2 0.5.
    np.testing.assert_allclose(out["mse"], [0.5])
    np.testing.assert_allclose(out["r2"], [0.5])


def test_regression_metrics_all_nan_target_is_all_nan() -> None:
    out = regression_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert {name: bool(np.isnan(v[0])) for name, v in out.items()} == {
        "pearson": True,
        "spearman": True,
        "r2": True,
        "mse": True,
        "mae": True,
    }


def test_regression_metrics_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"^shape mismatch: \(3, 1\) and \(2, 1\)$"):
        regression_metrics(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match=r"^shape mismatch: \(2, 2\) and \(2, 1\)$"):
        regression_metrics(np.zeros((2, 2)), np.zeros((2, 1)))


def test_score_is_nanmean_over_targets() -> None:
    task = make_task(None)
    y_true = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 2.0], [2.0, 2.0], [4.0, 2.0]])
    out = task.score(y_true, y_pred)
    # Column 0 is the hand-computed case; column 1 is a constant prediction, whose
    # correlations are nan and drop out of the mean. Column 1 errors [-1, 0, 1, 2]:
    # MSE 6/4, MAE 1, R2 = 1 - 6/5.
    assert out["pearson"] == pytest.approx(0.8)
    assert out["spearman"] == pytest.approx(0.8)
    assert out["mse"] == pytest.approx((0.5 + 1.5) / 2)
    assert out["mae"] == pytest.approx((0.5 + 1.0) / 2)
    assert out["r2"] == pytest.approx((0.6 + (1 - 6 / 5)) / 2)
