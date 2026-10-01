# tests/torchcell/benchmark/test_integrity.py
# [[tests.torchcell.benchmark.test_integrity]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_integrity.py
"""``torchcell.benchmark.integrity``: the two flags at their boundaries.

``test_exceeds_val`` fires only when test beats validation by more than the tolerance
(0.02 by default), in the metric's own direction: for an error metric, a lower test
value is the better one. The cases sit 0.005 inside and 0.01 outside the tolerance, not
on it, because 0.52 - 0.50 is not exactly 0.02 in binary floating point. ``val_test_divergence`` needs a full run of three scored
submissions in which test rose at every step while validation did not rise.
"""

import pytest

from torchcell.benchmark.integrity import (
    IntegrityFlag,
    IntegrityPolicy,
    flag_submission,
    oriented,
)

POLICY = IntegrityPolicy()
EXCEEDS = IntegrityFlag.TEST_EXCEEDS_VAL
DIVERGES = IntegrityFlag.VAL_TEST_DIVERGENCE


def test_defaults() -> None:
    assert POLICY.tolerance == 0.02
    assert POLICY.divergence_run == 3
    assert EXCEEDS.value == "test_exceeds_val"
    assert DIVERGES.value == "val_test_divergence"


def test_oriented_flips_error_metrics() -> None:
    assert oriented("pearson", 0.4) == 0.4
    assert oriented("r2", -0.5) == -0.5
    assert oriented("mse", 0.4) == -0.4
    assert oriented("mae", 2.0) == -2.0


@pytest.mark.parametrize(
    ("val", "test", "expected"),
    [
        (0.50, 0.50, []),
        (0.50, 0.40, []),
        (0.50, 0.515, []),  # inside the tolerance
        (0.50, 0.53, [EXCEEDS]),
    ],
)
def test_exceeds_val_boundary(
    val: float, test: float, expected: list[IntegrityFlag]
) -> None:
    assert flag_submission("pearson", val, test, [], POLICY) == expected


def test_exceeds_val_for_an_error_metric() -> None:
    assert flag_submission("mse", 1.00, 0.90, [], POLICY) == [EXCEEDS]
    assert flag_submission("mse", 0.90, 1.00, [], POLICY) == []


def test_divergence_needs_a_full_run() -> None:
    # one earlier submission: only two points, no run of three
    assert flag_submission("pearson", 0.59, 0.52, [(0.60, 0.50)], POLICY) == []
    history = [(0.60, 0.50), (0.60, 0.52)]
    assert flag_submission("pearson", 0.58, 0.54, history, POLICY) == [DIVERGES]


def test_divergence_breaks_when_validation_rises_or_test_stalls() -> None:
    history = [(0.60, 0.50), (0.60, 0.52)]
    assert flag_submission("pearson", 0.61, 0.54, history, POLICY) == []
    assert flag_submission("pearson", 0.58, 0.52, history, POLICY) == []


def test_divergence_looks_only_at_the_last_run() -> None:
    history = [(0.10, 0.90), (0.60, 0.50), (0.60, 0.52)]
    assert flag_submission("pearson", 0.58, 0.54, history, POLICY) == [DIVERGES]


def test_both_flags_together() -> None:
    history = [(0.50, 0.50), (0.50, 0.55)]
    assert flag_submission("pearson", 0.50, 0.60, history, POLICY) == [
        EXCEEDS,
        DIVERGES,
    ]


def test_policy_is_configurable() -> None:
    loose = IntegrityPolicy(tolerance=0.2, divergence_run=2)
    assert flag_submission("pearson", 0.5, 0.6, [], loose) == []
    assert flag_submission("pearson", 0.5, 0.6, [(0.5, 0.55)], loose) == [DIVERGES]
