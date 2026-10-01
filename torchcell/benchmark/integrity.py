# torchcell/benchmark/integrity.py
# [[torchcell.benchmark.integrity]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/integrity.py
# Test file: tests/torchcell/benchmark/test_integrity.py

"""Integrity flags on a submission's validation and test scores.

The labels of a published dataset are not secret, so the benchmark cannot prevent a
submitter from tuning on the test split. It reports validation and test together and
raises two flags, which mark a submission for review and never reject it:

- ``test_exceeds_val``: the primary metric is better on test than on validation by more
  than ``tolerance``. Selection on validation makes validation the optimistic number, so
  a clearly better test score is unusual.
- ``val_test_divergence``: across the submitter's last ``divergence_run`` scored
  submissions on the dataset, test improved at every step while validation did not.
  That is the trace left by iterating against the test split.

The two defaults in :class:`IntegrityPolicy` are policy choices that have not been
calibrated against real submissions; revisit them once the board has history. The
tolerance is in units of the primary metric, so it is meaningful for a bounded metric
(Pearson, Spearman, R2) and not for a scale-dependent one (MSE, MAE).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from torchcell.benchmark.grading import HIGHER_IS_BETTER, MetricName


class IntegrityFlag(StrEnum):
    """The flags a scored submission can carry."""

    TEST_EXCEEDS_VAL = "test_exceeds_val"
    VAL_TEST_DIVERGENCE = "val_test_divergence"


class IntegrityPolicy(BaseModel):
    """Thresholds for the two flags (uncalibrated defaults)."""

    model_config = ConfigDict(frozen=True)

    tolerance: float = Field(
        default=0.02, ge=0, description="Allowed test-over-validation margin."
    )
    divergence_run: int = Field(
        default=3, ge=2, description="Consecutive submissions that form a divergence."
    )


def oriented(metric: MetricName, value: float) -> float:
    """``value`` signed so that larger is always better."""
    return value if HIGHER_IS_BETTER[metric] else -value


def flag_submission(
    metric: MetricName,
    val: float,
    test: float,
    history: Sequence[tuple[float, float]],
    policy: IntegrityPolicy,
) -> list[IntegrityFlag]:
    """Flags for a new submission scoring ``val`` and ``test`` on the primary metric.

    ``history`` is the submitter's earlier scored submissions on the same dataset as
    ``(val, test)`` pairs of the primary metric, oldest first.
    """
    flags: list[IntegrityFlag] = []
    if oriented(metric, test) - oriented(metric, val) > policy.tolerance:
        flags.append(IntegrityFlag.TEST_EXCEEDS_VAL)
    run = [*history, (val, test)][-policy.divergence_run :]
    if len(run) == policy.divergence_run and all(
        oriented(metric, later[1]) > oriented(metric, earlier[1])
        and oriented(metric, later[0]) <= oriented(metric, earlier[0])
        for earlier, later in zip(run, run[1:])
    ):
        flags.append(IntegrityFlag.VAL_TEST_DIVERGENCE)
    return flags
