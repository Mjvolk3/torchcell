# tests/torchcell/benchmark/test_results.py
# [[tests.torchcell.benchmark.test_results]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_results.py
"""``torchcell.benchmark.results``: the two answers a submitter parses.

A rejected attempt has reasons and no scores; a scored one has both splits. A status
outside the life cycle, or a missing field, does not validate, so a client fails on a
response it does not understand instead of reading a default.
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.benchmark.db import SubmissionStatus
from torchcell.benchmark.results import Quota, SubmissionResult

REJECTED: dict[str, Any] = {
    "submission_id": "a" * 32,
    "dataset_slug": "toy-fitness",
    "status": "rejected",
    "submitted_at": "2026-10-01T12:00:00Z",
    "method_name": "ridge",
    "rejection_reasons": ["row 3: prediction is not a number"],
    "val": None,
    "test": None,
    "flags": [],
    "archive_sha256": None,
}
METRICS = {"pearson": 0.8, "spearman": 0.8, "mse": 0.5, "mae": 0.5, "r2": 0.6}
SCORES = {"n_records": 4, "macro": METRICS, "per_target": {"fitness": METRICS}}


def test_rejected_result() -> None:
    result = SubmissionResult.model_validate(REJECTED)
    assert result.status is SubmissionStatus.REJECTED
    assert result.submitted_at == datetime(2026, 10, 1, 12, tzinfo=UTC)
    assert result.rejection_reasons == ["row 3: prediction is not a number"]
    assert (result.val, result.test) == (None, None)


def test_scored_result_round_trips() -> None:
    scored = {
        **REJECTED,
        "status": "provisional",
        "rejection_reasons": [],
        "val": SCORES,
        "test": SCORES,
        "archive_sha256": "b" * 64,
    }
    result = SubmissionResult.model_validate(scored)
    assert result.val is not None and result.test is not None
    assert result.val.macro.pearson == 0.8
    assert result.test.per_target["fitness"].r2 == 0.6
    assert result.model_dump(mode="json") == scored


@pytest.mark.parametrize(
    "broken",
    [
        {**REJECTED, "status": "accepted"},
        {k: v for k, v in REJECTED.items() if k != "rejection_reasons"},
    ],
)
def test_result_outside_the_contract_does_not_validate(broken: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SubmissionResult.model_validate(broken)


def test_quota() -> None:
    quota = Quota.model_validate(
        {
            "max_per_window": 3,
            "window_hours": 24.0,
            "min_gap_minutes": 60.0,
            "used_in_window": 1,
            "remaining": 2,
            "next_allowed_at": "2026-10-01T13:00:00Z",
        }
    )
    assert quota.next_allowed_at == datetime(2026, 10, 1, 13, tzinfo=UTC)
    assert (quota.used_in_window, quota.remaining) == (1, 2)
    with pytest.raises(ValidationError):
        Quota.model_validate({"max_per_window": 3})
