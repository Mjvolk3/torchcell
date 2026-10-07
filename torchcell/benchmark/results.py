# torchcell/benchmark/results.py
# [[torchcell.benchmark.results]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/results.py
# Test file: tests/torchcell/benchmark/test_results.py

"""What the benchmark service answers to a submitter: a graded attempt and the quota.

The server (:mod:`torchcell.benchmark.app`) returns these models and the client
(:mod:`torchcell.benchmark.client`) parses them, so the two cannot drift apart.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from torchcell.benchmark.db import SubmissionStatus
from torchcell.benchmark.grading import SplitScores


class Quota(BaseModel):
    """An account's submission quota."""

    max_per_window: int
    window_hours: float
    min_gap_minutes: float
    used_in_window: int
    remaining: int
    next_allowed_at: datetime | None


class SubmissionResult(BaseModel):
    """One attempt as its owner sees it: scores, or the reasons it was rejected."""

    submission_id: str
    dataset_slug: str
    status: SubmissionStatus
    submitted_at: datetime
    method_name: str
    rejection_reasons: list[str]
    val: SplitScores | None
    test: SplitScores | None
    flags: list[str]
    archive_sha256: str | None
