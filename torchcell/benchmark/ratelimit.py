# torchcell/benchmark/ratelimit.py
# [[torchcell.benchmark.ratelimit]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/ratelimit.py
# Test file: tests/torchcell/benchmark/test_ratelimit.py

"""The per-account submission quota.

Two rules, both over an account's attempts on every dataset together: at most
``max_per_window`` attempts in any rolling ``window`` (3 in 24 hours), and at least
``min_gap`` between two attempts (1 hour). A rejected upload is an attempt, which is
why the validator can be run locally first. The function is pure: the caller passes the
attempt times it read under a row lock, so two concurrent uploads cannot both pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field


class SubmissionLimits(BaseModel):
    """The quota's three numbers."""

    model_config = ConfigDict(frozen=True)

    max_per_window: int = Field(default=3, ge=1)
    window: timedelta = timedelta(hours=24)
    min_gap: timedelta = timedelta(hours=1)


class QuotaStatus(BaseModel):
    """Whether an attempt is allowed now, and when the next one is if it is not."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    used_in_window: int
    remaining: int
    next_allowed_at: datetime | None
    reason: str | None


def evaluate_quota(
    attempts: Sequence[datetime], now: datetime, limits: SubmissionLimits
) -> QuotaStatus:
    """Apply ``limits`` to an account's attempt times (timezone-aware) at ``now``."""
    in_window = sorted(t for t in attempts if t > now - limits.window)
    used = len(in_window)
    blocks: list[tuple[datetime, str]] = []
    if in_window and in_window[-1] + limits.min_gap > now:
        minutes = int(limits.min_gap.total_seconds() // 60)
        blocks.append(
            (
                in_window[-1] + limits.min_gap,
                f"submissions must be {minutes} minutes apart",
            )
        )
    if used >= limits.max_per_window:
        hours = int(limits.window.total_seconds() // 3600)
        # The attempt whose expiry brings the count back under the maximum.
        releasing = in_window[used - limits.max_per_window]
        blocks.append(
            (
                releasing + limits.window,
                f"{limits.max_per_window} submissions are allowed per {hours} hours",
            )
        )
    if not blocks:
        return QuotaStatus(
            allowed=True,
            used_in_window=used,
            remaining=limits.max_per_window - used,
            next_allowed_at=None,
            reason=None,
        )
    next_allowed_at, reason = max(blocks)
    return QuotaStatus(
        allowed=False,
        used_in_window=used,
        remaining=max(limits.max_per_window - used, 0),
        next_allowed_at=next_allowed_at,
        reason=reason,
    )
