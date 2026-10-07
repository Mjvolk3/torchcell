# tests/torchcell/benchmark/test_ratelimit.py
# [[tests.torchcell.benchmark.test_ratelimit]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_ratelimit.py
"""``torchcell.benchmark.ratelimit``: three attempts per 24 hours, one hour apart.

Times are offsets from ``T0`` (2026-10-01 12:00 UTC). With attempts at 0 h, 2 h and
4 h the window is full until the first attempt leaves it, at T0 + 24 h exactly; an
attempt exactly one hour after the last one is allowed, one second earlier is not.
"""

from datetime import UTC, datetime, timedelta

from torchcell.benchmark.ratelimit import SubmissionLimits, evaluate_quota

T0 = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
LIMITS = SubmissionLimits()


def h(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def test_defaults() -> None:
    assert LIMITS.max_per_window == 3
    assert LIMITS.window == timedelta(hours=24)
    assert LIMITS.min_gap == timedelta(hours=1)


def test_no_attempts() -> None:
    assert evaluate_quota([], T0, LIMITS).model_dump() == {
        "allowed": True,
        "used_in_window": 0,
        "remaining": 3,
        "next_allowed_at": None,
        "reason": None,
    }


def test_gap_boundary() -> None:
    blocked = evaluate_quota([T0], h(1) - timedelta(seconds=1), LIMITS)
    assert blocked.allowed is False
    assert blocked.used_in_window == 1
    assert blocked.remaining == 2
    assert blocked.next_allowed_at == h(1)
    assert blocked.reason == "submissions must be 60 minutes apart"

    allowed = evaluate_quota([T0], h(1), LIMITS)
    assert allowed.allowed is True
    assert allowed.remaining == 2


def test_window_full_until_the_oldest_attempt_leaves() -> None:
    attempts = [h(0), h(2), h(4)]
    blocked = evaluate_quota(attempts, h(6), LIMITS)
    assert blocked.allowed is False
    assert blocked.used_in_window == 3
    assert blocked.remaining == 0
    assert blocked.next_allowed_at == h(24)
    assert blocked.reason == "3 submissions are allowed per 24 hours"

    still_blocked = evaluate_quota(attempts, h(24) - timedelta(seconds=1), LIMITS)
    assert still_blocked.next_allowed_at == h(24)

    freed = evaluate_quota(attempts, h(24), LIMITS)
    assert freed.allowed is True
    assert freed.used_in_window == 2
    assert freed.remaining == 1


def test_the_later_block_wins() -> None:
    # window full AND inside the gap: the window frees later than the gap does
    blocked = evaluate_quota([h(0), h(2), h(4)], h(4.5), LIMITS)
    assert blocked.next_allowed_at == h(24)
    # window frees at 0.5 h from now, but the gap still holds for an hour
    attempts = [h(0), h(1), h(23.5)]
    blocked_by_gap = evaluate_quota(attempts, h(23.75), LIMITS)
    assert blocked_by_gap.next_allowed_at == h(24.5)
    assert blocked_by_gap.reason == "submissions must be 60 minutes apart"


def test_order_of_attempts_does_not_matter() -> None:
    assert (
        evaluate_quota([h(4), h(0), h(2)], h(6), LIMITS).next_allowed_at
        == evaluate_quota([h(0), h(2), h(4)], h(6), LIMITS).next_allowed_at
    )


def test_custom_limits() -> None:
    once_a_day = SubmissionLimits(max_per_window=1, min_gap=timedelta(0))
    blocked = evaluate_quota([T0], h(12), once_a_day)
    assert blocked.next_allowed_at == h(24)
    assert blocked.reason == "1 submissions are allowed per 24 hours"
