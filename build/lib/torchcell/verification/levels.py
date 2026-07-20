# torchcell/verification/levels
# [[torchcell.verification.levels]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/levels
"""Reusable L0-L4 record-level checks (roadmap WS3).

Each function returns a :class:`LevelResult`. They are deliberately small,
composable building blocks: a per-dataset verifier (WS5-WS9) assembles the
relevant ones into a :class:`VerificationReport`. All checks operate on
pydantic/LMDB records -- no graph required (Phase A).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from typing import Any

from torchcell.verification.report import Level, LevelResult

# A validator turns one raw record into a model instance (or raises). Callers
# pass e.g. ``TypeAdapter(ExperimentType).validate_python`` or ``Model.model_validate``.
Validator = Callable[[Any], object]


def l0_structural(records: Iterable[Any], validator: Validator) -> LevelResult:
    """L0: every raw record validates against its schema model.

    Catches validation errors per record and reports them as data (this is the
    harness's job, not a silent fallback).
    """
    failures: list[dict[str, Any]] = []
    n = 0
    for i, record in enumerate(records):
        n += 1
        try:
            validator(record)
        except (ValueError, TypeError) as err:  # pydantic raises ValueError subclass
            failures.append({"index": i, "error": str(err)[:500]})
    passed = not failures
    msg = (
        f"{n} records validated"
        if passed
        else f"{len(failures)}/{n} records failed schema validation"
    )
    return LevelResult(
        level=Level.L0,
        name="structural",
        passed=passed,
        message=msg,
        details={
            "n_records": n,
            "n_failures": len(failures),
            "failures": failures[:10],
        },
    )


def l1_completeness(
    observed: Collection[Hashable],
    expected: Collection[Hashable],
    *,
    allow_extra: bool = True,
) -> LevelResult:
    """L1: observed keys cover the expected oracle set (nothing silently dropped)."""
    observed_set = set(observed)
    expected_set = set(expected)
    missing = expected_set - observed_set
    extra = observed_set - expected_set
    passed = not missing and (allow_extra or not extra)
    msg = (
        f"complete: {len(observed_set)}/{len(expected_set)} expected keys present"
        if passed
        else f"{len(missing)} missing"
        + ("" if allow_extra else f", {len(extra)} unexpected")
    )
    return LevelResult(
        level=Level.L1,
        name="completeness",
        passed=passed,
        message=msg,
        details={
            "n_observed": len(observed_set),
            "n_expected": len(expected_set),
            "missing": sorted(map(str, missing))[:20],
            "extra": sorted(map(str, extra))[:20],
        },
    )


def l1_count(observed: int, expected: int) -> LevelResult:
    """L1 variant: an exact record-count oracle (e.g. 4718 mutants + 122 WT)."""
    passed = observed == expected
    return LevelResult(
        level=Level.L1,
        name="count",
        passed=passed,
        message=f"observed {observed}, expected {expected}",
        details={"observed": observed, "expected": expected},
    )


def l2_value_fidelity(
    values: Iterable[float],
    *,
    allow_nan: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> LevelResult:
    """L2: numeric values are finite / in range / non-NaN as required."""
    n = 0
    bad: list[dict[str, Any]] = []
    for i, value in enumerate(values):
        n += 1
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            bad.append({"index": i, "value": repr(value), "reason": "non-numeric"})
            continue
        if math.isnan(value):
            if not allow_nan:
                bad.append({"index": i, "value": "nan", "reason": "nan"})
            continue
        if math.isinf(value):
            bad.append({"index": i, "value": repr(value), "reason": "inf"})
            continue
        if minimum is not None and value < minimum:
            bad.append({"index": i, "value": value, "reason": f"< {minimum}"})
        elif maximum is not None and value > maximum:
            bad.append({"index": i, "value": value, "reason": f"> {maximum}"})
    passed = not bad
    return LevelResult(
        level=Level.L2,
        name="value_fidelity",
        passed=passed,
        message=f"{n} values checked" if passed else f"{len(bad)}/{n} values invalid",
        details={"n_values": n, "n_bad": len(bad), "bad": bad[:20]},
    )


def l2_cross_method(
    a: Sequence[float], b: Sequence[float], *, tol: float = 1e-6
) -> LevelResult:
    """L2: two independently-derived aligned series agree within ``tol``."""
    if len(a) != len(b):
        return LevelResult(
            level=Level.L2,
            name="cross_method",
            passed=False,
            message=f"length mismatch: {len(a)} vs {len(b)}",
            details={"len_a": len(a), "len_b": len(b)},
        )
    disagreements: list[dict[str, Any]] = []
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > tol:
            disagreements.append({"index": i, "a": x, "b": y, "diff": abs(x - y)})
    passed = not disagreements
    return LevelResult(
        level=Level.L2,
        name="cross_method",
        passed=passed,
        message=(
            f"{len(a)} pairs agree within {tol}"
            if passed
            else f"{len(disagreements)}/{len(a)} pairs disagree > {tol}"
        ),
        details={
            "tol": tol,
            "n_disagreements": len(disagreements),
            "worst": disagreements[:20],
        },
    )


def l3_convention(name: str, holds: bool, *, detail: str = "") -> LevelResult:
    """L3: a named units/convention assertion (e.g. expression = log2(sample/ref))."""
    return LevelResult(
        level=Level.L3,
        name=name,
        passed=holds,
        message=detail or ("convention holds" if holds else "convention violated"),
        details={"detail": detail},
    )


def l4_cross_source(
    shared: Iterable[tuple[Hashable, float, float]], *, tol: float = 1e-6
) -> LevelResult:
    """L4: overlapping entities agree across two sources.

    ``shared`` is an iterable of ``(entity_id, value_source_a, value_source_b)``.
    """
    n = 0
    disagreements: list[dict[str, Any]] = []
    for key, va, vb in shared:
        n += 1
        if abs(va - vb) > tol:
            disagreements.append(
                {"entity": str(key), "a": va, "b": vb, "diff": abs(va - vb)}
            )
    passed = not disagreements
    return LevelResult(
        level=Level.L4,
        name="cross_source",
        passed=passed,
        message=(
            f"{n} overlapping entities agree within {tol}"
            if passed
            else f"{len(disagreements)}/{n} overlapping entities disagree > {tol}"
        ),
        details={
            "tol": tol,
            "n_overlap": n,
            "n_disagreements": len(disagreements),
            "worst": disagreements[:20],
        },
    )
