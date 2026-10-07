# torchcell/benchmark/validation.py
# [[torchcell.benchmark.validation]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/benchmark/validation.py
# Test file: tests/torchcell/benchmark/test_validation.py

"""Validate an uploaded predictions CSV against a dataset's public template.

Validation needs only public information (the template's set of scored pairs), so a
submitter runs the same check locally before spending an attempt::

    python -m torchcell.benchmark.validation predictions.csv --template template.csv

A file is accepted when it is UTF-8, its header is exactly ``record_id,split,target,
prediction``, every row parses as a :class:`PredictionRow`, every (record_id, target)
pair of the template appears exactly once with the template's split, no other pair
appears, and no (split, target) column of predictions is constant (its correlation
would be undefined). Anything else is rejected with reasons that name the line. At most
:data:`MAX_REPORTED_REASONS` reasons are listed; the rest are counted.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from torchcell.benchmark.bundle import SubmissionSpec
from torchcell.benchmark.grading import PairKey
from torchcell.benchmark.submission import PREDICTION_COLUMNS, PredictionRow, Split

MAX_REPORTED_REASONS = 20
# A file may carry bad rows on top of the expected ones; reading stops past this slack
# so a huge junk upload is rejected without being parsed to the end.
MAX_EXTRA_ROWS = 1000


class ValidationReport(BaseModel):
    """The outcome of validating one predictions file."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    n_rows: int
    reasons: list[str]


class _Reasons:
    """Collects rejection reasons, listing the first few and counting the rest."""

    def __init__(self) -> None:
        self.listed: list[str] = []
        self.total = 0

    def add(self, reason: str) -> None:
        self.total += 1
        if len(self.listed) < MAX_REPORTED_REASONS:
            self.listed.append(reason)

    def final(self) -> list[str]:
        hidden = self.total - len(self.listed)
        if hidden:
            return [*self.listed, f"and {hidden} further problems not listed"]
        return list(self.listed)


def _rejected(reason: str) -> tuple[ValidationReport, None]:
    return ValidationReport(ok=False, n_rows=0, reasons=[reason]), None


def validate_predictions(
    raw: bytes, spec: SubmissionSpec
) -> tuple[ValidationReport, dict[PairKey, float] | None]:
    """Validate ``raw`` against ``spec``; return the report and, when ok, the predictions."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        return _rejected(f"file is not UTF-8 text (byte {error.start})")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None:
        return _rejected("file is empty")
    if tuple(header) != PREDICTION_COLUMNS:
        return _rejected(
            f"header must be exactly {','.join(PREDICTION_COLUMNS)}; got {','.join(header)}"
        )

    reasons = _Reasons()
    predictions: dict[PairKey, float] = {}
    n_rows = 0
    row_limit = len(spec.expected) + MAX_EXTRA_ROWS
    for line, row in enumerate(reader, start=2):
        n_rows += 1
        if n_rows > row_limit:
            reasons.add(
                f"more than {row_limit} rows; the template has {len(spec.expected)}"
            )
            break
        if len(row) != len(PREDICTION_COLUMNS):
            reasons.add(f"line {line}: {len(row)} fields, expected 4")
            continue
        try:
            parsed = PredictionRow.model_validate(dict(zip(PREDICTION_COLUMNS, row)))
        except ValidationError as error:
            first = error.errors()[0]
            reasons.add(f"line {line}: {first['loc'][0]}: {first['msg']}")
            continue
        key = (parsed.record_id, parsed.target)
        expected_split = spec.expected.get(key)
        if expected_split is None:
            reasons.add(
                f"line {line}: ({parsed.record_id}, {parsed.target}) is not in the template"
            )
        elif expected_split != parsed.split:
            reasons.add(
                f"line {line}: {parsed.record_id} is in the {expected_split} split, "
                f"not {parsed.split}"
            )
        elif key in predictions:
            reasons.add(
                f"line {line}: duplicate row for ({parsed.record_id}, {parsed.target})"
            )
        else:
            predictions[key] = parsed.prediction

    missing = [key for key in spec.expected if key not in predictions]
    if missing and n_rows <= row_limit:
        shown = ", ".join(f"({r}, {t})" for r, t in missing[:3])
        reasons.add(
            f"{len(missing)} template pairs have no valid row, for example {shown}"
        )

    if not reasons.total:
        columns: dict[tuple[Split, str], set[float]] = {}
        for key, value in predictions.items():
            columns.setdefault((spec.expected[key], key[1]), set()).add(value)
        for (split, target), distinct in sorted(columns.items()):
            if len(distinct) == 1:
                reasons.add(
                    f"predictions for target {target} on the {split} split are constant, "
                    "so their correlation is undefined"
                )

    if reasons.total:
        return ValidationReport(ok=False, n_rows=n_rows, reasons=reasons.final()), None
    return ValidationReport(ok=True, n_rows=n_rows, reasons=[]), predictions


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: validate a predictions CSV against a downloaded ``template.csv``."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = SubmissionSpec.from_template_csv(args.template.read_text(encoding="utf-8"))
    report, _ = validate_predictions(args.predictions.read_bytes(), spec)
    print(report.model_dump_json(indent=2))
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
