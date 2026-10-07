# tests/torchcell/benchmark/test_validation.py
# [[tests.torchcell.benchmark.test_validation]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/benchmark/test_validation.py
"""``torchcell.benchmark.validation`` on the ``toy-fitness`` template.

Each rejection is asserted by its exact reason text, because that text is what a
submitter reads. Line numbers count the header as line 1. The reason list stops at 20
entries and then counts the rest, and reading stops 1000 rows past the template size.
"""

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from torchcell.benchmark.bundle import BenchmarkBundle
from torchcell.benchmark.validation import (
    MAX_EXTRA_ROWS,
    MAX_REPORTED_REASONS,
    main,
    validate_predictions,
)

Csv = Callable[[Mapping[tuple[str, str], float]], bytes]
HEADER = "record_id,split,target,prediction\n"
VALID_BODY = (
    "v1,val,fitness,1.0\nv2,val,fitness,2.5\nv3,val,fitness,3.0\nv4,val,fitness,4.0\n"
    "s1,test,fitness,0.5\ns2,test,fitness,1.0\ns3,test,fitness,1.5\ns4,test,fitness,2.5\n"
)


def _reasons(bundle: BenchmarkBundle, text: str) -> list[str]:
    report, predictions = validate_predictions(text.encode(), bundle.spec)
    assert not report.ok
    assert predictions is None
    return report.reasons


def test_valid_file_returns_predictions(bundle: BenchmarkBundle) -> None:
    report, predictions = validate_predictions(
        (HEADER + VALID_BODY).encode(), bundle.spec
    )
    assert report.model_dump() == {"ok": True, "n_rows": 8, "reasons": []}
    assert predictions is not None
    assert predictions[("v2", "fitness")] == 2.5
    assert len(predictions) == 8


def test_byte_order_mark_and_crlf_are_accepted(bundle: BenchmarkBundle) -> None:
    raw = b"\xef\xbb\xbf" + (HEADER + VALID_BODY).replace("\n", "\r\n").encode()
    report, _ = validate_predictions(raw, bundle.spec)
    assert report.ok


def test_not_utf8(bundle: BenchmarkBundle) -> None:
    report, _ = validate_predictions(b"record_id,\xff\xfe", bundle.spec)
    assert report.reasons == ["file is not UTF-8 text (byte 10)"]


def test_empty_and_wrong_header(bundle: BenchmarkBundle) -> None:
    assert _reasons(bundle, "") == ["file is empty"]
    assert _reasons(bundle, "record_id,split,prediction\nv1,val,1.0\n") == [
        "header must be exactly record_id,split,target,prediction; "
        "got record_id,split,prediction"
    ]


def test_row_level_reasons_name_the_line(bundle: BenchmarkBundle) -> None:
    body = (
        "v1,val,fitness,1.0\n"
        "v2,val,fitness\n"  # line 3: three fields
        "v3,val,fitness,nan\n"  # line 4: not finite
        "v4,test,fitness,4.0\n"  # line 5: wrong split
        "s1,test,fitness,0.5\n"
        "s1,test,fitness,0.6\n"  # line 7: duplicate
        "zz,test,fitness,1.0\n"  # line 8: unknown record
        "s2,test,growth,1.0\n"  # line 9: unknown target
        "s3,train,fitness,1.0\n"  # line 10: train is not a scored split
    )
    assert _reasons(bundle, HEADER + body) == [
        "line 3: 3 fields, expected 4",
        "line 4: prediction: Input should be a finite number",
        "line 5: v4 is in the val split, not test",
        "line 7: duplicate row for (s1, fitness)",
        "line 8: (zz, fitness) is not in the template",
        "line 9: (s2, growth) is not in the template",
        "line 10: split: Input should be <Split.VAL: 'val'> or <Split.TEST: 'test'>",
        "6 template pairs have no valid row, for example "
        "(s2, fitness), (s3, fitness), (s4, fitness)",
    ]


def test_missing_rows_are_counted(bundle: BenchmarkBundle) -> None:
    assert _reasons(bundle, HEADER + "v1,val,fitness,1.0\nv2,val,fitness,2.0\n") == [
        "6 template pairs have no valid row, for example "
        "(s1, fitness), (s2, fitness), (s3, fitness)"
    ]


def test_constant_predictions_are_rejected(bundle: BenchmarkBundle) -> None:
    body = (
        VALID_BODY.replace("s1,test,fitness,0.5", "s1,test,fitness,1.0")
        .replace("s3,test,fitness,1.5", "s3,test,fitness,1.0")
        .replace("s4,test,fitness,2.5", "s4,test,fitness,1.0")
    )
    assert _reasons(bundle, HEADER + body) == [
        "predictions for target fitness on the test split are constant, "
        "so their correlation is undefined"
    ]


def test_reason_list_is_capped(bundle: BenchmarkBundle) -> None:
    junk = "".join(f"x{i},val,fitness,1.0\n" for i in range(30))
    reasons = _reasons(bundle, HEADER + junk)
    assert len(reasons) == MAX_REPORTED_REASONS + 1
    assert reasons[0] == "line 2: (x0, fitness) is not in the template"
    # 30 unknown rows plus the missing-pairs reason, 20 of them listed
    assert reasons[-1] == "and 11 further problems not listed"


def test_reading_stops_past_the_row_limit(bundle: BenchmarkBundle) -> None:
    limit = len(bundle.spec.expected) + MAX_EXTRA_ROWS
    junk = "".join(f"x{i},val,fitness,1.0\n" for i in range(limit + 50))
    report, _ = validate_predictions((HEADER + junk).encode(), bundle.spec)
    assert report.n_rows == limit + 1
    assert (
        report.reasons[-1]
        == f"and {limit + 1 - MAX_REPORTED_REASONS} further problems not listed"
    )


def test_cli_exit_codes(
    datasets_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = datasets_root / "toy-fitness" / "template.csv"
    good = tmp_path / "good.csv"
    good.write_text(HEADER + VALID_BODY)
    with pytest.raises(SystemExit) as ok_exit:
        main([str(good), "--template", str(template)])
    assert ok_exit.value.code == 0
    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "n_rows": 8,
        "reasons": [],
    }

    bad = tmp_path / "bad.csv"
    bad.write_text(HEADER)
    with pytest.raises(SystemExit) as bad_exit:
        main([str(bad), "--template", str(template)])
    assert bad_exit.value.code == 1
    assert json.loads(capsys.readouterr().out)["reasons"] == [
        "8 template pairs have no valid row, for example "
        "(s1, fitness), (s2, fitness), (s3, fitness)"
    ]
