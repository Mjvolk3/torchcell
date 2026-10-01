# tests/torchcell/literature/test_scanned.py
# [[tests.torchcell.literature.test_scanned]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_scanned.py

"""Tests for torchcell.literature.scanned.

2026.10.01 (PR #585 review): the fake ``ocr_pdf`` writes what the real one writes,
``<stem>.md`` and ``<stem>_ocr_provenance.json`` beside the PDF, overwritten on every
pass. After the sweep the record holds every pass run under ``params["passes"]``.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

import torchcell.literature.scanned as scanned
from torchcell.literature.manifest import ProcessingRecord
from torchcell.literature.scanned import extract_scanned, shape_check


def test_shape_check_known_schema():
    rep = shape_check({"a", "b"}, expected_keys={"a", "b", "c"})
    assert rep.found == 2
    assert rep.expected == 3
    assert rep.missing_keys == ["c"]
    assert rep.complete is False

    rep_done = shape_check({"a", "b", "c"}, expected_keys={"a", "b", "c"})
    assert rep_done.complete is True
    assert rep_done.missing_keys == []


def test_shape_check_count_only():
    assert shape_check({"x", "y"}, expected_n=3).complete is False
    assert shape_check({"x", "y", "z"}, expected_n=3).complete is True


def _record(dpi: int) -> ProcessingRecord:
    return ProcessingRecord(
        processor="torchcell.literature.ocr.ocr_pdf",
        tool="mineru",
        version="2.7.6",
        params={"backend": "vlm-auto-engine", "dpi": dpi},
        input_sha256=["ab" * 32],
    )


def _fake_ocr_factory(
    tmp_path: Path, text_by_dpi: dict[int, str]
) -> Callable[..., Path]:
    """Return an ocr_pdf stand-in that writes per-dpi text and a per-pass record."""

    def fake_ocr_pdf(pdf_path, *, backend, dpi, **kwargs):
        md = Path(pdf_path).with_suffix(".md")
        md.write_text(text_by_dpi[dpi])
        md.with_name(f"{md.stem}_ocr_provenance.json").write_text(
            _record(dpi).model_dump_json()
        )
        return md

    return fake_ocr_pdf


def test_union_grows_and_stops_early(tmp_path, monkeypatch):
    # pass@250 finds a,b ; pass@350 adds c -> union complete after the 2nd pass.
    monkeypatch.setattr(
        scanned, "ocr_pdf", _fake_ocr_factory(tmp_path, {250: "a b", 350: "c"})
    )
    found, reports = extract_scanned(
        tmp_path / "x.pdf",
        parse_keys=lambda t: set(t.split()),
        dpis=(250, 350, 600),
        expected_keys={"a", "b", "c"},
    )
    assert found == {"a", "b", "c"}
    # third pass (600) is never run because the oracle cleared at 350.
    assert [dpi for dpi, _ in reports] == [250, 350]
    assert reports[0][1].complete is False
    assert reports[-1][1].complete is True


def test_stops_immediately_when_first_pass_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(scanned, "ocr_pdf", _fake_ocr_factory(tmp_path, {300: "a b c"}))
    found, reports = extract_scanned(
        tmp_path / "x.pdf",
        parse_keys=lambda t: set(t.split()),
        dpis=(300, 350),
        expected_n=3,
    )
    assert found == {"a", "b", "c"}
    assert len(reports) == 1


def test_the_record_beside_the_markdown_lists_every_pass(tmp_path, monkeypatch):
    """Two passes (250 then 350): the record left beside ``x.md`` is the 350 pass's
    record with ``params["passes"]`` holding both full records in order, not just
    the last pass that overwrote the file.
    """
    monkeypatch.setattr(
        scanned, "ocr_pdf", _fake_ocr_factory(tmp_path, {250: "a b", 350: "c"})
    )
    extract_scanned(
        tmp_path / "x.pdf",
        parse_keys=lambda t: set(t.split()),
        dpis=(250, 350),
        expected_n=3,
    )
    record = json.loads((tmp_path / "x_ocr_provenance.json").read_text())
    assert record == {
        "processor": "torchcell.literature.ocr.ocr_pdf",
        "tool": "mineru",
        "version": "2.7.6",
        "params": {
            "backend": "vlm-auto-engine",
            "dpi": 350,
            "passes": [_record(250).model_dump(), _record(350).model_dump()],
        },
        "input_sha256": ["ab" * 32],
    }


def test_an_empty_dpi_sweep_is_refused(tmp_path):
    with pytest.raises(ValueError) as refused:
        extract_scanned(tmp_path / "x.pdf", parse_keys=lambda t: set(), dpis=())
    assert str(refused.value) == "extract_scanned needs at least one DPI"
