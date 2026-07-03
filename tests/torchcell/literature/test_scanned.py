# tests/torchcell/literature/test_scanned.py
# [[tests.torchcell.literature.test_scanned]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_scanned.py

"""Tests for torchcell.literature.scanned."""

from collections.abc import Callable
from pathlib import Path

import torchcell.literature.scanned as scanned
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


def _fake_ocr_factory(
    tmp_path: Path, text_by_dpi: dict[int, str]
) -> Callable[..., Path]:
    """Return an ocr_pdf stand-in that writes per-dpi text to a markdown file."""

    def fake_ocr_pdf(pdf_path, *, backend, dpi, **kwargs):
        md = tmp_path / f"pass_{dpi}.md"
        md.write_text(text_by_dpi[dpi])
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
