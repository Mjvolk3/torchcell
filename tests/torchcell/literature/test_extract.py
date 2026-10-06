# tests/torchcell/literature/test_extract.py
# [[tests.torchcell.literature.test_extract]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_extract.py

"""Tests for torchcell.literature.extract.

2026.10.06 (Phase 21): the poppler wrappers run with ``extract._run`` replaced by a
recorder that answers each exact command tuple with hand-written ``pdffonts``,
``pdfinfo``, ``pdftotext`` and ``pdfimages -list`` output (the tools' own column
layout); ``_run`` itself is checked against a stubbed ``subprocess.run``. No poppler
binary runs. Page-area arithmetic: a page of ``W x H`` pt seen at ``x, y`` ppi is
``W/72*x`` by ``H/72*y`` px; an image counts when its px area is at least half that.
"""

import os
import re
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

import torchcell.literature.extract as extract
from torchcell.literature.extract import iter_layout_rows, pdf_kind, pdf_text

# A real born-digital SI table PDF to exercise the poppler-backed functions.
# Set this to e.g. the Ohya 2005 "table1 parameter description" PDF. The tests
# that need a real PDF skip cleanly when it is not set.
_SAMPLE_PDF = os.environ.get("TORCHCELL_SAMPLE_PDF")


def test_iter_layout_rows_drops_blanks_and_tokenizes():
    text = "  Stage_A    1   C11-1_A   Whole_cell_size   -\n\n   \n2 C12-1_A x\n"
    rows = iter_layout_rows(text)
    assert rows == [
        ["Stage_A", "1", "C11-1_A", "Whole_cell_size", "-"],
        ["2", "C12-1_A", "x"],
    ]


@pytest.mark.skipif(_SAMPLE_PDF is None, reason="TORCHCELL_SAMPLE_PDF not set")
def test_born_digital_pdf_detected():
    assert _SAMPLE_PDF is not None
    assert pdf_kind(_SAMPLE_PDF) == "born_digital"


@pytest.mark.skipif(_SAMPLE_PDF is None, reason="TORCHCELL_SAMPLE_PDF not set")
def test_text_layer_nonempty():
    assert _SAMPLE_PDF is not None
    assert len(pdf_text(_SAMPLE_PDF, layout=True).split()) > 100


# --------------------------------------------------------------------------- #
# 2026.10.06 (Phase 21): the poppler wrappers with the runner stubbed. Every CLI
# answer below is hand-written in the tool's own layout; no subprocess runs.
# --------------------------------------------------------------------------- #
_PDFFONTS = """\
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
ABCDEE+Helvetica                     TrueType          WinAnsi          yes yes no      12  0
[none]                               Type 3            Custom           yes no  no      30  0

"""
_PDFFONTS_EMPTY = """\
name                                 type              encoding         emb sub uni object ID
------------------------------------ ----------------- ---------------- --- --- --- ---------
"""
_PDFINFO = "Producer:       pdfTeX-1.40.21\nPages:          4\nPage size:      612 x 792 pts (letter)\n"
# Page 1: 2550 x 3300 px at 300 ppi on a 612 x 792 pt page = 2550 x 3300 px page: ratio 1.0.
# Page 2: 600 x 400 at 150 ppi; page 1275 x 1650 px; ratio 240000 / 2103750 = 0.114.
# Page 3: an smask with ppi 0 is skipped. Page 4: two page-sized images count once.
_PDFIMAGES = """\
page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi size ratio
--------------------------------------------------------------------------------------------
   1     0 image    2550  3300  gray    1   8  jpeg   no        10  0   300   300  412K 4.9%
   2     1 image     600   400  rgb     3   8  jpeg   no        15  0   150   150   40K 5.6%
   3     2 smask    2550  3300  gray    1   8  image  no        16  0     0     0    1B 0.0%
   4     3 image    1275  1650  gray    1   8  jpeg   no        20  0   150   150  100K 4.0%
   4     4 image    1275  1650  gray    1   8  jpeg   no        21  0   150   150  100K 4.0%
   truncated row
"""


class _Poppler:
    """``_run`` stand-in: answers by command tuple, records every call."""

    def __init__(self, answers: dict[tuple[str, ...], str]) -> None:
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> str:
        self.calls.append(cmd)
        return self.answers[tuple(cmd)]


def _poppler(monkeypatch: pytest.MonkeyPatch, **over: str) -> _Poppler:
    pdf = "/x/p.pdf"
    answers = {
        ("pdffonts", pdf): over.get("fonts", _PDFFONTS),
        ("pdfinfo", pdf): over.get("info", _PDFINFO),
        ("pdftotext", pdf, "-"): over.get("text", " ".join(["word"] * 20)),
        ("pdftotext", "-layout", pdf, "-"): over.get("layout", "a   b\n"),
        ("pdfimages", "-list", pdf): over.get("images", _PDFIMAGES),
    }
    fake = _Poppler(answers)
    monkeypatch.setattr(extract, "_run", fake)
    return fake


def test_run_calls_subprocess_with_check_and_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="OUT")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert extract._run(["pdfinfo", "a.pdf"]) == "OUT"
    assert calls == [
        (["pdfinfo", "a.pdf"], {"capture_output": True, "text": True, "check": True})
    ]


def test_pdf_fonts_skips_the_two_header_lines_and_blank_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _poppler(monkeypatch)
    assert extract.pdf_fonts("/x/p.pdf") == ["ABCDEE+Helvetica", "[none]"]
    assert fake.calls == [["pdffonts", "/x/p.pdf"]]


def test_pdf_page_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _poppler(monkeypatch)
    assert extract.pdf_page_count("/x/p.pdf") == 4


def test_pdf_page_count_refuses_output_without_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _poppler(monkeypatch, info="Producer: x\n")
    with pytest.raises(
        ValueError, match=re.escape("pdfinfo reported no page count for /x/p.pdf")
    ):
        extract.pdf_page_count("/x/p.pdf")


def test_pdf_text_command_with_and_without_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _poppler(monkeypatch)
    assert extract.pdf_text("/x/p.pdf") == "a   b\n"
    assert extract.pdf_text("/x/p.pdf", layout=False) == " ".join(["word"] * 20)
    assert fake.calls == [
        ["pdftotext", "-layout", "/x/p.pdf", "-"],
        ["pdftotext", "/x/p.pdf", "-"],
    ]


def test_page_sized_image_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pages 1 and 4 carry a page-sized image (page 4 twice, counted once); page 2's
    ratio is 0.114 < 0.5; page 3's zero-ppi smask and the short row are skipped.
    """
    fake = _poppler(monkeypatch)
    assert extract._page_sized_image_count("/x/p.pdf") == 2
    assert fake.calls == [["pdfimages", "-list", "/x/p.pdf"], ["pdfinfo", "/x/p.pdf"]]


def test_page_sized_image_count_reads_each_axis_at_its_own_ppi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anisotropic ppi on a 612 x 792 pt page. Page 1 at x 300 / y 150 ppi is
    2550 x 1650 px (4,207,500); a 2550 x 1000 image is 0.606 (counts), but reading the
    height at x-ppi would give 8,415,000 and 0.303. Page 2 at x 150 / y 300 is
    1275 x 3300 px (4,207,500); a 1275 x 2000 image is 0.606 (counts), but reading the
    width at y-ppi would give 0.303. Both count only when each axis uses its own ppi.
    """
    rows = (
        "h\n-\n"
        "   1     0 image    2550  1000  gray    1   8  jpeg   no        10  0   300   150  1K 1%\n"
        "   2     1 image    1275  2000  gray    1   8  jpeg   no        11  0   150   300  1K 1%\n"
    )
    _poppler(monkeypatch, images=rows)
    assert extract._page_sized_image_count("/x/p.pdf") == 2


def test_page_sized_image_count_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At 72 ppi a 612 x 792 pt page is 612 x 792 px; 612 x 396 is exactly 0.5 (counts),
    612 x 395 is 0.4987 (does not).
    """
    rows = (
        "h\n-\n"
        "   1     0 image     612   396  gray    1   8  jpeg   no        10  0    72    72  1K 1%\n"
        "   2     1 image     612   395  gray    1   8  jpeg   no        11  0    72    72  1K 1%\n"
    )
    _poppler(monkeypatch, images=rows)
    assert extract._page_sized_image_count("/x/p.pdf") == 1


def test_page_sized_image_count_without_a_page_size_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _poppler(monkeypatch, info="Pages: 4\n")
    assert extract._page_sized_image_count("/x/p.pdf") == 0


def test_pdf_kind_no_fonts_is_scanned_before_any_other_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _poppler(monkeypatch, fonts=_PDFFONTS_EMPTY)
    assert extract.pdf_kind("/x/p.pdf") == "scanned"
    assert fake.calls == [["pdffonts", "/x/p.pdf"]]


def test_pdf_kind_word_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """19 words is below ``_MIN_TEXT_WORDS = 20``: scanned without counting images."""
    fake = _poppler(monkeypatch, text=" ".join(["w"] * 19))
    assert extract.pdf_kind("/x/p.pdf") == "scanned"
    assert [c[0] for c in fake.calls] == ["pdffonts", "pdftotext"]


def test_pdf_kind_page_sized_image_on_every_page_is_scanned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = "h\n-\n" + "".join(
        f"   {p}     {p} image    2550  3300  gray    1   8  jpeg   no        1{p}  0   300   300  1K 1%\n"
        for p in (1, 2, 3, 4)
    )
    _poppler(monkeypatch, images=rows)
    assert extract.pdf_kind("/x/p.pdf") == "scanned"


def test_pdf_kind_born_digital(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fonts, 20 words, and 2 page-sized images on 4 pages: born digital."""
    fake = _poppler(monkeypatch)
    assert extract.pdf_kind("/x/p.pdf") == "born_digital"
    assert [c[0] for c in fake.calls] == [
        "pdffonts",
        "pdftotext",
        "pdfinfo",
        "pdfimages",
        "pdfinfo",
    ]


def test_pdf_kind_calls_a_scan_with_one_clean_page_born_digital(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding: the module comment and ``pdf_kind`` docstring call a PDF scanned when a
    page-sized image appears on MOST pages, but the code requires one on EVERY page
    (``count >= max(1, pages)``). A 4-page scan with one text-only page (a publisher
    cover sheet) is classified born digital, so its OCR'd text layer is trusted.
    Pinned until the threshold matches "most" (extract.py:124). Reach: the only
    caller is calmorph on a born-digital SI, so latent today.
    """
    rows = "h\n-\n" + "".join(
        f"   {p}     {p} image    2550  3300  gray    1   8  jpeg   no        1{p}  0   300   300  1K 1%\n"
        for p in (2, 3, 4)
    )
    _poppler(monkeypatch, images=rows)
    assert extract.pdf_kind("/x/p.pdf") == "born_digital"


def test_page_sized_image_count_skips_a_zero_area_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``0 x 0 pts`` page gives a zero page area: the image is skipped, not divided by."""
    _poppler(monkeypatch, info="Pages: 1\nPage size:      0 x 0 pts\n")
    assert extract._page_sized_image_count("/x/p.pdf") == 0
