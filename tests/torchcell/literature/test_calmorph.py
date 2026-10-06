# tests/torchcell/literature/test_calmorph.py
# [[tests.torchcell.literature.test_calmorph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_calmorph.py

"""Tests for torchcell.literature.calmorph."""

import os
import re

import pytest

import torchcell.literature.calmorph as calmorph
from torchcell.literature.calmorph import (
    extract_calmorph_parameters,
    parse_calmorph_table,
)

# A born-digital "table1 parameter description" PDF (Ohya 2005 SI). The full
# round-trip test against the manual schema skips cleanly when it is not set.
_SAMPLE_PDF = os.environ.get("TORCHCELL_SAMPLE_PDF")

# A miniature ``pdftotext -layout`` rendering exercising every row shape:
# a stage header row, a continuation row (sparse stage column), a CV variant,
# the suffix-less Total_stage block, and the lone source typo whose description
# carries a single embedded space ("..._in_whole cell").
_LAYOUT = """\
Table 1 501 parameters
  Nuclear_stage       No.            ID                 Description                 Definition
     Stage_A           1    C11-1_A              Whole_cell_size                        -
                       2    C12-1_A         Whole_cell_outline_length                   -
                      39    CCV11-1_A     Coefficient_of_variation_of_C11-1_A           -
   Total_stage        462   C119                no_bud_ratio                            -
                      346   D196_C    Maximal_intensity_of_nuclear_brightness_in_whole cell    D16-3
"""


def test_parse_handles_all_row_shapes():
    params = parse_calmorph_table(_LAYOUT)
    assert params == {
        "C11-1_A": "Whole_cell_size",
        "C12-1_A": "Whole_cell_outline_length",
        "CCV11-1_A": "Coefficient_of_variation_of_C11-1_A",
        "C119": "no_bud_ratio",  # Total_stage: suffix-less id accepted
        # single embedded space normalized to underscore, "cell" not lost to Definition
        "D196_C": "Maximal_intensity_of_nuclear_brightness_in_whole_cell",
    }


def test_title_and_header_lines_are_not_rows():
    params = parse_calmorph_table(_LAYOUT)
    assert "501" not in params
    assert "ID" not in params


@pytest.mark.skipif(_SAMPLE_PDF is None, reason="TORCHCELL_SAMPLE_PDF not set")
def test_extract_reproduces_manual_schema_exactly():
    # The provable claim: automated extraction == the hand-built calmorph schema.
    from torchcell.datamodels.calmorph_labels import CALMORPH_PARAMETERS

    assert _SAMPLE_PDF is not None
    extracted = extract_calmorph_parameters(_SAMPLE_PDF)
    assert extracted == CALMORPH_PARAMETERS


# 2026.10.06 (Phase 21): ``extract_calmorph_parameters`` with the poppler helpers
# replaced at their import site in ``calmorph`` (no PDF, no subprocess).
def test_extract_refuses_a_scanned_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    texts: list[str] = []

    def fake_text(p: object, layout: bool) -> str:
        texts.append(str(p))
        return ""

    monkeypatch.setattr(calmorph, "pdf_kind", lambda p: "scanned")
    monkeypatch.setattr(calmorph, "pdf_text", fake_text)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "table1.pdf is 'scanned'; this recipe needs a born-digital PDF with a "
            "trustworthy text layer (route scans through OCR)."
        ),
    ):
        calmorph.extract_calmorph_parameters("/x/table1.pdf")
    assert texts == []  # the text layer is never read for a scan


def test_extract_reads_the_layout_text_of_a_born_digital_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_text(p: object, layout: bool) -> str:
        calls.append((str(p), layout))
        return _LAYOUT

    monkeypatch.setattr(calmorph, "pdf_kind", lambda p: "born_digital")
    monkeypatch.setattr(calmorph, "pdf_text", fake_text)
    assert calmorph.extract_calmorph_parameters("/x/table1.pdf") == {
        "C11-1_A": "Whole_cell_size",
        "C12-1_A": "Whole_cell_outline_length",
        "CCV11-1_A": "Coefficient_of_variation_of_C11-1_A",
        "C119": "no_bud_ratio",
        "D196_C": "Maximal_intensity_of_nuclear_brightness_in_whole_cell",
    }
    assert calls == [("/x/table1.pdf", True)]


def test_extract_refuses_a_text_layer_with_no_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calmorph, "pdf_kind", lambda p: "born_digital")
    monkeypatch.setattr(
        calmorph, "pdf_text", lambda p, layout: "Table 1\nNo. ID Description\n"
    )
    with pytest.raises(
        ValueError, match=re.escape("No CalMorph parameters parsed from table1.pdf")
    ):
        calmorph.extract_calmorph_parameters("/x/table1.pdf")
