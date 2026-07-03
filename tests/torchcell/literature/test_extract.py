# tests/torchcell/literature/test_extract.py
# [[tests.torchcell.literature.test_extract]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_extract.py

"""Tests for torchcell.literature.extract."""

import os

import pytest

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
