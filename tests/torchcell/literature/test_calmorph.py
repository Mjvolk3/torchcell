# tests/torchcell/literature/test_calmorph.py
# [[tests.torchcell.literature.test_calmorph]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/literature/test_calmorph.py

"""Tests for torchcell.literature.calmorph."""

import os

import pytest

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
