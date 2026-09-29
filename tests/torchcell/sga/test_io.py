# tests/torchcell/sga/test_io.py
# [[tests.torchcell.sga.test_io]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_io.py
"""Readers for gitter DAT files and ECHO picklists, and the (row, col) layout merge.

Well decoding is bijective base 26: B2 -> (2, 2), P24 -> (16, 24), AA13 -> (27, 13),
AF1 -> (26 + 6, 1) = (32, 1). A gitter row with only three whitespace-separated fields
has NaN circularity and an empty flag string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torchcell.sga.io import (
    merge_layout,
    read_echo_picklist,
    read_gitter_dat,
    well_to_rowcol,
)


@pytest.mark.parametrize(
    ("well", "expected"),
    [
        ("A1", (1, 1)),
        ("B2", (2, 2)),
        ("P24", (16, 24)),
        ("Z1", (26, 1)),
        ("AA13", (27, 13)),
        ("AF1", (32, 1)),
    ],
)
def test_well_to_rowcol_base26(well: str, expected: tuple[int, int]) -> None:
    """Letters are bijective base 26 (Z = 26, AA = 27); digits are the column."""
    assert well_to_rowcol(well) == expected


def test_well_to_rowcol_normalizes_case_and_whitespace() -> None:
    """Lowercase and surrounding blanks are accepted; a digit-first or empty label is not."""
    assert well_to_rowcol(" b2 ") == (2, 2)
    with pytest.raises(ValueError, match="unparseable well label: '2B'"):
        well_to_rowcol("2B")
    with pytest.raises(ValueError, match="unparseable well label: ''"):
        well_to_rowcol("")


def test_read_gitter_dat_parses_comments_and_short_rows(tmp_path: Path) -> None:
    """Comment lines are skipped; tab and space separators mix; a 4-field row gets flag
    '' and a 3-field row also NaN circularity; sizes are numeric.
    """
    path = str(tmp_path / "plate.dat")
    with open(path, "w") as fh:
        fh.write("# gitter output\n# row col size circ flags\n")
        fh.write("1\t1\t345\t0.97\tC\n")
        fh.write("1 2 0 NA S\n")
        fh.write("1\t3\t410\t0.99\n")
        fh.write("2 1 380\n")
    df = read_gitter_dat(path)
    assert list(df.columns) == ["row", "col", "size", "circularity", "flags"]
    assert df["row"].tolist() == [1, 1, 1, 2]
    assert df["col"].tolist() == [1, 2, 3, 1]
    assert df["size"].tolist() == [345.0, 0.0, 410.0, 380.0]
    circ = df["circularity"].to_numpy()
    assert (
        circ[0] == 0.97 and np.isnan(circ[1]) and circ[2] == 0.99 and np.isnan(circ[3])
    )
    assert df["flags"].tolist() == ["C", "S", "", ""]
    assert df["row"].dtype == np.int64


def test_read_echo_picklist_decodes_wells_and_coerces_volumes(tmp_path: Path) -> None:
    """Destination Well -> (row, col); Sample Name -> strain (as str); Transfer Volume
    numeric with a non-number coerced to NaN; the original well label is kept.
    """
    path = str(tmp_path / "picklist.csv")
    pd.DataFrame(
        {
            "Source Well": ["A1", "A2", "A3"],
            "Destination Well": ["B2", "P24", "AA3"],
            "Sample Name": ["BY4741", "gene1", 42],
            "Transfer Volume": ["2.5", 5, "n/a"],
        }
    ).to_csv(path, index=False)
    layout = read_echo_picklist(path)
    assert list(layout.columns) == ["row", "col", "strain", "volume_nl", "well"]
    assert layout["row"].tolist() == [2, 16, 27]
    assert layout["col"].tolist() == [2, 24, 3]
    assert layout["strain"].tolist() == ["BY4741", "gene1", "42"]
    vol = layout["volume_nl"].to_numpy()
    assert vol[0] == 2.5 and vol[1] == 5.0 and np.isnan(vol[2])
    assert layout["well"].tolist() == ["B2", "P24", "AA3"]


def test_read_echo_picklist_names_missing_columns(tmp_path: Path) -> None:
    """The error lists the absent required columns, sorted."""
    path = str(tmp_path / "bad.csv")
    pd.DataFrame({"Destination Well": ["A1"]}).to_csv(path, index=False)
    with pytest.raises(
        ValueError, match=r"missing columns: \['Sample Name', 'Transfer Volume'\]"
    ):
        read_echo_picklist(path)


def test_merge_layout_left_joins_on_row_col() -> None:
    """Every colony survives; a colony with no layout entry has NaN strain/volume/well;
    the strain is attached by (row, col) not by order.
    """
    dat = pd.DataFrame({"row": [1, 1, 2], "col": [1, 2, 1], "size": [10.0, 20.0, 30.0]})
    layout = pd.DataFrame(
        {
            "row": [2, 1],
            "col": [1, 1],
            "strain": ["gene1", "BY4741"],
            "volume_nl": [5.0, 2.5],
            "well": ["B1", "A1"],
        }
    )
    out = merge_layout(dat, layout)
    assert len(out) == 3
    assert out["size"].tolist() == [10.0, 20.0, 30.0]
    strains = out["strain"].tolist()
    assert (strains[0], strains[2]) == ("BY4741", "gene1") and pd.isna(strains[1])
    assert out["volume_nl"].tolist()[0] == 2.5 and pd.isna(out["well"].tolist()[1])


def test_merge_layout_without_layout_adds_null_columns() -> None:
    """No layout: strain None, volume_nl and well NA, sizes untouched, input not mutated."""
    dat = pd.DataFrame({"row": [1], "col": [1], "size": [10.0]})
    out = merge_layout(dat, None)
    assert list(out.columns) == ["row", "col", "size", "strain", "volume_nl", "well"]
    assert out["strain"].tolist() == [None]
    assert out["volume_nl"].isna().all() and out["well"].isna().all()
    assert list(dat.columns) == ["row", "col", "size"]
