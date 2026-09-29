# tests/torchcell/sga/test_register.py
# [[tests.torchcell.sga.test_register]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_register.py
"""Orientation registration of an image-order grid to the plate layout.

A 2x3 image grid whose only empty spot is image (1, 1). The layout lives in plate
coordinates with the inner block starting at (2, 2), and its single ``Blank_media`` is
plate (3, 4). Image (1, 1) reaches plate (3, 4) only under rot180: nr = 2 + 1 - 1 = 2,
nc = 3 + 1 - 1 = 3, then +1 for the inner offset -> (3, 4). Under identity it lands on
plate (2, 2) (plated but empty) while the blank at (3, 4) maps back to a grown colony,
so identity agrees on 4 of 6 wells, flips on 4 of 6, rot180 on all 6.
"""

from __future__ import annotations

import pandas as pd
import pytest

from torchcell.sga.register import _orient, resolve_orientation

GRID = pd.DataFrame(
    {
        "row": [1, 1, 1, 2, 2, 2],
        "col": [1, 2, 3, 1, 2, 3],
        "size": [0, 100, 100, 100, 100, 100],
    }
)
LAYOUT = pd.DataFrame(
    {
        "row": [2, 2, 2, 3, 3, 3],
        "col": [2, 3, 4, 2, 3, 4],
        "strain": ["a", "b", "c", "d", "e", "Blank_media"],
    }
)


@pytest.mark.parametrize(
    ("op", "rows", "cols"),
    [
        ("identity", [1, 1, 1, 2, 2, 2], [1, 2, 3, 1, 2, 3]),
        ("rot180", [2, 2, 2, 1, 1, 1], [3, 2, 1, 3, 2, 1]),
        ("flip_v", [2, 2, 2, 1, 1, 1], [1, 2, 3, 1, 2, 3]),
        ("flip_h", [1, 1, 1, 2, 2, 2], [3, 2, 1, 3, 2, 1]),
    ],
)
def test_orient_dihedral_ops(op: str, rows: list[int], cols: list[int]) -> None:
    """rot180 reverses both axes (n + 1 - i), flip_v rows only, flip_h columns only; the
    size column and the input frame are untouched.
    """
    out = _orient(GRID, 2, 3, op)
    assert out["row"].tolist() == rows
    assert out["col"].tolist() == cols
    assert out["size"].tolist() == GRID["size"].tolist()
    assert GRID["row"].tolist() == [1, 1, 1, 2, 2, 2]


def test_orient_rejects_unknown_op() -> None:
    """The offending op is the error message."""
    with pytest.raises(ValueError, match="^transpose$"):
        _orient(GRID, 2, 3, "transpose")


def test_resolve_orientation_picks_rot180() -> None:
    """Only rot180 puts the empty image spot on the blank well: agreement 1.0, and the
    merged frame carries plate coordinates with the blank on the size-0 colony.
    """
    merged, op, agree = resolve_orientation(GRID, LAYOUT, n_rows=2, n_cols=3)
    assert (op, agree) == ("rot180", 1.0)
    assert len(merged) == 6
    rows = sorted(
        zip(merged["row"], merged["col"], merged["size"], merged["strain"], strict=True)
    )
    assert rows == [
        (2, 2, 100, "a"),
        (2, 3, 100, "b"),
        (2, 4, 100, "c"),
        (3, 2, 100, "d"),
        (3, 3, 100, "e"),
        (3, 4, 0, "Blank_media"),
    ]


def test_resolve_orientation_agreement_values_and_tie_break() -> None:
    """With every colony present the blank disagrees under all four ops (5/6 each) and
    the first op, identity, wins the tie; a custom blank name and threshold flow through:
    with ``empty_thresh=100`` nothing counts as present, so only the blank agrees (1/6).
    """
    full = GRID.assign(size=100)
    _, op, agree = resolve_orientation(full, LAYOUT, n_rows=2, n_cols=3)
    assert op == "identity"
    assert agree == pytest.approx(5 / 6)
    _, op2, agree2 = resolve_orientation(
        full, LAYOUT, n_rows=2, n_cols=3, empty_thresh=100.0
    )
    assert op2 == "identity"
    assert agree2 == pytest.approx(1 / 6)
    renamed = LAYOUT.replace({"strain": {"Blank_media": "none"}})
    _, op3, agree3 = resolve_orientation(
        GRID, renamed, n_rows=2, n_cols=3, blank_name="none"
    )
    assert (op3, agree3) == ("rot180", 1.0)


def test_resolve_orientation_inner_offset() -> None:
    """With ``inner_row0=inner_col0=1`` plate coordinates equal image coordinates, so only
    the layout wells (2, 2) and (2, 3) meet the image grid: under identity both hold grown
    colonies (agreement 1.0 on 2 wells); flip_v also scores 1.0 but identity is tried
    first and a tie keeps the first; rot180 maps image (1, 1) (empty) onto (2, 3) and
    scores 0.5.
    """
    merged, op, agree = resolve_orientation(
        GRID, LAYOUT, n_rows=2, n_cols=3, inner_row0=1, inner_col0=1
    )
    assert (op, agree) == ("identity", 1.0)
    assert sorted(zip(merged["row"], merged["col"], merged["strain"], strict=True)) == [
        (2, 2, "a"),
        (2, 3, "b"),
    ]
