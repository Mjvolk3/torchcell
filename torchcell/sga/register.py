# torchcell/sga/register.py
# [[torchcell.sga.register]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/register
"""Register an image-order colony grid to the plate layout.

Image analysis returns colony sizes in IMAGE order (row 1 = top, col 1 = left).
Which image corner is plate A1 is unknown -- the plate can be loaded in any of
the 4 shape-preserving orientations (identity, 180 deg rotation, or a flip about
either axis). We resolve it WITHOUT a fiducial by exploiting the no-cell control:
the layout's ``Blank_media`` wells must land on EMPTY image spots, and plated
wells on colonies. The orientation maximizing that agreement is the true one.
"""

from __future__ import annotations

import pandas as pd


def _orient(df: pd.DataFrame, n_rows: int, n_cols: int, op: str) -> pd.DataFrame:
    """Apply a shape-preserving dihedral op to image-order (row, col)."""
    r, c = df["row"], df["col"]
    if op == "identity":
        nr, nc = r, c
    elif op == "rot180":
        nr, nc = n_rows + 1 - r, n_cols + 1 - c
    elif op == "flip_v":  # top<->bottom
        nr, nc = n_rows + 1 - r, c
    elif op == "flip_h":  # left<->right
        nr, nc = r, n_cols + 1 - c
    else:
        raise ValueError(op)
    out = df.copy()
    out["row"], out["col"] = nr, nc
    return out


def resolve_orientation(
    grid: pd.DataFrame,
    layout: pd.DataFrame,
    n_rows: int = 14,
    n_cols: int = 22,
    inner_row0: int = 2,
    inner_col0: int = 2,
    blank_name: str = "Blank_media",
    empty_thresh: float = 1.0,
) -> tuple[pd.DataFrame, str, float]:
    """Return (merged, best_op, agreement).

    ``grid`` is the image-order quantification (row 1..n_rows, col 1..n_cols,
    size, ...). ``layout`` uses plate coordinates (from ``read_echo_picklist``),
    where the inner block starts at (``inner_row0``, ``inner_col0``). We try all 4
    orientations, map grid -> plate coords, join to layout, and pick the op whose
    blank/plated pattern best matches empty/present colonies.
    """
    best = None
    for op in ("identity", "rot180", "flip_v", "flip_h"):
        g = _orient(grid, n_rows, n_cols, op)
        g = g.assign(row=g["row"] + (inner_row0 - 1), col=g["col"] + (inner_col0 - 1))
        m = g.merge(layout, on=["row", "col"], how="inner")
        is_blank = m["strain"] == blank_name
        present = m["size"] > empty_thresh
        # agreement = blanks that are empty + plated wells that grew
        agree = ((is_blank & ~present) | (~is_blank & present)).mean()
        if best is None or agree > best[2]:
            best = (m, op, float(agree))
    assert best is not None, "no orientation evaluated"
    return best
