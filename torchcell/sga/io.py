# torchcell/sga/io.py
# [[torchcell.sga.io]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/io
"""Readers that bring the two real inputs into one tidy per-colony table.

Input 1 - gitter DAT (image-quantification output, the SGAtools stage we skip
because we already produce it): whitespace/tab-delimited, comment lines start
with '#', columns ``row col size circularity flags`` (circularity + anything
after it are optional; SGAtools ignores them).

Input 2 - ECHO acoustic picklist (our layout, in place of an SGA array-layout
file): a CSV with ``Destination Well`` (e.g. ``B2``), ``Sample Name`` (strain),
and ``Transfer Volume`` (nL). The destination well letter+number is decoded to
gitter's 1-indexed ``row`` (A->1) and ``col``.
"""

from __future__ import annotations

import re

import pandas as pd

_WELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def well_to_rowcol(well: str) -> tuple[int, int]:
    """Decode a plate well label (``'B2'``, ``'AA13'``) to 1-indexed (row, col).

    Row letters are base-26 bijective (A=1 .. Z=26, AA=27), which covers 384
    (P=16) and 1536 (AF=32) plate formats without special-casing.
    """
    m = _WELL_RE.match(well.strip())
    if m is None:
        raise ValueError(f"unparseable well label: {well!r}")
    letters, digits = m.group(1).upper(), m.group(2)
    row = 0
    for ch in letters:
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row, int(digits)


def read_gitter_dat(path: str) -> pd.DataFrame:
    """Read a gitter ``.dat`` into a tidy frame: row, col, size, circularity, flags."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["row", "col", "size", "circularity", "flags"],
        engine="python",
    )
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["circularity"] = pd.to_numeric(df["circularity"], errors="coerce")
    df["flags"] = df["flags"].fillna("").astype(str).str.strip()
    return df


def read_echo_picklist(path: str) -> pd.DataFrame:
    """Read an ECHO picklist into a layout frame: row, col, strain, volume_nl, well."""
    raw = pd.read_csv(path)
    required = {"Destination Well", "Sample Name", "Transfer Volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"picklist {path} missing columns: {sorted(missing)}")
    rc = raw["Destination Well"].map(well_to_rowcol)
    layout = pd.DataFrame(
        {
            "row": [r for r, _ in rc],
            "col": [c for _, c in rc],
            "strain": raw["Sample Name"].astype(str).values,
            "volume_nl": pd.to_numeric(raw["Transfer Volume"], errors="coerce").values,
            "well": raw["Destination Well"].astype(str).values,
        }
    )
    return layout


def merge_layout(dat: pd.DataFrame, layout: pd.DataFrame | None) -> pd.DataFrame:
    """Attach strain + volume to each quantified colony by (row, col).

    With no layout, ``strain``/``volume_nl`` are None; downstream normalization
    still runs, but scoring is skipped (SGAtools behaves the same way when no
    array-layout file is supplied).
    """
    if layout is None:
        out = dat.copy()
        out["strain"] = None
        out["volume_nl"] = pd.NA
        out["well"] = pd.NA
        return out
    return dat.merge(layout, on=["row", "col"], how="left")
