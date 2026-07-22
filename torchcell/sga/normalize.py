# torchcell/sga/normalize.py
# [[torchcell.sga.normalize]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/normalize
"""Positional-artifact correction, adapted from SGAtools / Baryshnikova 2010.

Colony size on an agar array reflects genotype fitness PLUS position: colonies
near edges get more nutrient and less competition, and there are row/column and
local gradients. Normalization removes the positional part so the residual is a
fitness proxy where 1.0 = the plate-average strain.

Pipeline (each step is optional via ``NormalizationConfig``):
  1. Validity flags   - missing (size<=min_size), gitter 'S'/'C' flags, blanks.
  2. Reference set    - non-missing, non-flagged, non-blank colonies define ALL
                        correction factors (blanks are no-cell, so excluded).
  3. Row/col gradient - divide by per-row and per-column median factors.
  4. Spatial surface  - divide by a local neighbourhood-median surface. Valid
                        here ONLY because the ECHO layout is randomized, so a
                        local window mixes genotypes and estimates position.
  5. Plate scaling    - divide by the corrected reference median -> ``norm``.
  6. Jackknife        - flag within-strain replicate outliers ('JK'); this is
                        also what surfaces CRISPR escapers.

NOT ported from SGA: linkage correction (no query -> no query-array linkage
artifact).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from torchcell.sga.models import NormalizationConfig


def _reference_mask(df: pd.DataFrame, cfg: NormalizationConfig) -> pd.Series:
    return ~df["is_missing"] & ~df["is_flagged"] & ~df["is_blank"]


def _flag_validity(df: pd.DataFrame, cfg: NormalizationConfig) -> pd.DataFrame:
    df = df.copy()
    df["is_missing"] = df["size"].isna() | (df["size"] <= cfg.min_size)
    flags = df["flags"].fillna("").astype(str)
    # 'S' spill/gash, 'M' multiple-colony, and 'N' (neighbour a duplicate crowds)
    # cells are always rejected: in each the measured size is not a faithful
    # single-colony fitness readout.
    is_flagged = (
        flags.str.contains("S") | flags.str.contains("M") | flags.str.contains("N")
    )
    if cfg.exclude_low_circularity:
        is_flagged = is_flagged | flags.str.contains("C")
    df["is_flagged"] = is_flagged & ~df["is_missing"]
    if "strain" in df.columns:
        df["is_blank"] = df["strain"].astype("string").eq(cfg.blank_name).fillna(False)
    else:
        df["is_blank"] = False
    df["is_reference"] = _reference_mask(df, cfg)
    return df


def _row_col_correct(df: pd.DataFrame) -> pd.Series:
    """Divide out per-row and per-column median gradients (SGA row/col step)."""
    ref = df[df["is_reference"]]
    m = ref["size"].median()
    row_fac = (ref.groupby("row")["size"].median() / m).to_dict()
    col_fac = (ref.groupby("col")["size"].median() / m).to_dict()
    rf = df["row"].map(row_fac).fillna(1.0)
    cf = df["col"].map(col_fac).fillna(1.0)
    denom = (rf * cf).replace(0, np.nan)
    return df["size"] / denom


def _spatial_correct(
    df: pd.DataFrame, value_col: str, cfg: NormalizationConfig
) -> pd.Series:
    """Divide out a local neighbourhood-median surface (SGA spatial step).

    For each colony, the "expected" size is the median of REFERENCE colonies in
    a (2*radius+1) square around it (excluding itself). Positions with too few
    reference neighbours fall back to the plate median (no local correction).
    """
    n_rows, n_cols = int(df["row"].max()), int(df["col"].max())
    grid = np.full((n_rows + 1, n_cols + 1), np.nan)
    ref = df[df["is_reference"]]
    grid[ref["row"].to_numpy(), ref["col"].to_numpy()] = ref[value_col].to_numpy()

    plate_med = float(np.nanmedian(grid))
    w = cfg.spatial_radius
    expected = np.empty(len(df))
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    for i in range(len(df)):
        r, c = rows[i], cols[i]
        r0, r1 = max(1, r - w), min(n_rows, r + w)
        c0, c1 = max(1, c - w), min(n_cols, c + w)
        window = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        # exclude self
        window[r - r0, c - c0] = np.nan
        vals = window[~np.isnan(window)]
        expected[i] = (
            np.median(vals) if vals.size >= cfg.spatial_min_neighbors else plate_med
        )
    exp = pd.Series(expected, index=df.index).replace(0, np.nan)
    ref_med = float(np.nanmedian(df.loc[df["is_reference"], value_col]))
    return df[value_col] * ref_med / exp


def _jackknife(df: pd.DataFrame, cfg: NormalizationConfig) -> pd.Series:
    """Flag within-strain replicate outliers by robust (MAD) z-score."""
    jk = pd.Series(False, index=df.index)
    if "strain" not in df.columns:
        return jk
    eligible = df["is_reference"]
    for strain, idx in df[eligible].groupby("strain").groups.items():
        vals = df.loc[idx, "norm"]
        med = vals.median()
        mad = (vals - med).abs().median()
        if mad == 0 or np.isnan(mad):
            continue
        z = 0.6745 * (vals - med) / mad
        jk.loc[idx] = z.abs() > cfg.jackknife_z
    return jk


def normalize_plate(
    df: pd.DataFrame, cfg: NormalizationConfig | None = None
) -> pd.DataFrame:
    """Run the full normalization on one plate's merged colony table.

    Returns a copy with added columns: is_missing, is_flagged, is_blank,
    is_reference, size_rc, size_spatial, norm, is_jackknife, status.
    """
    cfg = cfg or NormalizationConfig()
    df = _flag_validity(df, cfg)

    df["size_rc"] = _row_col_correct(df) if cfg.row_col_correction else df["size"]
    df["size_spatial"] = (
        _spatial_correct(df, "size_rc", cfg)
        if cfg.spatial_correction
        else df["size_rc"]
    )

    ref_med = float(np.nanmedian(df.loc[df["is_reference"], "size_spatial"]))
    df["norm"] = df["size_spatial"] / ref_med
    df.loc[df["is_missing"], "norm"] = np.nan

    df["is_capped"] = False
    if cfg.cap_norm is not None:
        over = df["norm"] > cfg.cap_norm
        df.loc[over, "is_capped"] = True
        df.loc[over, "norm"] = cfg.cap_norm

    df["is_jackknife"] = _jackknife(df, cfg) if cfg.jackknife else False

    df["status"] = df.apply(_status_code, axis=1)
    return df


def _status_code(row: pd.Series) -> str:
    codes = []
    if row["is_missing"]:
        codes.append("MISS")
    if row["is_flagged"]:
        codes.append("FLAG")
    if row.get("is_blank", False):
        codes.append("BLANK")
    if row.get("is_jackknife", False):
        codes.append("JK")
    if row.get("is_capped", False):
        codes.append("CP")
    return ";".join(codes) if codes else "OK"
