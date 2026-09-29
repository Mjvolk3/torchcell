# tests/torchcell/sga/test_assay.py
# [[tests.torchcell.sga.test_assay]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_assay.py
"""Assay-development metrics per dispense volume.

Two-volume normalized table:

* 2.5 nL: BY4741 norms 0.9, 1.0, 1.1 (sizes 90, 100, 110); geneA 0.4, 0.5, 0.6 plus one
  missing colony; one blank. n_plated 7, missing rate 1/7, WT median raw size 100,
  WT CV 0.1 / 1.0 = 0.10, geneA CV 0.1 / 0.5 = 0.20 -> median within-strain CV 0.15,
  dynamic range 1.0 / 0.5 = 2, Z' = 1 - 3 (0.1 + 0.1) / 0.5 = -0.2.
* 5.0 nL: BY4741 0.8, 1.0, 1.2 (CV 0.2); geneA 0.2, 0.5, 0.8 (SD 0.3, CV 0.6); one
  blank; nothing missing. Median CV 0.4, dynamic range 2, Z' = 1 - 3 (0.2 + 0.3) / 0.5
  = -2.0.

Recommendation: both Z' are negative so that term is zero; desirabilities (lower is
better, scaled to [0, 1] across the two rows) give 2.5 nL 0 + 0.35 + 0.15 = 0.50 and
5.0 nL 0.35 + 0 + 0 = 0.35, so 2.5 nL wins with the rationale
"2.5 nL: missing 14.3%, WT CV 0.10, within-strain CV 0.15; no strain separation at
either volume (Z'<0)".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from torchcell.sga.assay import (
    recommend_volume,
    shape_by_volume,
    volume_assay_metrics,
    volume_position_confound,
    zfactor,
)


def _rows(
    vol: float, strain: str, norms: list[float], col0: int, missing: int = 0
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, n in enumerate(norms):
        out.append(
            {
                "row": 1 if vol == 2.5 else 2,
                "col": col0 + i,
                "volume_nl": vol,
                "strain": strain,
                "size": n * 100,
                "norm": n,
                "circularity": 1.0,
                "flags": "",
                "is_missing": False,
                "is_flagged": False,
                "is_blank": strain == "Blank_media",
                "is_jackknife": False,
            }
        )
    for j in range(missing):
        out.append(
            {
                "row": 1 if vol == 2.5 else 2,
                "col": col0 + len(norms) + j,
                "volume_nl": vol,
                "strain": strain,
                "size": 0.0,
                "norm": np.nan,
                "circularity": np.nan,
                "flags": "",
                "is_missing": True,
                "is_flagged": False,
                "is_blank": False,
                "is_jackknife": False,
            }
        )
    return out


def _table() -> pd.DataFrame:
    rows = (
        _rows(2.5, "BY4741", [0.9, 1.0, 1.1], 1)
        + _rows(2.5, "geneA", [0.4, 0.5, 0.6], 4, missing=1)
        + _rows(2.5, "Blank_media", [0.02], 8)
        + _rows(5.0, "BY4741", [0.8, 1.0, 1.2], 1)
        + _rows(5.0, "geneA", [0.2, 0.5, 0.8], 4)
        + _rows(5.0, "Blank_media", [0.03], 7)
    )
    return pd.DataFrame(rows)


def test_zfactor_closed_forms() -> None:
    """Zero spread -> 1.0; SDs 0.1 and 0.1 across a 0.5 gap -> 1 - 0.6 / 0.5 = -0.2;
    equal means -> -inf (no window at all).
    """
    assert zfactor(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])) == 1.0
    assert zfactor(
        np.array([0.9, 1.0, 1.1]), np.array([0.4, 0.5, 0.6])
    ) == pytest.approx(-0.2)
    assert zfactor(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == float("-inf")


def test_volume_position_confound_detects_disjoint_spans() -> None:
    """2.5 nL in columns 1-3 and 5 nL in 4-6 (rows shared) is confounded on ``col`` with
    the exact diagnosis; overlapping spans (or a single volume) are not.
    """
    df = pd.DataFrame(
        {
            "row": [1, 2, 1, 2, 1, 2],
            "col": [1, 2, 3, 4, 5, 6],
            "volume_nl": [2.5, 2.5, 2.5, 5, 5, 5],
        }
    )
    out = volume_position_confound(df)
    assert out == {
        "confounded": True,
        "axis": "col",
        "detail": (
            "2.5 nL occupies col 1-3, 5.0 nL occupies col 4-6 (no overlap): volume is fully "
            "confounded with col position. Randomize volume across position to compare it."
        ),
    }
    mixed = df.assign(col=[1, 4, 2, 5, 3, 6])
    assert volume_position_confound(mixed) == {
        "confounded": False,
        "axis": None,
        "detail": "",
    }
    assert volume_position_confound(df.assign(volume_nl=2.5)) == {
        "confounded": False,
        "axis": None,
        "detail": "",
    }


def test_volume_position_confound_checks_rows_second() -> None:
    """Columns overlap but rows do not: the row axis is reported."""
    df = pd.DataFrame(
        {"row": [1, 1, 2, 2], "col": [1, 2, 1, 2], "volume_nl": [2.5, 2.5, 5.0, 5.0]}
    )
    out = volume_position_confound(df)
    assert (out["confounded"], out["axis"]) == (True, "row")
    assert str(out["detail"]).startswith(
        "2.5 nL occupies row 1-1, 5.0 nL occupies row 2-2 (no overlap)"
    )


def test_shape_by_volume_excludes_gash_missing_and_blank() -> None:
    """2.5 nL: circularities 0.9 and 0.8 (median 0.85, 50% below 0.90 and below 0.85,
    median size 15); 5.0 nL: the S-flagged 0.7 is dropped, leaving 0.6 and 0.95 (median
    0.775, median size 45). The missing colony and the blank never enter.
    """
    df = pd.DataFrame(
        {
            "volume_nl": [2.5, 2.5, 5.0, 5.0, 5.0, 2.5, 5.0],
            "circularity": [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.99],
            "size": [10, 20, 30, 40, 50, 60, 70],
            "is_blank": [False] * 6 + [True],
            "is_missing": [False] * 5 + [True, False],
            "flags": ["", "", "S", "", "", "", ""],
        }
    )
    out = shape_by_volume(df)
    assert list(out.columns) == [
        "volume_nl",
        "n",
        "median_circularity",
        "mean_circularity",
        "pct_circ_below_0.90",
        "pct_circ_below_0.85",
        "median_size_px",
    ]
    assert out["volume_nl"].tolist() == [2.5, 5.0]
    assert out["n"].tolist() == [2, 2]
    assert out["median_circularity"].tolist() == pytest.approx([0.85, 0.775])
    assert out["mean_circularity"].tolist() == pytest.approx([0.85, 0.775])
    assert out["pct_circ_below_0.90"].tolist() == [0.5, 0.5]
    assert out["pct_circ_below_0.85"].tolist() == [0.5, 0.5]
    assert out["median_size_px"].tolist() == [15.0, 45.0]


def test_volume_assay_metrics_values() -> None:
    """The per-volume metrics derived in the module docstring."""
    m = volume_assay_metrics(_table())
    assert m["volume_nl"].tolist() == [2.5, 5.0]
    assert m["n_plated"].tolist() == [7, 6]
    assert m["n_missing"].tolist() == [1, 0]
    assert m["missing_rate"].tolist() == pytest.approx([1 / 7, 0.0])
    assert m["n_flagged"].tolist() == [0, 0]
    assert m["wt_median_raw"].tolist() == [100.0, 100.0]
    assert m["wt_cv"].tolist() == pytest.approx([0.1, 0.2])
    assert m["median_within_strain_cv"].tolist() == pytest.approx([0.15, 0.4])
    assert m["weakest_strain"].tolist() == ["geneA", "geneA"]
    assert m["dynamic_range"].tolist() == pytest.approx([2.0, 2.0])
    assert m["zfactor_wt_vs_weakest"].tolist() == pytest.approx([-0.2, -2.0])


def test_volume_assay_metrics_nan_when_too_few_replicates() -> None:
    """A WT with a single replicate has no CV and no Z'; the weakest strain is still
    named.
    """
    df = _table()
    df = df[~((df["volume_nl"] == 5.0) & (df["strain"] == "BY4741") & (df["col"] > 1))]
    m = volume_assay_metrics(df)
    row = m[m["volume_nl"] == 5.0].iloc[0].to_dict()
    assert np.isnan(row["wt_cv"]) and np.isnan(row["zfactor_wt_vs_weakest"])
    assert row["weakest_strain"] == "geneA"
    assert row["n_plated"] == 4


def test_recommend_volume_reliability_dominates() -> None:
    """With both Z' negative the separation term is zero and 2.5 nL wins 0.50 to 0.35."""
    vol, why = recommend_volume(volume_assay_metrics(_table()))
    assert vol == 2.5
    assert (
        why
        == "2.5 nL: missing 14.3%, WT CV 0.10, within-strain CV 0.15; no strain separation at either volume (Z'<0)"
    )


def test_recommend_volume_positive_z_breaks_a_tie() -> None:
    """Hand-built metrics identical except Z': the tied desirabilities (0.5 each) give
    0.35 * 0.5 + 0.35 * 0.5 + 0.15 * 0.5 = 0.425 to both, and the clipped Z' term adds
    0.15 to the row with Z' 0.6 only; the rationale then quotes the window.
    """
    m = pd.DataFrame(
        {
            "volume_nl": [2.5, 5.0],
            "missing_rate": [0.1, 0.1],
            "wt_cv": [0.1, 0.1],
            "median_within_strain_cv": [0.2, 0.2],
            "weakest_strain": ["geneA", "geneB"],
            "zfactor_wt_vs_weakest": [-0.5, 0.6],
        }
    )
    vol, why = recommend_volume(m)
    assert vol == 5.0
    assert (
        why
        == "5.0 nL: missing 10.0%, WT CV 0.10, within-strain CV 0.20; Z'(WT vs geneB) 0.60"
    )
    assert list(m.columns)[-1] == "zfactor_wt_vs_weakest"  # the input is not mutated
