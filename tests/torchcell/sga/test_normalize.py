# tests/torchcell/sga/test_normalize.py
# [[tests.torchcell.sga.test_normalize]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_normalize.py
"""Positional normalization on hand-built plates.

* Multiplicative gradient: sizes R[r] * C[c] with R = (1, 2, 4), C = (10, 20, 30, 40) on
  a 3x4 plate. Plate median m = 40; row factors = R * median(C) / m = (0.625, 1.25,
  2.5); column factors = median(R) * C / m = (0.5, 1, 1.5, 2); every corrected size is
  R C / (R * 25 / 40 * 2 C / 40) = 1600 / 50 = 32, so the spatial step sees a flat
  plate and ``norm`` is exactly 1.0 everywhere.
* Left/right step: a 4x4 plate with 10 in columns 1-2 and 30 in columns 3-4, row/col
  correction off, spatial radius 1 with at least 3 neighbors. Excluding self, every
  window median is the colony's own side (10 or 30); the reference median is 20, so
  ``size_spatial`` = size * 20 / expected = 20 everywhere and ``norm`` = 1.0. With both
  corrections off ``norm`` = size / 20 = 0.5 or 1.5.
* Flags, cap and jackknife (both corrections off so ``norm`` = size / reference median):
  five BY4741 at 100, 110, 90, 100, 300 give reference median 100 and norms 1.0, 1.1,
  0.9, 1.0, 3.0; ``cap_norm=2.0`` caps the 3.0 (CP); the jackknife then sees
  (1.0, 1.1, 0.9, 1.0, 2.0): median 1.0, MAD 0.1, robust z of the capped value
  0.6745 * 1.0 / 0.1 = 6.7 > 3.5 -> JK, while 1.1 scores 0.67.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from torchcell.sga.models import NormalizationConfig
from torchcell.sga.normalize import (
    _flag_validity,
    _jackknife,
    _row_col_correct,
    _spatial_correct,
    _status_code,
    normalize_plate,
)

NEW_COLUMNS = [
    "is_missing",
    "is_flagged",
    "is_blank",
    "is_reference",
    "size_rc",
    "size_spatial",
    "norm",
    "is_capped",
    "is_jackknife",
    "status",
]


def _col(df: pd.DataFrame, name: str) -> NDArray[np.float64]:
    """A column as a float64 array (pandas-stubs types ``to_numpy`` too loosely)."""
    return np.asarray(df[name], dtype=np.float64)


def _multiplicative_plate() -> pd.DataFrame:
    rows = []
    for r, rf in enumerate((1, 2, 4), 1):
        for c, cf in enumerate((10, 20, 30, 40), 1):
            rows.append(
                {
                    "row": r,
                    "col": c,
                    "size": float(rf * cf),
                    "flags": "",
                    "strain": f"s{c}",
                }
            )
    return pd.DataFrame(rows)


def _step_plate() -> pd.DataFrame:
    rows = [
        {
            "row": r,
            "col": c,
            "size": 10.0 if c <= 2 else 30.0,
            "flags": "",
            "strain": "x",
        }
        for r in range(1, 5)
        for c in range(1, 5)
    ]
    return pd.DataFrame(rows)


def _flag_plate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"row": 1, "col": 1, "size": 100.0, "flags": "", "strain": "BY4741"},
            {"row": 1, "col": 2, "size": 110.0, "flags": "", "strain": "BY4741"},
            {"row": 1, "col": 3, "size": 90.0, "flags": "", "strain": "BY4741"},
            {"row": 1, "col": 4, "size": 100.0, "flags": "", "strain": "BY4741"},
            {"row": 2, "col": 1, "size": 300.0, "flags": "", "strain": "BY4741"},
            {"row": 2, "col": 2, "size": 0.0, "flags": "", "strain": "geneA"},
            {"row": 2, "col": 3, "size": 50.0, "flags": "C", "strain": "geneA"},
            {"row": 2, "col": 4, "size": 50.0, "flags": "S", "strain": "geneA"},
            {"row": 3, "col": 1, "size": 40.0, "flags": "", "strain": "Blank_media"},
            {"row": 3, "col": 2, "size": np.nan, "flags": None, "strain": "geneB"},
            {"row": 3, "col": 3, "size": 60.0, "flags": "M", "strain": "geneB"},
            {"row": 3, "col": 4, "size": 1.0, "flags": "S", "strain": "geneB"},
        ]
    )


def test_multiplicative_gradient_normalizes_to_one() -> None:
    """Row/col factors divide the separable gradient out exactly: ``size_rc`` is 32 on
    every well and ``norm`` is 1.0 (to 1e-12); the input frame is not mutated.
    """
    df = _multiplicative_plate()
    before = df.copy()
    out = normalize_plate(df)
    pd.testing.assert_frame_equal(df, before)
    assert list(out.columns) == ["row", "col", "size", "flags", "strain"] + NEW_COLUMNS
    assert_allclose(_col(out, "size_rc"), 32.0, atol=1e-12)
    assert_allclose(_col(out, "size_spatial"), 32.0, atol=1e-12)
    assert_allclose(_col(out, "norm"), 1.0, atol=1e-12)
    assert out["status"].tolist() == ["OK"] * 12
    assert out["is_reference"].all() and not out["is_jackknife"].any()


def test_row_col_factors_directly() -> None:
    """``_row_col_correct`` on the flagged multiplicative plate: corner (3, 4) is
    160 / (2.5 * 2) = 32; a colony in a row with no reference neighbors keeps factor 1.
    """
    df = _flag_validity(_multiplicative_plate(), NormalizationConfig())
    assert_allclose(np.asarray(_row_col_correct(df), dtype=float), 32.0, atol=1e-12)
    # drop row 3 from the reference set: its row factor falls back to 1.0. Reference
    # median m = median(10, 20, 30, 40, 20, 40, 60, 80) = 35; row factors 25/35 and
    # 50/35; column factors median(C, 2C) / 35 = 1.5 C / 35, so column 1 is 15/35.
    df.loc[df["row"] == 3, "is_reference"] = False
    rc = _row_col_correct(df)
    assert rc.iloc[0] == pytest.approx(10 / ((25 / 35) * (15 / 35)))
    assert rc.iloc[8] == pytest.approx(40 / (1.0 * (15 / 35)))


def test_step_plate_spatial_correction_flattens() -> None:
    """Spatial-only correction with radius 1: every window median equals the colony's
    own side, the reference median is 20, so ``size_spatial`` is 20 everywhere and
    ``norm`` 1.0; with both corrections off ``norm`` is size / 20.
    """
    cfg = NormalizationConfig(
        row_col_correction=False,
        spatial_radius=1,
        spatial_min_neighbors=3,
        jackknife=False,
    )
    out = normalize_plate(_step_plate(), cfg)
    assert_allclose(_col(out, "size_rc"), _col(out, "size"))
    assert_allclose(_col(out, "size_spatial"), 20.0, atol=1e-12)
    assert_allclose(_col(out, "norm"), 1.0, atol=1e-12)
    raw = normalize_plate(
        _step_plate(),
        NormalizationConfig(
            row_col_correction=False, spatial_correction=False, jackknife=False
        ),
    )
    assert raw["norm"].tolist() == [0.5, 0.5, 1.5, 1.5] * 4
    assert raw["is_jackknife"].tolist() == [False] * 16


def test_spatial_correct_falls_back_to_plate_median_with_few_neighbors() -> None:
    """A 2x2 plate has at most 3 neighbors per well; with ``spatial_min_neighbors=4``
    every expected value is the plate median 20 and the output is size * 20 / 20.
    """
    df = pd.DataFrame(
        {
            "row": [1, 1, 2, 2],
            "col": [1, 2, 1, 2],
            "size": [10.0, 30.0, 10.0, 30.0],
            "flags": "",
        }
    )
    df = _flag_validity(df, NormalizationConfig())
    out = _spatial_correct(
        df, "size", NormalizationConfig(spatial_radius=1, spatial_min_neighbors=4)
    )
    assert out.tolist() == [10.0, 30.0, 10.0, 30.0]
    out3 = _spatial_correct(
        df, "size", NormalizationConfig(spatial_radius=1, spatial_min_neighbors=3)
    )
    # neighbors of (1,1): 30, 10, 30 -> median 30 -> 10 * 20 / 30
    assert out3.iloc[0] == pytest.approx(20 / 3)
    assert out3.iloc[1] == pytest.approx(30 * 20 / 10)


def test_validity_flags_cap_and_jackknife() -> None:
    """Missing beats flag (size 1.0 with S is MISS only); C counts as a flag by default;
    S and M always; the blank is excluded from the reference but still normalized
    (0.4); the 300 outlier is capped to 2.0 (CP) and then jackknifed (JK).
    """
    cfg = NormalizationConfig(
        row_col_correction=False, spatial_correction=False, cap_norm=2.0
    )
    out = normalize_plate(_flag_plate(), cfg)
    assert out["is_missing"].tolist() == [False] * 5 + [
        True,
        False,
        False,
        False,
        True,
        False,
        True,
    ]
    assert out["is_flagged"].tolist() == [False] * 6 + [
        True,
        True,
        False,
        False,
        True,
        False,
    ]
    assert out["is_blank"].tolist() == [False] * 8 + [True, False, False, False]
    assert out["is_reference"].tolist() == [True] * 5 + [False] * 7
    norm = _col(out, "norm")
    assert_allclose(norm[:5], [1.0, 1.1, 0.9, 1.0, 2.0])
    assert np.isnan(norm[[5, 9, 11]]).all()
    assert_allclose(norm[[6, 7, 8, 10]], [0.5, 0.5, 0.4, 0.6])
    assert out["is_capped"].tolist() == [False] * 4 + [True] + [False] * 7
    assert out["is_jackknife"].tolist() == [False] * 4 + [True] + [False] * 7
    assert out["status"].tolist() == [
        "OK",
        "OK",
        "OK",
        "OK",
        "JK;CP",
        "MISS",
        "FLAG",
        "FLAG",
        "BLANK",
        "MISS",
        "FLAG",
        "MISS",
    ]


def test_low_circularity_flag_is_optional() -> None:
    """``exclude_low_circularity=False`` leaves the C colony in the reference set; S and
    M still flag.
    """
    cfg = NormalizationConfig(
        row_col_correction=False,
        spatial_correction=False,
        exclude_low_circularity=False,
    )
    out = normalize_plate(_flag_plate(), cfg)
    assert out["is_flagged"].tolist() == [False] * 7 + [True, False, False, True, False]
    assert bool(out["is_reference"].iloc[6]) is True


def test_no_strain_column_means_no_blanks_and_no_jackknife() -> None:
    """Without a layout nothing is blank and the jackknife has no groups."""
    cfg = NormalizationConfig(row_col_correction=False, spatial_correction=False)
    out = normalize_plate(_flag_plate().drop(columns=["strain"]), cfg)
    assert not out["is_blank"].any() and not out["is_jackknife"].any()
    assert out["status"].tolist()[8] == "OK"  # the would-be blank is just a colony


def test_jackknife_skips_groups_with_zero_mad() -> None:
    """A strain whose replicates are 1, 1, 1, 5 has MAD 0 and is left unflagged; a
    strain with MAD 0.1 flags its 3.0 (z = 0.6745 * 2 / 0.1 = 13.5) but not its 0.9.
    """
    df = pd.DataFrame(
        {
            "strain": ["a"] * 4 + ["b"] * 5,
            "norm": [1.0, 1.0, 1.0, 5.0, 0.9, 1.0, 1.1, 1.0, 3.0],
            "is_reference": [True] * 9,
        }
    )
    jk = _jackknife(df, NormalizationConfig())
    assert jk.tolist() == [False] * 8 + [True]
    assert (
        _jackknife(df.drop(columns=["strain"]), NormalizationConfig()).tolist()
        == [False] * 9
    )


def test_status_code_order() -> None:
    """Codes join in the fixed order MISS, FLAG, BLANK, JK, CP; none -> OK."""
    row = pd.Series(
        {
            "is_missing": True,
            "is_flagged": True,
            "is_blank": True,
            "is_jackknife": True,
            "is_capped": True,
        }
    )
    assert _status_code(row) == "MISS;FLAG;BLANK;JK;CP"
    assert _status_code(pd.Series({"is_missing": False, "is_flagged": False})) == "OK"
