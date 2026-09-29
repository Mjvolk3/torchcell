# tests/torchcell/sga/test_score.py
# [[tests.torchcell.sga.test_score]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_score.py
"""Relative-fitness scoring against the on-plate wild type.

Hand-built normalized table (13 colonies):

* BY4741 norms 0.8, 0.9, 1.1, 1.2 (no ties): median 1.0, mean 1.0, sample SD
  sqrt((0.04 + 0.01 + 0.01 + 0.04) / 3) = sqrt(1/30) = 0.182574;
* geneA norms 0.4, 0.5, 0.6 plus one missing colony (NaN): n_total 4, n_used 3, median
  0.5, SD 0.1, relative fitness 0.5, log2 -1, fitness SD 0.1 / 1.0; every geneA value is
  below every WT value, so the exact two-sided Mann-Whitney p is 2 / C(7, 3) = 2 / 35;
* geneB 1.5, 1.7: n = 2 (< 3) so no p-value; SD 0.1 * sqrt(2); log2(1.6) = 0.678072;
* geneC 2.0 alone: no SD, no p, relative fitness 2.0, log2 1.0;
* Blank_media 0.05, 0.15: reported as the background control with median 0.10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from torchcell.sga.models import NormalizationConfig, StrainScore
from torchcell.sga.score import _used, score_plate, score_table


def _table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strain": ["BY4741"] * 4
            + ["geneA"] * 4
            + ["geneB"] * 2
            + ["geneC"]
            + ["Blank_media"] * 2,
            "norm": [
                0.8,
                0.9,
                1.1,
                1.2,
                0.4,
                0.5,
                0.6,
                np.nan,
                1.5,
                1.7,
                2.0,
                0.05,
                0.15,
            ],
            "is_missing": [False] * 7 + [True] + [False] * 5,
            "is_flagged": [False] * 13,
            "is_blank": [False] * 11 + [True] * 2,
            "is_jackknife": [False] * 13,
        }
    )


def test_score_plate_report() -> None:
    """Plate-level fields: WT median 1.0, blank median 0.1, 13 colonies, 1 missing,
    0 flagged; strains sorted by name (uppercase first) with the derived statistics.
    """
    rep = score_plate(_table(), plate_id="P1")
    assert (rep.plate_id, rep.wt_name, rep.blank_name) == (
        "P1",
        "BY4741",
        "Blank_media",
    )
    assert (rep.wt_median_norm, rep.blank_median_norm) == (1.0, 0.1)
    assert (rep.n_colonies, rep.n_missing, rep.n_flagged) == (13, 1, 0)
    assert [s.strain for s in rep.strains] == [
        "BY4741",
        "Blank_media",
        "geneA",
        "geneB",
        "geneC",
    ]
    by = {s.strain: s for s in rep.strains}

    wt = by["BY4741"]
    assert (wt.n_total, wt.n_used, wt.median_norm, wt.mean_norm) == (4, 4, 1.0, 1.0)
    assert wt.sd_norm == pytest.approx(np.sqrt(1 / 30))
    assert (wt.relative_fitness, wt.log2_fitness, wt.pvalue) == (1.0, 0.0, None)
    assert wt.note == "wild-type reference"

    a = by["geneA"]
    assert (a.n_total, a.n_used, a.median_norm) == (4, 3, 0.5)
    assert a.mean_norm == pytest.approx(0.5)
    assert a.sd_norm == pytest.approx(0.1)
    assert a.relative_fitness == 0.5 and a.log2_fitness == -1.0
    assert a.fitness_sd == pytest.approx(0.1)
    assert a.pvalue == pytest.approx(2 / 35, rel=1e-12)
    assert (a.n_jackknife, a.note) == (0, "")

    b = by["geneB"]
    assert (b.n_used, b.pvalue) == (2, None)
    assert b.median_norm == pytest.approx(1.6)
    assert b.sd_norm == pytest.approx(0.1 * np.sqrt(2))
    assert b.log2_fitness == pytest.approx(np.log2(1.6))

    c = by["geneC"]
    assert (
        c.n_used,
        c.sd_norm,
        c.fitness_sd,
        c.pvalue,
        c.relative_fitness,
        c.log2_fitness,
    ) == (1, None, None, None, 2.0, 1.0)

    blank = by["Blank_media"]
    assert (blank.n_total, blank.n_used, blank.median_norm, blank.relative_fitness) == (
        2,
        0,
        0.1,
        None,
    )
    assert blank.note == "no-cell control (background); norm should be ~0"


def test_score_plate_counts_jackknifed_replicates() -> None:
    """A JK replicate leaves ``n_used`` and the median but is counted in n_jackknife;
    the WT median denominator excludes it too.
    """
    df = _table()
    df.loc[3, "is_jackknife"] = True  # WT 1.2 -> WT becomes 0.8, 0.9, 1.1 (median 0.9)
    df.loc[6, "is_jackknife"] = True  # geneA 0.6 -> 0.4, 0.5 (n_used 2, no p)
    rep = score_plate(df)
    by = {s.strain: s for s in rep.strains}
    assert rep.wt_median_norm == 0.9
    assert (by["BY4741"].n_used, by["BY4741"].n_jackknife) == (3, 1)
    assert (by["geneA"].n_used, by["geneA"].n_jackknife, by["geneA"].pvalue) == (
        2,
        1,
        None,
    )
    assert by["geneA"].relative_fitness == pytest.approx(0.45 / 0.9)


def test_score_plate_custom_names_and_zero_wt() -> None:
    """Renamed WT/blank flow through the config; a WT median of 0 disables the ratios
    (relative_fitness and fitness_sd None) instead of dividing by zero.
    """
    df = _table().replace({"strain": {"BY4741": "WT", "Blank_media": "empty"}})
    cfg = NormalizationConfig(wt_name="WT", blank_name="empty")
    rep = score_plate(df, cfg)
    assert (rep.wt_name, rep.blank_name, rep.wt_median_norm) == ("WT", "empty", 1.0)
    df.loc[df["strain"] == "WT", "norm"] = 0.0
    rep0 = score_plate(df, cfg)
    a = next(s for s in rep0.strains if s.strain == "geneA")
    assert rep0.wt_median_norm == 0.0
    assert (a.relative_fitness, a.fitness_sd, a.log2_fitness) == (None, None, None)
    assert a.median_norm == 0.5


def test_score_plate_requires_a_layout() -> None:
    """No ``strain`` column, or an all-null one, is the normalize-only path."""
    with pytest.raises(ValueError, match="score_plate needs a 'strain' column"):
        score_plate(_table().drop(columns=["strain"]))
    df = _table()
    df["strain"] = None
    with pytest.raises(ValueError, match="normalize-only"):
        score_plate(df)


def test_used_excludes_missing_flagged_blank_and_jackknife() -> None:
    """Each exclusion removes exactly its row; the frame without ``is_jackknife`` is
    handled by the ``df.get`` default.
    """
    df = _table()
    df.loc[0, "is_flagged"] = True
    df.loc[1, "is_jackknife"] = True
    used = _used(df)
    assert used.index.tolist() == [2, 3, 4, 5, 6, 8, 9, 10]
    assert _used(df.drop(columns=["is_jackknife"])).index.tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        9,
        10,
    ]


def test_score_table_flattens_the_report() -> None:
    """One row per strain in report order, columns = StrainScore fields in order."""
    tbl = score_table(score_plate(_table(), plate_id="P1"))
    assert list(tbl.columns) == list(StrainScore.model_fields)
    assert tbl["strain"].tolist() == [
        "BY4741",
        "Blank_media",
        "geneA",
        "geneB",
        "geneC",
    ]
    assert tbl.set_index("strain").loc["geneA", "relative_fitness"] == 0.5
    assert tbl.set_index("strain").loc["BY4741", "note"] == "wild-type reference"
