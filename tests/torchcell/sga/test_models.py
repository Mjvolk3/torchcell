# tests/torchcell/sga/test_models.py
# [[tests.torchcell.sga.test_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/sga/test_models.py
"""The pydantic records of the SGA pipeline: defaults, bounds, required fields."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from torchcell.sga.models import NormalizationConfig, ScoreReport, StrainScore


def test_normalization_config_defaults() -> None:
    """The documented defaults: min_size 1.0, both corrections on, radius 2 with 4
    neighbors, jackknife at 3.5 MAD-z, no cap, BY4741 / Blank_media names.
    """
    cfg = NormalizationConfig()
    assert cfg.model_dump() == {
        "min_size": 1.0,
        "exclude_low_circularity": True,
        "wt_name": "BY4741",
        "blank_name": "Blank_media",
        "row_col_correction": True,
        "spatial_correction": True,
        "spatial_radius": 2,
        "spatial_min_neighbors": 4,
        "jackknife": True,
        "jackknife_z": 3.5,
        "cap_norm": None,
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"spatial_radius": 0},
        {"spatial_min_neighbors": 0},
        {"jackknife_z": 0.0},
        {"jackknife_z": -1.0},
    ],
)
def test_normalization_config_bounds(bad: dict[str, float]) -> None:
    """Radius and min-neighbors are >= 1, jackknife_z > 0; each violation is a
    ValidationError naming the field.
    """
    with pytest.raises(ValidationError) as ei:
        NormalizationConfig.model_validate(bad)
    assert ei.value.errors()[0]["loc"] == (next(iter(bad)),)


def test_strain_score_required_and_defaults() -> None:
    """strain, n_total, n_used are required; every statistic defaults to None,
    n_jackknife to 0 and note to ''.
    """
    s = StrainScore.model_validate({"strain": "geneA", "n_total": 4, "n_used": 3})
    assert s.model_dump() == {
        "strain": "geneA",
        "n_total": 4,
        "n_used": 3,
        "median_norm": None,
        "mean_norm": None,
        "sd_norm": None,
        "relative_fitness": None,
        "fitness_sd": None,
        "log2_fitness": None,
        "pvalue": None,
        "n_jackknife": 0,
        "note": "",
    }
    with pytest.raises(ValidationError) as ei:
        StrainScore.model_validate({"strain": "geneA", "n_total": 4})
    assert ei.value.errors()[0]["loc"] == ("n_used",)


def test_score_report_holds_strains() -> None:
    """The report keeps its strains in order and the optional medians default to None."""
    rep = ScoreReport.model_validate(
        {
            "plate_id": "P1",
            "wt_name": "BY4741",
            "blank_name": "Blank_media",
            "n_colonies": 2,
            "n_missing": 0,
            "n_flagged": 1,
            "strains": [
                {"strain": "b", "n_total": 1, "n_used": 1},
                {"strain": "a", "n_total": 1, "n_used": 0},
            ],
        }
    )
    assert [s.strain for s in rep.strains] == ["b", "a"]
    assert all(isinstance(s, StrainScore) for s in rep.strains)
    assert (rep.wt_median_norm, rep.blank_median_norm, rep.n_flagged) == (None, None, 1)
    without_strains: dict[str, Any] = {
        "plate_id": "P1",
        "wt_name": "w",
        "blank_name": "b",
        "n_colonies": 0,
        "n_missing": 0,
        "n_flagged": 0,
    }
    with pytest.raises(ValidationError) as ei:
        ScoreReport.model_validate(without_strains)
    assert ei.value.errors()[0]["loc"] == ("strains",)
