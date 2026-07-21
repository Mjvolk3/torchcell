# torchcell/sga/models.py
# [[torchcell.sga.models]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/models
"""Typed records + config for the SGAtools-style colony-fitness pipeline.

This is an adaptation of the Boone-lab SGAtools normalization & scoring stage
(Wagih et al. 2013, *Nucleic Acids Res.*; normalization math from Baryshnikova
et al. 2010, *Nat. Methods*) to a NON-SGA design: single-gene CRISPR-Cas
knockouts arrayed by acoustic (ECHO) dispensing, randomized over plate
position, with an on-plate wild-type reference (BY4741) and a no-cell control
(Blank_media). There is no query x array cross, so SGA's double-mutant score
S = C_ij - C_i * C_j does not apply; "score" is redefined as fitness relative
to the on-plate wild-type. See `[[torchcell.sga]]` and CLAUDE.md provenance
rules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizationConfig(BaseModel):
    """Knobs for the normalization stage. Defaults follow the SGA spirit but are
    adapted to a randomized single-KO layout (see class docstring in models).
    """

    # --- colony validity ---
    min_size: float = Field(
        1.0,
        description="Colony sizes <= this are treated as MISSING (gitter reports "
        "absent colonies as size 0).",
    )
    exclude_low_circularity: bool = Field(
        True,
        description="Drop gitter 'C' (low-circularity) colonies from the reference "
        "fit and from scoring. 'S' (spill/edge) is always dropped.",
    )

    # --- names of the special layout groups ---
    wt_name: str = Field(
        "BY4741", description="Sample name of the on-plate wild-type reference strain."
    )
    blank_name: str = Field(
        "Blank_media",
        description="Sample name of the no-cell control (background / contamination).",
    )

    # --- positional artifact correction (Baryshnikova 2010) ---
    row_col_correction: bool = Field(
        True,
        description="Divide out per-row and per-column median gradients "
        "(SGA row/column normalization).",
    )
    spatial_correction: bool = Field(
        True,
        description="Divide out a local neighbourhood-median surface (SGA spatial "
        "normalization). Legitimate here because the ECHO layout is randomized, so a "
        "local window is genotype-agnostic and estimates only positional bias.",
    )
    spatial_radius: int = Field(
        2,
        ge=1,
        description="Chebyshev radius of the spatial window; radius 2 = a 5x5 "
        "neighbourhood.",
    )
    spatial_min_neighbors: int = Field(
        4,
        ge=1,
        description="Minimum reference neighbours in the window; below this the "
        "colony falls back to the plate median (no local correction).",
    )

    # --- replicate filtering ---
    jackknife: bool = Field(
        True,
        description="Flag within-strain outlier replicates (SGA jackknife, adapted to "
        "LOGICAL replicate groups since ECHO replicates are scattered, not blocked). "
        "Also the instrument for catching CRISPR escapers.",
    )
    jackknife_z: float = Field(
        3.5,
        gt=0,
        description="MAD-based robust z threshold above which a replicate is flagged "
        "'JK' and excluded from that strain's score.",
    )

    # --- capping (SGA 'CP') ---
    cap_norm: float | None = Field(
        None,
        description="If set, normalized sizes above this are flagged 'CP' and capped. "
        "None = flag-free (recommended until growth settings are stable).",
    )


class StrainScore(BaseModel):
    """Per-strain aggregate: the analog of an SGAtools combined-file row."""

    strain: str
    n_total: int = Field(description="Replicate colonies placed for this strain.")
    n_used: int = Field(description="Replicates surviving all filters (used to score).")
    median_norm: float | None = Field(
        None, description="Median normalized colony size across used replicates."
    )
    mean_norm: float | None = None
    sd_norm: float | None = Field(
        None, description="SD of normalized size across used replicates (SGA 'SD')."
    )
    relative_fitness: float | None = Field(
        None,
        description="median_norm / wild-type median_norm on this plate. 1.0 = as fit "
        "as BY4741; 0.6 = 60% of WT; 1.3 = 30% fitter.",
    )
    fitness_sd: float | None = Field(
        None,
        description="SD of the per-replicate fitness ratios (= sd_norm / wt_median). "
        "Directly comparable to a published single-mutant-fitness standard deviation.",
    )
    log2_fitness: float | None = Field(
        None, description="log2(relative_fitness); 0 = WT-like, negative = sick."
    )
    pvalue: float | None = Field(
        None,
        description="Two-sided Mann-Whitney U of this strain's normalized replicates "
        "vs the wild-type replicates (SGA 'PV'; nonparametric to resist escapers).",
    )
    n_jackknife: int = Field(0, description="Replicates flagged 'JK' and excluded.")
    note: str = Field("", description="Free-text status (e.g. reference, control).")


class ScoreReport(BaseModel):
    """Top-level result of scoring one plate against its wild-type reference."""

    plate_id: str
    wt_name: str
    blank_name: str
    wt_median_norm: float | None = Field(
        None, description="Plate wild-type median normalized size (score denominator)."
    )
    blank_median_norm: float | None = Field(
        None,
        description="Median normalized size at no-cell control wells; should be ~0. "
        "Large values indicate contamination.",
    )
    n_colonies: int
    n_missing: int
    n_flagged: int
    strains: list[StrainScore]
