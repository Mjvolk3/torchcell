# torchcell/sga/score.py
# [[torchcell.sga.score]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/score
"""Redefined "scoring": single-KO fitness relative to the on-plate wild-type.

SGAtools scores double-mutant epistasis (S = C_ij - C_i * C_j), which needs a
query x array cross. This design has single CRISPR knockouts and no query, so
that formula has no inputs. Here each strain is scored against the BY4741
wild-type replicated on the same plate:

    relative_fitness = median(norm | strain) / median(norm | wild-type)
    p-value          = Mann-Whitney U (strain norms vs wild-type norms)

The Mann-Whitney choice is deliberate: CRISPR wells can contain unedited
escapers that grow wild-type-sized, so the replicate distribution is not
Gaussian; a rank test resists those tails. Blank_media is reported separately
as a background / contamination indicator (its norm should be ~0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from torchcell.sga.models import NormalizationConfig, ScoreReport, StrainScore


def _used(df: pd.DataFrame) -> pd.DataFrame:
    """Colonies eligible to contribute to a strain's score."""
    return df[
        ~df["is_missing"]
        & ~df["is_flagged"]
        & ~df.get("is_jackknife", False)
        & ~df["is_blank"]
    ]


def score_plate(
    df: pd.DataFrame, cfg: NormalizationConfig | None = None, plate_id: str = "plate"
) -> ScoreReport:
    """Score a normalized plate table (output of ``normalize_plate``).

    Requires a ``strain`` column (a layout must have been merged); without it
    there are no replicate groups to score, mirroring SGAtools.
    """
    cfg = cfg or NormalizationConfig()
    if "strain" not in df.columns or df["strain"].isna().all():
        raise ValueError(
            "score_plate needs a 'strain' column from a merged layout; "
            "normalize-only (no scoring) is the correct path without a layout."
        )

    used = _used(df)
    wt = used[used["strain"] == cfg.wt_name]["norm"].dropna()
    wt_median = float(wt.median()) if len(wt) else None

    blank = df[df["is_blank"]]["norm"].dropna()
    blank_median = float(blank.median()) if len(blank) else None

    strains: list[StrainScore] = []
    for strain, g in df.groupby("strain", sort=True):
        if strain == cfg.blank_name:
            strains.append(
                StrainScore(
                    strain=strain,
                    n_total=len(g),
                    n_used=0,
                    median_norm=blank_median,
                    note="no-cell control (background); norm should be ~0",
                )
            )
            continue

        gu = used[used["strain"] == strain]
        vals = gu["norm"].dropna()
        n_used = int(len(vals))
        median_norm = float(vals.median()) if n_used else None
        rel = (
            median_norm / wt_median
            if (median_norm is not None and wt_median not in (None, 0))
            else None
        )
        sd_norm = float(vals.std(ddof=1)) if n_used >= 2 else None
        fit_sd = (
            sd_norm / wt_median
            if (sd_norm is not None and wt_median not in (None, 0))
            else None
        )
        pval = None
        if strain != cfg.wt_name and n_used >= 3 and len(wt) >= 3:
            pval = float(mannwhitneyu(vals, wt, alternative="two-sided").pvalue)

        strains.append(
            StrainScore(
                strain=strain,
                n_total=int((df["strain"] == strain).sum()),
                n_used=n_used,
                median_norm=median_norm,
                mean_norm=float(vals.mean()) if n_used else None,
                sd_norm=sd_norm,
                relative_fitness=rel,
                fitness_sd=fit_sd,
                log2_fitness=(
                    float(np.log2(rel)) if (rel is not None and rel > 0) else None
                ),
                pvalue=pval,
                n_jackknife=int(
                    ((df["strain"] == strain) & df.get("is_jackknife", False)).sum()
                ),
                note="wild-type reference" if strain == cfg.wt_name else "",
            )
        )

    return ScoreReport(
        plate_id=plate_id,
        wt_name=cfg.wt_name,
        blank_name=cfg.blank_name,
        wt_median_norm=wt_median,
        blank_median_norm=blank_median,
        n_colonies=len(df),
        n_missing=int(df["is_missing"].sum()),
        n_flagged=int(df["is_flagged"].sum()),
        strains=strains,
    )


def score_table(report: ScoreReport) -> pd.DataFrame:
    """Flatten a ScoreReport into a per-strain DataFrame for CSV output."""
    return pd.DataFrame([s.model_dump() for s in report.strains])
