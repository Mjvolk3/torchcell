# torchcell/sga/assay.py
# [[torchcell.sga.assay]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/sga/assay
"""Assay-development metrics: given one plate with a designed condition variable
(here dispense VOLUME, 2.5 vs 5 nL), quantify which condition gives the best
assay so you can pick a plating setting.

"Best" for a fitness screen means: (1) colonies reliably grow and are measurable
(low missing rate, growth in a good size range), (2) replicates are reproducible
(low within-strain CV, tight wild-type), and (3) the assay separates phenotypes
(a wide window between wild-type and a sick strain -- the Z'-factor). These are
computed per condition from the normalized colony sizes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from torchcell.sga.models import NormalizationConfig


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        ~df["is_missing"]
        & ~df["is_flagged"]
        & ~df.get("is_jackknife", False)
        & ~df["is_blank"]
    ]


def shape_by_volume(
    df: pd.DataFrame, cfg: NormalizationConfig | None = None
) -> pd.DataFrame:
    """Colony-SHAPE metrics per condition (volume) -- for asking whether a larger
    dispense volume produces abnormal (spread / merged / irregular) colonies.

    Uses circularity (4*pi*area/perimeter^2, 1 = perfect disc). Gash-flagged
    colonies are excluded (mechanical tear, not a volume effect). NOTE: circularity
    is mildly SIZE-biased for small colonies (coarser perimeter), and larger volume
    grows larger colonies, so compare the distributions AND median size together
    (see colony_shape_by_volume scatter).
    """
    cfg = cfg or NormalizationConfig()
    plated = df[~df["is_blank"] & ~df["is_missing"]]
    nogash = plated[~plated["flags"].fillna("").str.contains("S")]
    rows = []
    for vol, sub in nogash.groupby("volume_nl"):
        c = sub["circularity"].dropna()
        rows.append(
            {
                "volume_nl": vol,
                "n": int(len(c)),
                "median_circularity": float(c.median()),
                "mean_circularity": float(c.mean()),
                "pct_circ_below_0.90": float((c < 0.90).mean()),
                "pct_circ_below_0.85": float((c < 0.85).mean()),
                "median_size_px": float(sub["size"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("volume_nl").reset_index(drop=True)


def volume_position_confound(df: pd.DataFrame) -> dict[str, Any]:
    """Detect whether the condition variable (volume) is confounded with plate
    position (row/col). If the volume groups do not overlap in column (or row)
    span, the volume effect cannot be separated from a spatial effect and any
    volume recommendation is unsafe. Returns a dict with the diagnosis.
    """
    out = {"confounded": False, "axis": None, "detail": ""}
    for axis in ("col", "row"):
        spans = {
            v: (int(sub[axis].min()), int(sub[axis].max()))
            for v, sub in df.groupby("volume_nl")
        }
        vols = sorted(spans)
        if len(vols) != 2:
            continue
        (a0, a1), (b0, b1) = spans[vols[0]], spans[vols[1]]
        overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
        if overlap == 0:
            out.update(
                confounded=True,
                axis=axis,
                detail=(
                    f"{vols[0]} nL occupies {axis} {a0}-{a1}, {vols[1]} nL occupies "
                    f"{axis} {b0}-{b1} (no overlap): volume is fully confounded with "
                    f"{axis} position. Randomize volume across position to compare it."
                ),
            )
            return out
    return out


def zfactor(a: np.ndarray, b: np.ndarray) -> float:
    """Z'-factor between two colony populations (assay separation window).

    Z' = 1 - 3*(sd_a + sd_b) / |mean_a - mean_b|. >0.5 excellent, 0-0.5 usable,
    <0 no separation. Uses the two most-separated groups to measure the window
    the assay actually offers.
    """
    sep = abs(np.mean(a) - np.mean(b))
    if sep == 0:
        return float("-inf")
    return float(1 - 3 * (np.std(a, ddof=1) + np.std(b, ddof=1)) / sep)


def volume_assay_metrics(
    df: pd.DataFrame, cfg: NormalizationConfig | None = None
) -> pd.DataFrame:
    """One row per condition (volume) with the assay-quality metrics."""
    cfg = cfg or NormalizationConfig()
    rows = []
    for vol, sub in df.groupby("volume_nl"):
        plated = sub[~sub["is_blank"]]
        used = _valid(sub)
        wt = used[used["strain"] == cfg.wt_name]["norm"].dropna()

        # per-strain median normalized sizes (used to find the separation window)
        strain_med = used.groupby("strain")["norm"].median()
        weakest = strain_med.drop(cfg.wt_name, errors="ignore").idxmin()
        weak_vals = used[used["strain"] == weakest]["norm"].dropna()

        # within-strain reproducibility (median CV across strains)
        cvs = used.groupby("strain")["norm"].agg(
            lambda x: x.std(ddof=1) / x.mean() if len(x) >= 2 and x.mean() else np.nan
        )

        rows.append(
            {
                "volume_nl": vol,
                "n_plated": int(len(plated)),
                "n_missing": int(plated["is_missing"].sum()),
                "missing_rate": float(plated["is_missing"].mean()),
                "n_flagged": int(plated["is_flagged"].sum()),
                "wt_median_raw": float(
                    plated[plated["strain"] == cfg.wt_name]["size"].median()
                ),
                "wt_cv": float(wt.std(ddof=1) / wt.mean()) if len(wt) >= 2 else np.nan,
                "median_within_strain_cv": float(cvs.median()),
                "weakest_strain": weakest,
                "dynamic_range": float(strain_med.max() / strain_med.min()),
                "zfactor_wt_vs_weakest": zfactor(wt.to_numpy(), weak_vals.to_numpy())
                if len(wt) >= 2 and len(weak_vals) >= 2
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("volume_nl").reset_index(drop=True)


def recommend_volume(metrics: pd.DataFrame) -> tuple[float, str]:
    """Pick the best volume for a fitness screen.

    RELIABILITY dominates: a plating volume is only useful if colonies grow and
    are reproducibly measured. So missing rate and wild-type CV carry the most
    weight (0.35 each), within-strain CV less (0.15), and the Z'-factor only
    counts when POSITIVE (0.15) -- when the assay separates no strains at either
    volume (both Z'<0, e.g. near-neutral knockouts), it must not decide the call.

    Returns (volume, human-readable rationale).
    """
    m = metrics.copy()

    def _lo(x: pd.Series) -> pd.Series:  # lower is better -> desirability in [0,1]
        return (
            (x.max() - x) / (x.max() - x.min()) if x.max() != x.min() else x * 0 + 0.5
        )

    def _hi(x: pd.Series) -> pd.Series:  # higher is better
        return (
            (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x * 0 + 0.5
        )

    z = m["zfactor_wt_vs_weakest"].clip(lower=0)  # negative Z' is not separation
    m["score"] = (
        0.35 * _lo(m["missing_rate"])
        + 0.35 * _lo(m["wt_cv"])
        + 0.15 * _lo(m["median_within_strain_cv"])
        + 0.15 * (_hi(z) if (z > 0).any() else z * 0)
    )
    best = m.loc[m["score"].idxmax()]
    zbest = best["zfactor_wt_vs_weakest"]
    sep = (
        "no strain separation at either volume (Z'<0)"
        if (m["zfactor_wt_vs_weakest"] < 0).all()
        else f"Z'(WT vs {best['weakest_strain']}) {zbest:.2f}"
    )
    rationale = (
        f"{best['volume_nl']} nL: missing {best['missing_rate']:.1%}, "
        f"WT CV {best['wt_cv']:.2f}, within-strain CV {best['median_within_strain_cv']:.2f}; {sep}"
    )
    return float(best["volume_nl"]), rationale
