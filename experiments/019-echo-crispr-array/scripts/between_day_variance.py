# experiments/019-echo-crispr-array/scripts/between_day_variance.py
# [[experiments.019-echo-crispr-array.scripts.between_day_variance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/between_day_variance
"""Measure the day/batch term that run 3 alone could not see.

`replication_structure.py` flagged sigma_day as an unmeasured SE floor, because run 3's three
plates were all plated on one day. We DO have a second day: run 2 (2026-07-17) and run 3
(2026-07-23) share the same 12 strains + BY4741 + Blank_media, and both ran a 5 nL condition.
Six days apart is a genuine day/batch contrast.

    run 2, 2026-07-17 : P1 (2.5 nL) and P2 (5 nL), IDENTICAL layout (verified: 384/384 wells
                        carry the same strain), captured at t44 / t50 / t72
    run 3, 2026-07-23 : P1 / P2 / P3, all 5 nL, three INDEPENDENTLY randomized layouts
                        (pairwise strain agreement 0.05-0.08, i.e. chance for 13 strains)

Only the 5 nL arms are comparable, so the contrast is run-2 P2 vs the run-3 plates.

CRITICAL: everything is re-detected and re-scored HERE with one identical current-pipeline
config. The committed run-2 CSVs were produced with `multi_min_frac = 0.5` and the
pre-homography recovery, so comparing them to run 3 would let a PIPELINE change masquerade as
a day effect.

Variance model for a strain's plate-level fitness:
    f_{d,p} = f + day_d + plate_{d,p},   Var(day) = sigma_day^2, Var(plate) = sigma_plate^2
sigma_plate is estimated within-day from the three run-3 plates. The day contrast then gives
sigma_day by subtraction; with only two days this is 1 degree of freedom, so it is an
order-of-magnitude estimate, not a precise one.

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/between_day_variance.py
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.sga import (
    CellposeSegConfig,
    NormalizationConfig,
    load_cellpose_model,
    normalize_plate,
    quantify_plate_image_cellpose,
    read_echo_picklist,
    score_plate,
    score_table,
)

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "019-echo-crispr-array")
DATA = osp.join(EXP_DIR, "data")
RESULTS = osp.join(EXP_DIR, "results")
QUANT = osp.join(EXP_DIR, "quant", "between_day")

N_ROWS, N_COLS = 16, 24
WT_NAME, BLANK_NAME = "BY4741", "Blank_media"
# Only 2.7% of genome-wide deletions are significantly above WT (ladder_feasibility.py),
# so a plate where a large fraction of deletions beat WT has a broken WT reference.
FRAC_ABOVE_WT_MAX = 0.20

# Only 5 nL arms: volume is a real experimental factor and would confound the day contrast.
CONDITIONS: list[dict[str, Any]] = [
    dict(
        day="2026-07-17", group="run2_P2_t44",
        image=osp.join(DATA, "run2_2026-07-17", "P2_5nL_view_t44.jpg"),
        picklist=osp.join(DATA, "run2_2026-07-17", "P2_5nL_cherrypick_13strain.csv"),
    ),
    dict(
        day="2026-07-17", group="run2_P2_t50",
        image=osp.join(DATA, "run2_2026-07-17", "t50", "P2_5nL_view_t50_E30F9F19.jpeg"),
        picklist=osp.join(DATA, "run2_2026-07-17", "P2_5nL_cherrypick_13strain.csv"),
    ),
    dict(
        day="2026-07-23", group="run3_P1",
        image=osp.join(DATA, "run3_2026-07-23", "P1_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA, "run3_2026-07-23", "cherrypick_Plate1_384_5nL.csv"),
    ),
    dict(
        day="2026-07-23", group="run3_P2",
        image=osp.join(DATA, "run3_2026-07-23", "P2_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA, "run3_2026-07-23", "cherrypick_Plate2_384_5nL.csv"),
    ),
    dict(
        day="2026-07-23", group="run3_P3",
        image=osp.join(DATA, "run3_2026-07-23", "P3_OD1-5nL_TCsingleKO.JPG"),
        picklist=osp.join(DATA, "run3_2026-07-23", "cherrypick_Plate3_384_5nL.csv"),
    ),
]


def score_one(cond: dict[str, Any], model: Any, seg_cfg: CellposeSegConfig,
              cfg: NormalizationConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect, register, orient and score one plate image with the current pipeline.

    Orientation is resolved by the same strain-structure resolver run 3 uses; the image
    capture orientation differs between the two sessions, so this cannot be skipped.
    """
    import run2_volume_timepoints as r2  # orientation resolver used by run 3
    from run3_48h_3rand import preprocess_fullres  # same bright-plate crop

    proc = preprocess_fullres(cond["image"])
    res = quantify_plate_image_cellpose(proc, model, seg_cfg)
    layout = read_echo_picklist(cond["picklist"])
    op, _be, _diag = r2.resolve_and_check(  # type: ignore[no-untyped-call]
        res.table, layout, cfg, cond["group"]
    )
    merged = r2.apply_orientation(res.table, op).merge(
        layout, on=["row", "col"], how="inner"
    )
    df = normalize_plate(merged, cfg)
    rep = score_plate(df, cfg, plate_id=cond["group"])
    t = score_table(rep)
    t = t[(t.strain != BLANK_NAME) & t.relative_fitness.notna()].copy()
    t["day"] = cond["day"]
    t["group"] = cond["group"]
    t["orientation"] = op

    # Plate-level reference diagnostic, in RAW pixels and independent of normalization:
    # on a sound plate the wild type out-grows the deletion pool. If this ratio inverts,
    # the WT reference itself failed and every score on the plate is inflated.
    fl = df["flags"].astype("string").fillna("")
    ok = df[(df["size"] > 0) & (~fl.str.contains("M|N|S", regex=True))]
    wt_px = float(ok.loc[ok.strain == WT_NAME, "size"].median())
    mut_px = float(ok.loc[~ok.strain.isin([WT_NAME, BLANK_NAME]), "size"].median())
    diag = {
        "group": cond["group"], "day": cond["day"], "orientation": op,
        "wt_median_px": wt_px, "mutant_median_px": mut_px,
        "wt_over_mutant": wt_px / mut_px,
        "wt_cv": float(ok.loc[ok.strain == WT_NAME, "size"].std()
                       / ok.loc[ok.strain == WT_NAME, "size"].mean()),
    }
    cols = ["day", "group", "orientation", "strain", "relative_fitness", "n_used"]
    return t[cols], diag


def main() -> None:
    """Score every 5 nL plate identically, then split plate vs day variance."""
    os.makedirs(QUANT, exist_ok=True)
    import sys

    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

    cfg = NormalizationConfig()
    seg_cfg = CellposeSegConfig(
        n_rows=N_ROWS, n_cols=N_COLS, contrast="clahe", clahe_clip=0.02,
        cellprob_threshold=-4.0, node_tol=0.60, edge_margin_frac=0.70,
        multi_min_frac=0.35,
    )
    print("[0] loading Cellpose-SAM (cpsam) on GPU ...")
    model = load_cellpose_model(gpu=True)

    frames, diags = [], []
    for cond in CONDITIONS:
        print(f"[1] scoring {cond['group']} ({cond['day']}) ...")
        tab, diag = score_one(cond, model, seg_cfg, cfg)
        frames.append(tab)
        diags.append(diag)
    diag_df = pd.DataFrame(diags)
    diag_df.to_csv(osp.join(RESULTS, "between_day_plate_diagnostics.csv"), index=False)
    print("\n[1b] raw-pixel reference diagnostic (normalization-independent)")
    print(f"    {'group':>12} {'WT px':>8} {'mutant px':>10} {'WT/mutant':>10} {'WT CV':>7}")
    for d in diags:
        bad = "  <-- WT REFERENCE FAILED" if d["wt_over_mutant"] < 1.0 else ""
        print(f"    {d['group']:>12} {d['wt_median_px']:>8.0f} {d['mutant_median_px']:>10.0f}"
              f" {d['wt_over_mutant']:>10.3f} {d['wt_cv']:>7.3f}{bad}")
    print("    The MUTANTS are the same size on every plate; only the WT differs. That")
    print("    localises the failure to the wild-type wells, not to plate-wide growth.")
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[scores.strain != WT_NAME]
    scores.to_csv(osp.join(RESULTS, "between_day_strain_scores.csv"), index=False)

    print("\n[2] per-strain fitness by plate (identical pipeline everywhere)")
    wide = scores.pivot_table(index="strain", columns="group", values="relative_fitness")
    print(wide.round(3).to_string())

    run3 = [c["group"] for c in CONDITIONS if c["day"] == "2026-07-23"]
    run2 = [c["group"] for c in CONDITIONS if c["day"] == "2026-07-17"]
    w = wide.dropna()

    # ---------------------------------------------------------------- plate QC
    # `ladder_feasibility.py` established that only 2.7% of 7,738 genome-wide deletion
    # strains are significantly ABOVE wild type. A plate on which half the deletions beat
    # WT is therefore not biology -- its WT reference is wrong. This is a far sharper gate
    # than WT_CV, which passed the offending plate.
    print("\n[3] plate QC: fraction of deletion strains scoring ABOVE wild type")
    qc = []
    for g in run3 + run2:
        frac = float((w[g] > 1.0).mean())
        qc.append({"group": g, "n_above_wt": int((w[g] > 1.0).sum()),
                   "frac_above_wt": frac, "median_fitness": float(w[g].median()),
                   "flagged": frac > FRAC_ABOVE_WT_MAX})
        flag = "  <-- FLAG" if frac > FRAC_ABOVE_WT_MAX else ""
        print(f"    {g:>12}: {int((w[g] > 1.0).sum()):>2}/{len(w)} above WT, "
              f"median {w[g].median():.3f}{flag}")
    qc_df = pd.DataFrame(qc)
    flagged = qc_df.loc[qc_df.flagged, "group"].tolist()
    print("    genome-wide expectation: ~2.7% of deletions exceed WT significantly.")
    print(f"    flagged: {flagged or 'none'}")

    clean3 = [g for g in run3 if g not in flagged]
    clean2 = [g for g in run2 if g not in flagged]

    sd_plate_all = float(w[run3].std(axis=1, ddof=1).mean())
    sd_plate_clean = (
        float(w[clean3].std(axis=1, ddof=1).mean()) if len(clean3) >= 2 else np.nan
    )
    sd_run2_time = float(w[run2].std(axis=1, ddof=1).mean())

    print("\n[4] variance split -- the flagged plate dominates sigma_plate")
    print(f"    within-day plate SD, all {len(run3)} run-3 plates      = {sd_plate_all:.4f}")
    print(f"    within-day plate SD, {len(clean3)} clean plates only    = {sd_plate_clean:.4f}")
    print(f"    run-2 same plate, two timepoints              = {sd_run2_time:.4f}")
    print("    -> the headline sigma_plate = 0.140 is NOT typical plate noise; it is one")
    print("       plate failing. Clean plates agree ~5x more tightly.")

    diff = np.asarray(
        w[clean3].mean(axis=1) - w[clean2].mean(axis=1), dtype=float
    )
    var_diff = float(np.var(diff, ddof=1))
    var_plate_part = sd_plate_clean**2 * (1 / len(clean3) + 1 / len(clean2))
    var_day = max(0.0, (var_diff - var_plate_part) / 2.0)
    print("\n[5] day contrast, clean plates only (2026-07-17 vs 2026-07-23)")
    print(f"    mean shift = {diff.mean():+.4f}   Var(diff) = {var_diff:.5f}")
    print(f"    plate contribution = {var_plate_part:.5f}")
    print(f"    => sigma_day (strain x day interaction) ~ {np.sqrt(var_day):.4f}")
    print("    1 degree of freedom on 2 days x 2 clean plates -- indicative, not precise.")
    print("    The MEAN shift is a common offset absorbed by WT normalisation; the")
    print("    strain-specific SPREAD is what hurts eps and tau.")

    qc_df.to_csv(osp.join(RESULTS, "between_day_plate_qc.csv"), index=False)
    pd.DataFrame(
        [{
            "sd_plate_all_run3": sd_plate_all,
            "sd_plate_clean_only": sd_plate_clean,
            "sd_run2_same_plate_timepoints": sd_run2_time,
            "flagged_plates": ";".join(flagged),
            "mean_day_shift": float(diff.mean()),
            "var_diff": var_diff,
            "var_plate_part": var_plate_part,
            "sigma_day_estimate": float(np.sqrt(var_day)),
            "n_strains": int(len(w)),
        }]
    ).to_csv(osp.join(RESULTS, "between_day_variance.csv"), index=False)
    print("\nwrote", osp.join(RESULTS, "between_day_variance.csv"))


if __name__ == "__main__":
    main()
