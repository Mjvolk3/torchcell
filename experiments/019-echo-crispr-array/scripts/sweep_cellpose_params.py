# experiments/019-echo-crispr-array/scripts/sweep_cellpose_params.py
# [[experiments.019-echo-crispr-array.scripts.sweep_cellpose_params]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/sweep_cellpose_params
"""Tune Cellpose-SAM to capture faint colonies + stop under-sizing, optimizing the
metric we actually care about: per-genotype replicate SD after bias reduction.

Motivation (visual QC, 2026.07.21): with the defaults (cellprob_threshold=0,
flow_threshold=0.4) Cellpose MISSES faint 2.5 nL colonies (bottom rows) and
UNDER-SIZES some (the colony visibly peeks past the mask). Lowering
``cellprob_threshold`` makes the mask reconstruction more permissive -> more faint
cells detected and masks grown out to the true edge; ``flow_threshold`` trades mask
count vs shape quality. This sweeps that grid and, for each setting, scores the
plates and reports the mean per-strain fitness SD (lower = tighter replicates =
better measurement) alongside capture (colonies used) and mean colony size (mask
growth). We optimize SD, NOT the Costanzo correlation.

Runs the sweep on SWEEP_CONDITIONS (cheap), then re-runs the winning setting over
ALL conditions, writing outline-only QC overlays for review.

Run from repo root on a GPU node:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/sweep_cellpose_params.py
"""

from __future__ import annotations

import os.path as osp
import sys

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
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_cellpose_segmentation as base  # noqa: E402  (reuses preprocess_fullres, dirs)
import run2_volume_timepoints as r2  # noqa: E402

load_dotenv()
RESULTS_DIR = base.RESULTS_DIR
IMG_DIR = base.IMG_DIR
N_ROWS, N_COLS = base.N_ROWS, base.N_COLS

# grid: cellprob_threshold is the main knob (lower -> more/larger masks); flow at two
# levels. Extend as needed.
CELLPROB = [0.0, -1.0, -2.0, -3.0, -4.0]
FLOW = [0.4, 0.6]
# sweep on the 2.5 nL plates where faint colonies are the problem (cheap); the winner
# is then applied to all six.
SWEEP_CONDITIONS = ["P1_t50", "P1_t72"]


def _mean_strain_sd(report, blank_name: str) -> float:
    """Mean per-strain fitness SD over non-blank strains with >=2 replicates -- the
    replicate-precision objective (lower is better).
    """
    sds = [
        s.fitness_sd
        for s in report.strains
        if s.strain != blank_name and s.fitness_sd is not None
    ]
    return float(np.mean(sds)) if sds else np.nan


def _evaluate(model, cfg_norm, seg_cfg, cond) -> dict[str, float]:
    proc = base.preprocess_fullres(cond["image"])
    res = quantify_plate_image_cellpose(proc, model, seg_cfg)
    grid = res.table
    layout = read_echo_picklist(cond["picklist"])
    op, _blanks, _diag = r2.resolve_and_check(grid, layout, cfg_norm, cond["group"])
    merged = r2.apply_orientation(grid, op).merge(layout, on=["row", "col"], how="inner")
    df = normalize_plate(merged, cfg_norm)
    rep = score_plate(df, cfg_norm, plate_id=cond["group"])
    used = int(sum(s.n_used for s in rep.strains if s.strain != cfg_norm.blank_name))
    occ = int((df["size"] > 0).sum())
    mean_size = float(df.loc[df["size"] > 0, "size"].mean())
    return dict(
        mean_strain_sd=_mean_strain_sd(rep, cfg_norm.blank_name),
        colonies_used=used,
        occupied=occ,
        mean_size=mean_size,
        instances=res.n_instances,
    )


def main() -> None:
    cfg_norm = NormalizationConfig()
    conds = {c["group"]: c for c in r2.CONDITIONS}

    print("[0] loading Cellpose-SAM (cpsam) on GPU ...")
    model = load_cellpose_model(gpu=True)

    print(f"[1] sweep cellprob x flow on {SWEEP_CONDITIONS} "
          f"({len(CELLPROB) * len(FLOW)} settings)")
    rows = []
    for cp in CELLPROB:
        for fl in FLOW:
            seg_cfg = CellposeSegConfig(
                n_rows=N_ROWS, n_cols=N_COLS, cellprob_threshold=cp, flow_threshold=fl
            )
            per = [_evaluate(model, cfg_norm, seg_cfg, conds[g]) for g in SWEEP_CONDITIONS]
            agg = {
                "cellprob": cp,
                "flow": fl,
                "mean_strain_sd": float(np.mean([p["mean_strain_sd"] for p in per])),
                "colonies_used": int(np.mean([p["colonies_used"] for p in per])),
                "occupied": int(np.mean([p["occupied"] for p in per])),
                "mean_size": float(np.mean([p["mean_size"] for p in per])),
            }
            rows.append(agg)
            print(f"    cellprob={cp:+.0f} flow={fl:.1f} -> "
                  f"strain_SD={agg['mean_strain_sd']:.4f} used={agg['colonies_used']} "
                  f"occ={agg['occupied']} size={agg['mean_size']:.0f}")

    sweep = pd.DataFrame(rows).sort_values("mean_strain_sd").reset_index(drop=True)
    sweep.to_csv(osp.join(RESULTS_DIR, "run2_cellpose_param_sweep.csv"), index=False)
    print("\n[2] sweep table (sorted by mean per-strain SD, lower is better):")
    print(sweep.round(4).to_string(index=False))

    best = sweep.iloc[0]
    bcp, bfl = float(best["cellprob"]), float(best["flow"])
    print(f"\n[3] best: cellprob={bcp:+.0f} flow={bfl:.1f} "
          f"(strain_SD={best['mean_strain_sd']:.4f}); "
          f"re-running ALL conditions + writing outline overlays")
    seg_cfg = CellposeSegConfig(
        n_rows=N_ROWS, n_cols=N_COLS, cellprob_threshold=bcp, flow_threshold=bfl
    )
    for g, cond in conds.items():
        proc = base.preprocess_fullres(cond["image"])
        quantify_plate_image_cellpose(
            proc,
            model,
            seg_cfg,
            overlay_path=osp.join(IMG_DIR, f"run2_cellpose_tuned_overlay_{g}.png"),
        )
    print(f"wrote sweep -> {osp.join(RESULTS_DIR, 'run2_cellpose_param_sweep.csv')}")
    print(f"wrote tuned overlays -> {IMG_DIR}")


if __name__ == "__main__":
    main()
