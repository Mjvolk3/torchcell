# experiments/019-echo-crispr-array/scripts/run2_cellpose_segmentation.py
# [[experiments.019-echo-crispr-array.scripts.run2_cellpose_segmentation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run2_cellpose_segmentation
"""Run-2 colony quantification with Cellpose-SAM instance segmentation (GPU).

Swaps ONLY the per-colony boundary step of the run-2 pipeline: the array-lattice
fit, orientation resolution, normalization and scoring are reused verbatim from
``run2_volume_timepoints`` so the fitness numbers are directly comparable. For
each of the six (plate x timepoint) images this:

  1. crops to the plate at FULL resolution (no downscale -- the reason for the
     switch: the 1400-px downscale created the shadow-tail artifacts),
  2. runs Cellpose-SAM (``cpsam``) once per plate -> instance masks,
  3. assigns each instance to its array well, invalidating wells that hold two or
     more colonies ('M') and counting off-grid contaminants,
  4. normalizes to the on-plate BY4741 wild-type and scores per-strain fitness,
  5. writes a Cellpose-vs-classical comparison (per-well size correlation, WT CV,
     occupied/rejected counts) on the SAME full-res pixels.

Decision + validation plan: [[experiments.019-echo-crispr-array.cellpose-segmentation-plan]].

Run from repo root on a GPU node (see gh_cellpose_segmentation.slurm):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run2_cellpose_segmentation.py
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

from torchcell.sga import (
    CellposeSegConfig,
    NormalizationConfig,
    load_cellpose_model,
    normalize_plate,
    quantify_plate_image,
    quantify_plate_image_cellpose,
    read_echo_picklist,
    score_plate,
    score_table,
)

# Reuse the run-2 flow verbatim (conditions, geometry, orientation resolver).
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_volume_timepoints as r2  # noqa: E402

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS_DIR = osp.join(EXP_DIR, "results")
QUANT_DIR = osp.join(EXP_DIR, "quant", "cellpose_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "cellpose")
for d in (RESULTS_DIR, QUANT_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)

N_ROWS, N_COLS = r2.N_ROWS, r2.N_COLS


def preprocess_fullres(path: str) -> str:
    """Crop to the plate with the same bright-plate detector as
    ``r2._preprocess`` but KEEP native resolution (no downscale).
    """
    out = osp.join(QUANT_DIR, osp.splitext(osp.basename(path))[0] + "_fullres.png")
    if osp.exists(out):
        return out  # deterministic crop cache -> safe for parallel readers
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    g = np.asarray(im.convert("L"), float)
    bright = ndimage.gaussian_filter(g, 40) > 0.80 * np.percentile(g, 99)
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    big = int(sizes.argmax()) + 1
    ys, xs = np.where(lab == big)
    pad = int(0.02 * max(g.shape))
    r0, r1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + pad)
    c0, c1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + pad)
    out = osp.join(QUANT_DIR, osp.splitext(osp.basename(path))[0] + "_fullres.png")
    im.crop((c0, r0, c1, r1)).save(out)
    return out


def _draw_classical_overlay(path: str, det: np.ndarray, out: str) -> None:
    """Classical-segmentation boundary (green) on the SAME full-res crop, so the
    Cellpose overlay and this one can be flipped side-by-side. The classical mask is
    a single union (touching colonies are NOT separated) -- that's exactly the
    limitation Cellpose's instance masks fix, and it shows here visually.
    """
    from skimage.segmentation import find_boundaries

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    over = np.asarray(im).copy()
    over[find_boundaries(det.astype(bool), mode="inner")] = [0, 255, 0]
    Image.fromarray(over).save(out)


def _wt_cv(df: pd.DataFrame, wt_name: str) -> float:
    """Coefficient of variation of raw colony size across the on-plate wild-type
    wells (lower = tighter reference = better assay). Target <= classical ~0.11-0.17.
    """
    wt = df.loc[(df["strain"] == wt_name) & (~df["is_missing"]), "size"].to_numpy(float)
    return float(np.std(wt) / np.mean(wt)) if wt.size and np.mean(wt) else np.nan


def main() -> None:
    cfg = NormalizationConfig()
    # finalize recipe (sizing + detection sweeps, 2026.07.22 round 2): CLAHE 0.02 +
    # cellprob -4, Otsu size-tightening (removes the ~35% halo), wider edge margin to
    # keep row A/P colonies, node_tol 0.60, and the multi_min_frac gate, with the
    # colony-validity invalidation model (M/N/C) active.
    seg_cfg = CellposeSegConfig(
        n_rows=N_ROWS,
        n_cols=N_COLS,
        contrast="clahe",
        clahe_clip=0.02,
        cellprob_threshold=-4.0,
        node_tol=0.60,
        edge_margin_frac=0.70,
        multi_min_frac=0.5,
    )

    print("[0] loading Cellpose-SAM (cpsam) on GPU ...")
    model = load_cellpose_model(gpu=True)

    colonies, reports, cmp_rows = {}, {}, []
    print(
        "[1] full-res crop -> Cellpose instance seg -> register -> normalize (6 images)"
    )
    for cond in r2.CONDITIONS:
        g = cond["group"]
        proc = preprocess_fullres(cond["image"])

        res = quantify_plate_image_cellpose(
            proc,
            model,
            seg_cfg,
            overlay_path=osp.join(IMG_DIR, f"run2_cellpose_overlay_{g}.png"),
        )
        grid = res.table
        layout = read_echo_picklist(cond["picklist"])
        op, blanks_empty, _diag = r2.resolve_and_check(grid, layout, cfg, g)
        merged = r2.apply_orientation(grid, op).merge(
            layout, on=["row", "col"], how="inner"
        )
        df = normalize_plate(merged, cfg)
        for k in ("group", "plate", "volume_nl", "hours", "agar"):
            df[k] = cond[k]
        colonies[g] = df
        reports[g] = score_plate(df, cfg, plate_id=g)

        # classical segmentation on the SAME full-res pixels -> apples-to-apples
        cls, cls_det = quantify_plate_image(
            proc, n_rows=N_ROWS, n_cols=N_COLS, grid_mode="lattice", return_masks=True
        )
        _draw_classical_overlay(
            proc, cls_det, osp.join(IMG_DIR, f"run2_classical_overlay_{g}.png")
        )
        cls_m = r2.apply_orientation(cls, op).merge(
            layout, on=["row", "col"], how="inner"
        )
        cls_df = normalize_plate(cls_m, cfg)
        j = df.merge(cls_df, on=["row", "col"], suffixes=("_cp", "_cl"))
        both = (j["size_cp"] > 0) & (j["size_cl"] > 0)
        pr = (
            pearsonr(j.loc[both, "size_cp"], j.loc[both, "size_cl"])[0]
            if both.sum() > 2
            else np.nan
        )
        sr = (
            spearmanr(j.loc[both, "size_cp"], j.loc[both, "size_cl"])[0]
            if both.sum() > 2
            else np.nan
        )
        cmp_rows.append(
            dict(
                group=g,
                n_instances=res.n_instances,
                n_offgrid=res.n_offgrid,
                occupied_cp=int((df["size"] > 0).sum()),
                occupied_cl=int((cls_df["size"] > 0).sum()),
                multi_cp=int(df["flags"].fillna("").str.contains("M").sum()),
                multi_cl=int(cls_df["flags"].fillna("").str.contains("M").sum()),
                size_pearson=pr,
                size_spearman=sr,
                wt_cv_cp=_wt_cv(df, cfg.wt_name),
                wt_cv_cl=_wt_cv(cls_df, cfg.wt_name),
                wt_median_norm=reports[g].wt_median_norm,
            )
        )
        print(
            f"    {g}: op={op} instances={res.n_instances} offgrid={res.n_offgrid} "
            f"occupied {int((df['size'] > 0).sum())}/{len(df)} "
            f"M={int(df['flags'].fillna('').str.contains('M').sum())} "
            f"blanks_empty={blanks_empty}/6 size_r={pr:.3f} "
            f"WT_CV cp={_wt_cv(df, cfg.wt_name):.3f} cl={_wt_cv(cls_df, cfg.wt_name):.3f}"
        )

    all_col = pd.concat(colonies.values(), ignore_index=True)
    all_col.to_csv(
        osp.join(RESULTS_DIR, "run2_cellpose_colonies_registered.csv"), index=False
    )

    print("\n[2] per-strain fitness (mutant / BY4741) per condition")
    rows = []
    for cond in r2.CONDITIONS:
        g = cond["group"]
        for s in reports[g].strains:
            if s.strain == cfg.blank_name or s.relative_fitness is None:
                continue
            rows.append(
                dict(
                    group=g,
                    plate=cond["plate"],
                    volume_nl=cond["volume_nl"],
                    hours=cond["hours"],
                    strain=s.strain,
                    fitness=s.relative_fitness,
                    fitness_sd=s.fitness_sd,
                    # standard error of the mean fitness across the strain's replicate
                    # colonies (SD / sqrt(n)); the reference-comparable uncertainty.
                    fitness_se=(
                        s.fitness_sd / (s.n_used**0.5)
                        if s.fitness_sd is not None and s.n_used
                        else None
                    ),
                    n_used=s.n_used,
                    n_total=s.n_total,
                    pvalue=s.pvalue,
                )
            )
        score_table(reports[g]).sort_values("relative_fitness").to_csv(
            osp.join(RESULTS_DIR, f"run2_cellpose_strain_scores_{g}.csv"), index=False
        )
    pd.DataFrame(rows).to_csv(
        osp.join(RESULTS_DIR, "run2_cellpose_fitness_by_condition.csv"), index=False
    )

    print("\n[3] Cellpose vs classical (same full-res pixels)")
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(osp.join(RESULTS_DIR, "run2_cellpose_vs_classical.csv"), index=False)
    print(cmp.round(3).to_string(index=False))
    print(f"\nwrote results -> {RESULTS_DIR}")
    print(f"wrote overlays -> {IMG_DIR}")


if __name__ == "__main__":
    main()
