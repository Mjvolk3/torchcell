# experiments/019-echo-crispr-array/scripts/run3_correct_p3.py
# [[experiments.019-echo-crispr-array.scripts.run3_correct_p3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run3_correct_p3
"""Correct P3's detection overlay (fig 47): its lattice fit one row too low (the faint top
row was skipped, parking the bottom grid row on the frame). Re-fit the grid shifted UP one
row pitch so it spans rows 1-16, then re-quantify from the cached masks and re-write the
overlay. This fixes the REGISTRATION only; P3 remains a QC failure on contamination (blank
wells grown in every layout x orientation), so it stays excluded from the batch-effect --
this just makes the overlay show the true detection instead of the misregistered one.

Run from repo root (after run3_48h_3rand.py has written the cached masks):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run3_correct_p3.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv

import torchcell.sga.cellpose_seg as cs
from torchcell.sga.cellpose_seg import (
    CellposeSegConfig,
    _fit_lattice,
    _instance_props,
    _relax_lattice,
    quantify_plate_image_cellpose,
)
from torchcell.sga.image import _fit_lines, _grayscale, _rotate

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
QUANT_DIR = osp.join(EXP_DIR, "quant", "run3_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "run3")


def spanning_grid(crop: str, masks: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Fit the row/column lines DIRECTLY to the Cellpose centroids so the grid spans the
    full detected extent -- top row to bottom row. P3's original lattice fit one row too
    low with too small a row pitch (16 rows couldn't span all 16 colony rows); ``_fit_lines``
    (which penalises out-of-range indices, forcing the pitch to stretch) lands all 16 rows
    on the detected colonies. Then relax onto the centroid medians for the residual
    non-uniform spacing.
    """
    g = _grayscale(crop)
    _n0, pitch, _inv, theta, center, _roi = _fit_lattice(g, n_rows, n_cols, "auto")
    props = _instance_props(masks, 50)
    allc = np.array([[p["cy"], p["cx"]] for p in props])
    dd = np.sqrt(((allc[:, None, :] - allc[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dd, np.inf)
    cf = allc[dd.min(axis=1) < 1.8 * pitch]
    cr = _rotate(cf, -theta, center)
    ys = _fit_lines(cr[:, 0], n_rows)  # span-forcing row lines (top -> bottom)
    xs = _fit_lines(cr[:, 1], n_cols)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    seed = _rotate(np.stack([gy.ravel(), gx.ravel()], 1), theta, center).reshape(
        n_rows, n_cols, 2
    )
    return _relax_lattice(seed, cf, n_rows, n_cols, theta, center)


def main() -> None:
    crop = osp.join(QUANT_DIR, "P3_OD1-5nL_TCsingleKO_crop.png")
    masks = np.load(osp.join(QUANT_DIR, "run3_masks_P3.npy"))
    # recover_missed_wells OFF: Cellpose already detected P3's top row, and with the grid
    # shifted onto the true rows the recovery probe otherwise fires on the lid-seam frame
    # just above row 1 (blue frame artifacts). Detection of the real colonies is unchanged.
    cfg = CellposeSegConfig(
        n_rows=16, n_cols=24, contrast="clahe", clahe_clip=0.02, cellprob_threshold=-4.0,
        node_tol=0.60, edge_margin_frac=0.70, multi_min_frac=0.5, tighten_size=False,
        recover_missed_wells=False,
    )
    grid = spanning_grid(crop, masks, cfg.n_rows, cfg.n_cols)
    orig = cs._homography_lattice
    cs._homography_lattice = lambda nodes, cents, nr, nc, p, **k: grid
    res = quantify_plate_image_cellpose(
        crop, model=None, cfg=cfg, precomputed_masks=masks,
        overlay_path=osp.join(IMG_DIR, "run3_overlay_P3.png"),
    )
    cs._homography_lattice = orig
    df = res.table
    occ = int((df["size"] > 0).sum())
    print(f"P3 corrected overlay -> occupied {occ}/384 (top row now registered)")
    print(f"wrote {osp.join(IMG_DIR, 'run3_overlay_P3.png')}")


if __name__ == "__main__":
    main()
