# experiments/019-echo-crispr-array/scripts/run3_correct_p3.py
# [[experiments.019-echo-crispr-array.scripts.run3_correct_p3]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run3_correct_p3
"""Regenerate P3's run-3 detection overlay from cached masks using the CANONICAL pipeline
seg-config -- no special grid override. GPU-free (reads `run3_masks_P3.npy`), so it just
re-renders the same overlay `run3_48h_3rand.py` produces for P1/P2, keeping all three
consistent without re-running Cellpose.

Supersedes the earlier spanning-grid re-registration. That correction was written when the
default lattice fit P3's top row one row low; the pipeline's `homography_refit` (a projective
index->pixel fit, now on by default) already recovers all 16 rows, so the override became
redundant AND harmful: it re-derived the 24 column lines with `_fit_lines`, and on a plate
with missing/contaminated wells that force-fit introduced column phase-slips -- collapsing
pairs of legitimate neighbour colonies onto single grid nodes and firing ~50 false multi (M,
red) + collision (X, pink) flags. The canonical grid registers P3 cleanly: 350/384 occupied,
rows 1-16, ZERO false multis.

P3's detection is therefore fine; P3 fails QC for wet-lab reasons only (blank wells
contaminated in every layout x orientation -> ambiguous orientation + noisy WT), so it stays
excluded from the batch-effect and is flagged for re-plate. This overlay shows the honest
detection.

Run from repo root (after run3_48h_3rand.py has written the cached masks):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run3_correct_p3.py
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
from dotenv import load_dotenv

from torchcell.sga.cellpose_seg import CellposeSegConfig, quantify_plate_image_cellpose

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
QUANT_DIR = osp.join(EXP_DIR, "quant", "run3_proc")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "run3")


def main() -> None:
    crop = osp.join(QUANT_DIR, "P3_OD1-5nL_TCsingleKO_crop.png")
    masks = np.load(osp.join(QUANT_DIR, "run3_masks_P3.npy"))
    # EXACT run3_48h_3rand.py seg_cfg (tighten_size + recover_missed_wells at their
    # True defaults) -- no grid monkeypatch, so this reproduces the canonical overlay.
    cfg = CellposeSegConfig(
        n_rows=16, n_cols=24, contrast="clahe", clahe_clip=0.02, cellprob_threshold=-4.0,
        node_tol=0.60, edge_margin_frac=0.70, multi_min_frac=0.5,
    )
    res = quantify_plate_image_cellpose(
        crop, model=None, cfg=cfg, precomputed_masks=masks,
        overlay_path=osp.join(IMG_DIR, "run3_overlay_P3.png"),
    )
    df = res.table
    occ = int((df["size"] > 0).sum())
    n_multi = int(df["flags"].str.contains("M").sum())
    print(f"P3 overlay (canonical grid) -> occupied {occ}/384, multi(M) flags {n_multi}")
    print(f"wrote {osp.join(IMG_DIR, 'run3_overlay_P3.png')}")


if __name__ == "__main__":
    main()
