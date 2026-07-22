# experiments/019-echo-crispr-array/scripts/sweep_cellpose_contrast.py
# [[experiments.019-echo-crispr-array.scripts.sweep_cellpose_contrast]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/sweep_cellpose_contrast
"""Fix Cellpose DETECTION of faint colonies via contrast pre-processing.

The faint 2.5 nL colonies are visible to the eye but Cellpose misses them because
their contrast against the bright agar is low; lowering ``cellprob`` only grew masks
it already found (capture stayed flat). This compares contrast pre-steps applied to
the image Cellpose sees -- ``none`` vs ``clahe`` (adaptive histogram equalization)
vs ``flatfield`` (divide out the illumination background) -- at the tuned
``cellprob_threshold=-4``, and reports how many colonies each captures (occupied
wells + raw instances). Outline-only overlays are written per (plate, contrast) so
the missed cells can be counted by eye. The lattice fit is unchanged (original
image); only Cellpose's input is enhanced.

Run from repo root on a GPU node:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/sweep_cellpose_contrast.py
"""

from __future__ import annotations

import os.path as osp
import sys

import pandas as pd
from dotenv import load_dotenv

from torchcell.sga import (
    CellposeSegConfig,
    load_cellpose_model,
    quantify_plate_image_cellpose,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_cellpose_segmentation as base  # noqa: E402
import run2_volume_timepoints as r2  # noqa: E402

load_dotenv()
RESULTS_DIR = base.RESULTS_DIR
IMG_DIR = base.IMG_DIR
N_ROWS, N_COLS = base.N_ROWS, base.N_COLS

CELLPROB = -4.0
CONTRASTS = ["none", "clahe", "flatfield"]
CONDITIONS = ["P1_t50", "P1_t72"]  # the faint 2.5 nL plates with missed colonies


def main() -> None:
    conds = {c["group"]: c for c in r2.CONDITIONS}
    print("[0] loading Cellpose-SAM (cpsam) on GPU ...")
    model = load_cellpose_model(gpu=True)

    rows = []
    for g in CONDITIONS:
        proc = base.preprocess_fullres(conds[g]["image"])
        for contrast in CONTRASTS:
            seg_cfg = CellposeSegConfig(
                n_rows=N_ROWS,
                n_cols=N_COLS,
                cellprob_threshold=CELLPROB,
                contrast=contrast,
            )
            res = quantify_plate_image_cellpose(
                proc,
                model,
                seg_cfg,
                overlay_path=osp.join(
                    IMG_DIR, f"run2_cellpose_contrast_{g}_{contrast}.png"
                ),
            )
            occ = int((res.table["size"] > 0).sum())
            rows.append(
                dict(
                    group=g,
                    contrast=contrast,
                    instances=res.n_instances,
                    occupied=occ,
                    offgrid=res.n_offgrid,
                )
            )
            print(f"    {g} contrast={contrast:9s} -> instances={res.n_instances} "
                  f"occupied={occ} offgrid={res.n_offgrid}")

    out = pd.DataFrame(rows)
    out.to_csv(osp.join(RESULTS_DIR, "run2_cellpose_contrast_sweep.csv"), index=False)
    print("\n[1] capture by contrast method (higher instances/occupied = more captured):")
    print(out.to_string(index=False))
    print(f"\nwrote overlays -> {IMG_DIR}/run2_cellpose_contrast_<cond>_<method>.png")
    print(f"wrote table    -> {osp.join(RESULTS_DIR, 'run2_cellpose_contrast_sweep.csv')}")


if __name__ == "__main__":
    main()
