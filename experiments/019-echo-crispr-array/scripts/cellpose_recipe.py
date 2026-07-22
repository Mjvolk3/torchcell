# experiments/019-echo-crispr-array/scripts/cellpose_recipe.py
# [[experiments.019-echo-crispr-array.scripts.cellpose_recipe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/cellpose_recipe
"""Run ONE Cellpose recipe (contrast x clahe_clip x cellprob) on chosen plates.

Single parameterized entry point used both by the parameter-chase workflow (one
combo per agent, on the faint plates) and by the finalize pass (winning combo on
all six). For each condition it preprocesses (full-res crop), segments, writes an
outline-only QC overlay, optionally saves the crop + instance masks (for the
vector-contour montage), and prints a one-line JSON capture summary:

    RECIPE_JSON {"contrast":..,"clahe_clip":..,"cellprob":..,"stats":[{group,occupied,instances,offgrid},..]}

Run from repo root on a GPU node, e.g.:
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/cellpose_recipe.py \
        --contrast clahe --clahe_clip 0.02 --cellprob -4 --conditions P1_t50,P1_t72 --tag probe0
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv
from PIL import Image, ImageOps

from torchcell.sga import (
    CellposeSegConfig,
    load_cellpose_model,
    quantify_plate_image_cellpose,
)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_cellpose_segmentation as base  # noqa: E402
import run2_volume_timepoints as r2  # noqa: E402

load_dotenv()
IMG_DIR = base.IMG_DIR
QUANT_DIR = base.QUANT_DIR
N_ROWS, N_COLS = base.N_ROWS, base.N_COLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="none", choices=["none", "clahe", "flatfield"])
    ap.add_argument("--clahe_clip", type=float, default=0.01)
    ap.add_argument("--cellprob", type=float, default=-4.0)
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--conditions", default="P1_t50,P1_t72", help="comma list or 'all'")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--save_masks", action="store_true", help="save crop+masks for montage")
    args = ap.parse_args()

    conds = {c["group"]: c for c in r2.CONDITIONS}
    groups = list(conds) if args.conditions == "all" else args.conditions.split(",")

    model = load_cellpose_model(gpu=True)
    seg_cfg = CellposeSegConfig(
        n_rows=N_ROWS,
        n_cols=N_COLS,
        cellprob_threshold=args.cellprob,
        flow_threshold=args.flow,
        contrast=args.contrast,
        clahe_clip=args.clahe_clip,
    )

    stats = []
    for g in groups:
        proc = base.preprocess_fullres(conds[g]["image"])
        res = quantify_plate_image_cellpose(
            proc,
            model,
            seg_cfg,
            overlay_path=osp.join(IMG_DIR, f"run2_cellpose_{args.tag}_overlay_{g}.png"),
            return_masks=args.save_masks,
        )
        if args.save_masks and res.masks is not None:
            np.save(osp.join(QUANT_DIR, f"run2_cellpose_{args.tag}_masks_{g}.npy"), res.masks)
            crop = ImageOps.exif_transpose(Image.open(proc)).convert("RGB")
            crop.save(osp.join(QUANT_DIR, f"run2_cellpose_{args.tag}_crop_{g}.png"))
            with open(osp.join(QUANT_DIR, f"run2_cellpose_{args.tag}_color_{g}.json"), "w") as fh:
                json.dump({str(k): v for k, v in res.kept_color.items()}, fh)
        stats.append(
            dict(
                group=g,
                occupied=int((res.table["size"] > 0).sum()),
                instances=res.n_instances,
                offgrid=res.n_offgrid,
            )
        )
        print(f"    {g}: occupied={stats[-1]['occupied']} instances={res.n_instances}")

    print(
        "RECIPE_JSON "
        + json.dumps(
            dict(
                contrast=args.contrast,
                clahe_clip=args.clahe_clip,
                cellprob=args.cellprob,
                flow=args.flow,
                tag=args.tag,
                stats=stats,
            )
        )
    )


if __name__ == "__main__":
    main()
