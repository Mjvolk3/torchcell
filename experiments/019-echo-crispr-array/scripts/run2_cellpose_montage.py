# experiments/019-echo-crispr-array/scripts/run2_cellpose_montage.py
# [[experiments.019-echo-crispr-array.scripts.run2_cellpose_montage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run2_cellpose_montage
"""All six Cellpose segmentation overlays in one side-by-side montage.

Same idea as the threshold-vs-watershed seg-compare figure: a single sheet showing
every (plate x timepoint) capture with the Cellpose outlines drawn, so the whole
run can be eyeballed at once. Rows = volume (P1 2.5 nL, P2 5 nL); columns = growth
time (44, 50, 72 h). Reads the outline overlays already written by the segmentation
/ sweep runners (``PREFIX`` selects which set).

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run2_cellpose_montage.py [PREFIX]
where PREFIX defaults to ``run2_cellpose_tuned_overlay`` (the tuned pass); pass
``run2_cellpose_overlay`` for the default-parameter overlays.
"""

from __future__ import annotations

import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "cellpose")

ROWS = [("P1", "2.5 nL"), ("P2", "5 nL")]
COLS = [("t44", "44 h"), ("t50", "50 h"), ("t72", "72 h")]


def main() -> None:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "run2_cellpose_tuned_overlay"
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(15, 7.5))
    missing = []
    for i, (plate, vol) in enumerate(ROWS):
        for j, (t, hrs) in enumerate(COLS):
            ax = axes[i, j]
            ax.axis("off")
            p = osp.join(IMG_DIR, f"{prefix}_{plate}_{t}.png")
            if not osp.exists(p):
                missing.append(osp.basename(p))
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                continue
            ax.imshow(Image.open(p))
            ax.set_title(f"{plate} ({vol}) - {hrs}", fontsize=9)
    fig.suptitle(f"Cellpose segmentation, all run-2 captures ({prefix})", fontsize=11)
    fig.tight_layout()
    out = osp.join(IMG_DIR, "run2_cellpose_montage.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    if missing:
        print(f"WARNING missing overlays: {missing}")
    print(f"wrote montage -> {out} (+ .svg)")


if __name__ == "__main__":
    main()
