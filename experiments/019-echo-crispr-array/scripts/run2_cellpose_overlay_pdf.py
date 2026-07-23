# experiments/019-echo-crispr-array/scripts/run2_cellpose_overlay_pdf.py
# [[experiments.019-echo-crispr-array.scripts.run2_cellpose_overlay_pdf]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run2_cellpose_overlay_pdf
"""One-plate-per-page PDF of the six Cellpose segmentation overlays for visual review.

The side-by-side montage is too small to inspect 384 wells; this renders each
(plate x timepoint) overlay full-bleed on its own page so the reviewer can zoom to
individual colonies. Page titles carry the per-plate occupied / multi counts read from
``run2_cellpose_vs_classical.csv`` (written by ``run2_cellpose_segmentation.py``), so the
grid-registration fix (homography lattice refit -- e.g. P2_t72 recovering rows E/L) is
auditable against the numbers. Reads the overlays the segmentation runner already wrote.

Run from repo root (after run2_cellpose_segmentation.py):
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run2_cellpose_overlay_pdf.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "019-echo-crispr-array", "cellpose")
RESULTS_DIR = osp.join(EXP_DIR, "results")

# (group, plate, volume, hours) in review order
PLATES = [
    ("P1_t44", "P1", "2.5 nL", "44 h"),
    ("P2_t44", "P2", "5 nL", "44 h"),
    ("P1_t50", "P1", "2.5 nL", "50 h"),
    ("P2_t50", "P2", "5 nL", "50 h"),
    ("P1_t72", "P1", "2.5 nL", "72 h"),
    ("P2_t72", "P2", "5 nL", "72 h"),
]


def main() -> None:
    cmp_path = osp.join(RESULTS_DIR, "run2_cellpose_vs_classical.csv")
    counts = (
        pd.read_csv(cmp_path).set_index("group") if osp.exists(cmp_path) else None
    )
    out = osp.join(IMG_DIR, "run2_cellpose_redetection.pdf")
    with PdfPages(out) as pdf:
        for group, plate, vol, hrs in PLATES:
            p = osp.join(IMG_DIR, f"run2_cellpose_overlay_{group}.png")
            if not osp.exists(p):
                print(f"WARNING missing overlay: {osp.basename(p)}")
                continue
            im = Image.open(p)
            w, h = im.size
            fig = plt.figure(figsize=(11, 11 * h / w + 0.5))
            ax = fig.add_axes((0, 0, 1, h / w / (h / w + 0.5 / 11)))
            ax.imshow(im)
            ax.axis("off")
            sub = ""
            if counts is not None and group in counts.index:
                r = counts.loc[group]
                sub = (
                    f"   occupied {int(r['occupied_cp'])}/384   "
                    f"multi-flags {int(r['multi_cp'])}   WT CV {r['wt_cv_cp']:.3f}"
                )
            fig.suptitle(
                f"{group}   {plate} ({vol}) - {hrs}{sub}",
                fontsize=12,
                y=0.995,
            )
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
    print(f"wrote review PDF -> {out}")


if __name__ == "__main__":
    main()
