# experiments/019-echo-crispr-array/scripts/compare_segmentation_svg.py
# [[experiments.019-echo-crispr-array.scripts.compare_segmentation_svg]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/compare_segmentation_svg
"""Side-by-side threshold-vs-watershed colony segmentation for all six Run-2
conditions, as zoomable SVGs (vector colony boundaries over the processed plate
photo, so boundaries stay crisp at any zoom).

One SVG per volume; each has three rows (43.7 h, 50.3 h, 72.2 h) x two columns
(threshold in red, watershed in cyan). The base image is the SAME 1400 px processed
capture the pipeline segments, so what you see is exactly what the segmenter sees;
the boundary is drawn as a vector contour of the detected-colony mask returned by
``quantify_plate_image(..., return_masks=True)`` -- one segmentation implementation,
shared with the pipeline (torchcell/sga/image.py).

Run from repo root:
    /Users/michaelvolk/miniconda3/bin/python \
        experiments/019-echo-crispr-array/scripts/compare_segmentation_svg.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from skimage.measure import find_contours

from torchcell.sga.image import _grayscale, quantify_plate_image

# reuse the runner's preprocessing + condition table (single source of truth)
import run2_volume_timepoints as run2

load_dotenv()
IMG_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-echo-crispr-array")
N_ROWS, N_COLS = run2.N_ROWS, run2.N_COLS

METHODS = [("threshold", "#B85450"), ("watershed", "#1FB8C4")]  # red / cyan


def segment_both(proc_path):
    """Return (grayscale, {method: det_mask}, {method: (n_colonies, median_size)})."""
    g = _grayscale(proc_path)
    dets, stats = {}, {}
    for method, _ in METHODS:
        df, det = quantify_plate_image(
            proc_path,
            n_rows=N_ROWS,
            n_cols=N_COLS,
            grid_mode="lattice",
            seg_method=method,
            return_masks=True,
        )
        found = df[df["size"] >= 20]
        dets[method] = det
        stats[method] = (len(found), float(found["size"].median()))
    return g, dets, stats


def render_volume(vol_tag, conds):
    """One SVG: rows = timepoints, cols = (threshold, watershed)."""
    procs = [(c, run2._preprocess(c["image"])) for c in conds]
    g0 = _grayscale(procs[0][1])
    aspect = g0.shape[0] / g0.shape[1]
    panel_w = 6.0  # inches per panel; SVG so real size only sets embed resolution
    fig, axes = plt.subplots(
        len(conds),
        2,
        figsize=(2 * panel_w, len(conds) * panel_w * aspect),
        squeeze=False,
    )
    for ri, (cond, proc) in enumerate(procs):
        g, dets, stats = segment_both(proc)
        for ci, (method, color) in enumerate(METHODS):
            ax = axes[ri][ci]
            ax.imshow(g, cmap="gray", interpolation="none")
            for contour in find_contours(dets[method].astype(float), 0.5):
                ax.plot(contour[:, 1], contour[:, 0], color=color, lw=0.6)
            n, med = stats[method]
            ax.set_title(
                f"{cond['volume_nl']:g} nL, {int(round(cond['hours']))} h  --  "
                f"{method}   n={n}, median={med:.0f}px",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"Colony segmentation: threshold (red) vs watershed (cyan)  --  {vol_tag}\n"
        f"same 1400 px processed capture; vector boundaries (zoom to compare)",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # reserve top band for the 2-line suptitle
    out = osp.join(IMG_DIR, f"run2_seg_compare_{vol_tag}.svg")
    fig.savefig(out)  # vector: crisp boundaries when zoomed (the deliverable)
    fig.savefig(out[:-4] + ".png", dpi=200)  # raster sibling for the PDF build
    plt.close(fig)
    print(
        f"  wrote {osp.basename(out)}(.svg/.png)  ({osp.getsize(out) / 1e6:.1f} MB svg)"
    )


def main():
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none"})
    by_vol = {}
    for c in run2.CONDITIONS:
        by_vol.setdefault(
            f"P{1 if c['volume_nl'] == 2.5 else 2}_{c['volume_nl']:g}nL", []
        ).append(c)
    print(f"rendering {len(by_vol)} volume SVGs to {IMG_DIR}")
    for vol_tag, conds in by_vol.items():
        conds = sorted(conds, key=lambda c: c["hours"])
        render_volume(vol_tag, conds)
    print("done")


if __name__ == "__main__":
    main()
