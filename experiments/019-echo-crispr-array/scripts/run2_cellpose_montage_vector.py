# experiments/019-echo-crispr-array/scripts/run2_cellpose_montage_vector.py
# [[experiments.019-echo-crispr-array.scripts.run2_cellpose_montage_vector]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-echo-crispr-array/scripts/run2_cellpose_montage_vector
"""2x2 Cellpose segmentation montage with plate row/column axes (vector, zoomable).

Columns = dispensed volume (2.5 nL / 5 nL); rows = growth time (default 50 h, 72 h).
Each panel is the plate image (raster) with every colony's Cellpose boundary drawn
as a VECTOR contour, tightened by a small erosion so the outline hugs the colony
(the raw mask includes the diffuse halo, which leaves a crescent of agar inside a
loose outline). Plate coordinates A-P (rows) x 1-24 (columns), A1 at TOP-LEFT, are
labelled with tick marks on ALL FOUR edges (derived from the lattice fit), so any
colony's well can be read off from whichever axis is nearest.

Consumes the crop PNG + instance-mask .npy that ``cellpose_recipe.py --save_masks``
wrote for a given ``--tag`` (default ``best``).

Run from repo root:
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/019-echo-crispr-array/scripts/run2_cellpose_montage_vector.py [--tag best]
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage.measure import find_contours
from skimage.morphology import disk

from torchcell.sga.cellpose_seg import _CATEGORY_COLOR, _fit_lattice

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import run2_cellpose_segmentation as base  # noqa: E402

load_dotenv()
IMG_DIR = base.IMG_DIR
QUANT_DIR = base.QUANT_DIR
N_ROWS, N_COLS = base.N_ROWS, base.N_COLS  # 16 x 24 (full 384, A1 origin)

# columns = volume, rows = all three growth times (3x2 = all six captures).
COLUMNS = [("P1", "2.5 nL"), ("P2", "5 nL")]
ROWS = [("t44", "44 h"), ("t50", "50 h"), ("t72", "72 h")]
ERODE_PX = 3  # tighten the outline inward off the diffuse halo
LINE_WIDTH = 0.25  # matplotlib points; kept hairline-thin (vector, so it stays crisp)


def _row_labels() -> list[str]:
    return [chr(ord("A") + r) for r in range(N_ROWS)]


def _draw_panel(
    ax: plt.Axes,
    tag: str,
    plate: str,
    t: str,
    title: str,
    tick_fs: float = 3.5,
    title_fs: float = 9,
) -> None:
    crop_p = osp.join(QUANT_DIR, f"run2_cellpose_{tag}_crop_{plate}_{t}.png")
    mask_p = osp.join(QUANT_DIR, f"run2_cellpose_{tag}_masks_{plate}_{t}.npy")
    if not (osp.exists(crop_p) and osp.exists(mask_p)):
        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    img = np.asarray(Image.open(crop_p))
    ax.imshow(img)
    masks = np.load(mask_p)

    # only the pipeline's kept (in-gel, on-grid) instances are drawn, each coloured by
    # its invalidation category (green accepted / red multi / orange neighbour / purple
    # non-circular) -- so off-plate/frame detections never appear.
    color_p = osp.join(QUANT_DIR, f"run2_cellpose_{tag}_color_{plate}_{t}.json")
    kept_color = json.load(open(color_p)) if osp.exists(color_p) else {}
    sel = disk(ERODE_PX)
    for i_str, cat in kept_color.items():
        i = int(i_str)
        ys, xs = np.where(masks == i)
        if ys.size == 0:
            continue
        y0, x0 = int(ys.min()), int(xs.min())
        sub = masks[y0 : ys.max() + 1, x0 : xs.max() + 1] == i
        eroded = binary_erosion(sub, structure=sel)
        sub = (eroded if eroded.any() else sub).astype(float)
        sub = np.pad(sub, 1)
        rgb = tuple(c / 255 for c in _CATEGORY_COLOR.get(cat, _CATEGORY_COLOR[""]))
        for c in find_contours(sub, 0.5):
            ax.plot(c[:, 1] + x0 - 1, c[:, 0] + y0 - 1, color=rgb, lw=LINE_WIDTH)

    # plate coordinate axes from the lattice fit: A-P rows, 1-24 cols, A1 top-left
    g = np.asarray(Image.open(crop_p).convert("L"), float)
    nodes, _pitch, _inv, _theta, _ctr, _roi = _fit_lattice(g, N_ROWS, N_COLS, "auto")
    row_y = nodes[:, :, 0].mean(axis=1)
    col_x = nodes[:, :, 1].mean(axis=0)
    ax.set_xticks(col_x)
    ax.set_xticklabels([str(c + 1) for c in range(N_COLS)], fontsize=tick_fs)
    ax.set_yticks(row_y)
    ax.set_yticklabels(_row_labels(), fontsize=tick_fs)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.tick_params(
        top=True, bottom=True, left=True, right=True,
        labeltop=True, labelbottom=True, labelleft=True, labelright=True,
        length=2, width=0.4, pad=1,
    )
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
    ax.set_title(title, fontsize=title_fs, pad=12)


def _write_per_page(tag: str) -> None:
    """One standalone full-plate figure per capture (all six), for a page-filling,
    zoomable read of a single plate. Same vector validity outlines + A1-P24 axes as a
    montage panel, but larger axis fonts since each plate now owns the whole page --
    lets the green accepted boundary be inspected well by well."""
    for plate, vol in COLUMNS:
        for t, hrs in ROWS:
            fig, ax = plt.subplots(figsize=(8.5, 8.9))
            _draw_panel(ax, tag, plate, t, f"{vol} - {hrs}", tick_fs=5.0, title_fs=13)
            fig.tight_layout()
            out = osp.join(IMG_DIR, f"run2_cellpose_page_{plate}_{t}.svg")
            fig.savefig(out)
            fig.savefig(out.replace(".svg", ".png"), dpi=200)
            plt.close(fig)
            print(f"wrote per-page plate -> {out} (+ .png)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="best")
    args = ap.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)

    fig, axes = plt.subplots(len(ROWS), len(COLUMNS), figsize=(13, 13.5))
    for i, (t, hrs) in enumerate(ROWS):
        for j, (plate, vol) in enumerate(COLUMNS):
            _draw_panel(axes[i, j], args.tag, plate, t, f"{vol} - {hrs}")
    fig.suptitle("Cellpose segmentation (vector outlines, plate A1-P24 axes)", fontsize=11)
    fig.tight_layout()
    out = osp.join(IMG_DIR, "run2_cellpose_montage.svg")
    fig.savefig(out)
    fig.savefig(out.replace(".svg", ".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {len(ROWS)}x{len(COLUMNS)} montage with plate axes -> {out} (+ .png)")

    _write_per_page(args.tag)


if __name__ == "__main__":
    main()
