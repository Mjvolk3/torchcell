# experiments/028-knockout-expression/scripts/comparison_designs.py
# [[experiments.028-knockout-expression.scripts.comparison_designs]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-knockout-expression/scripts/comparison_designs
"""Schematic of every comparison design used in the cross-study analysis.

Each panel draws the objects compared (strain x gene matrices, cells of one genotype,
the wild-type reference) and what one number in the corresponding result figure is
computed from. Colors are the panel colors used throughout: Kemmeren yellow, Sameith
purple, Nadal-Ribelles orange, Messner red, Caudal purple.

  a  per-strain r:      one deletion's row in study A against its row in study B,
                        over the genes both report; one r per shared deletion.
  b  per-reporter r:    one gene's column across the shared deletions in A against
                        the same column in B; one r per shared reporter (test-retest).
  c  cross-batch:       one genotype's cells in batch i and batch j, each pseudobulked
                        against the WT cells of ITS OWN batch; r between the two
                        profiles. Null: batch i of genotype g against batch j of
                        another genotype.
  d  split-half:        one genotype's cells in one batch split in half, the batch's
                        WT cells split in half; half A vs WT half A, half B vs WT half
                        B; r between the two profiles. Nothing is shared.
  e  gene co-variation: within one panel, the correlation of two genes across that
                        panel's strains, for every gene pair; the upper triangles of
                        two panels compared by Spearman. Strains need not be shared,
                        which is what admits the Caudal isolates.
  f  the paper's own:   per genotype, the count of DE genes in each study (shown by
                        the paper) and the Spearman between the two profiles
                        (computed by the paper's script, not shown).

Run from the repo root:
    python experiments/028-knockout-expression/scripts/comparison_designs.py
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

C = {
    "kem": (PLOT_PALETTE[3], PLOT_PALETTE_FILL[3]),
    "sam": (PLOT_PALETTE[2], PLOT_PALETTE_FILL[2]),
    "nad": (PLOT_PALETTE[0], PLOT_PALETTE_FILL[0]),
    "mes": (PLOT_PALETTE[1], PLOT_PALETTE_FILL[1]),
    "cau": (PLOT_PALETTE[2], PLOT_PALETTE_FILL[2]),
    "wt": (PLOT_PALETTE[5], "#E6E6E6"),
}


def matrix(
    ax, x, y, w, h, key, title, rows="strains", cols="genes", hl_row=None, hl_col=None
):
    """A strains x genes matrix as a rectangle with faint gridlines and an optional
    highlighted row or column.
    """
    line, fill = C[key]
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=line, lw=0.8))
    n = 6
    for i in range(1, n):
        ax.plot([x, x + w], [y + h * i / n] * 2, color=line, lw=0.25, alpha=0.6)
        ax.plot([x + w * i / n] * 2, [y, y + h], color=line, lw=0.25, alpha=0.6)
    if hl_row is not None:
        ax.add_patch(
            Rectangle(
                (x, y + h * hl_row / n),
                w,
                h / n,
                facecolor=line,
                edgecolor="black",
                lw=0.5,
            )
        )
    if hl_col is not None:
        ax.add_patch(
            Rectangle(
                (x + w * hl_col / n, y),
                w / n,
                h,
                facecolor=line,
                edgecolor="black",
                lw=0.5,
            )
        )
    ax.text(x + w / 2, y + h + 0.03, title, ha="center", va="bottom", fontsize=6)
    ax.text(x + w / 2, y - 0.03, cols, ha="center", va="top", fontsize=5)
    if rows:
        ax.text(
            x - 0.02, y + h / 2, rows, ha="right", va="center", fontsize=5, rotation=90
        )


def cells(ax, x, y, w, h, key, n=18, title=None, seed=0):
    """A cloud of cells (small circles) inside a box."""
    import numpy as np

    line, fill = C[key]
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=line, lw=0.6))
    rng = np.random.default_rng(seed)
    px = x + 0.06 * w + rng.uniform(0, 0.88 * w, n)
    py = y + 0.1 * h + rng.uniform(0, 0.8 * h, n)
    ax.scatter(px, py, s=5, color=line, lw=0)
    if title:
        ax.text(x + w / 2, y + h + 0.03, title, ha="center", va="bottom", fontsize=6)
    return px, py


def arrow(ax, p, q, text=None, dy=0.04):
    ax.add_patch(
        FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.6, color="black")
    )
    if text:
        ax.text(
            (p[0] + q[0]) / 2,
            (p[1] + q[1]) / 2 + dy,
            text,
            ha="center",
            va="bottom",
            fontsize=5,
        )


def setup(ax, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    images = osp.join(os.environ["ASSET_IMAGES_DIR"], "028-knockout-expression")
    os.makedirs(images, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "axes.titlesize": 6,
        }
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(100))
    )

    # a. per-strain r
    ax = axes[0, 0]
    setup(ax, "per-strain r: one deletion in both studies")
    matrix(ax, 0.08, 0.34, 0.34, 0.42, "kem", "Kemmeren", hl_row=3)
    matrix(ax, 0.58, 0.34, 0.34, 0.42, "nad", "Nadal-Ribelles A", rows="", hl_row=3)
    yr = 0.34 + 0.42 * 3.5 / 6
    arrow(ax, (0.43, yr), (0.57, yr), "same deletion", dy=0.03)
    ax.text(
        0.5,
        0.15,
        "Pearson over the genes both report; one r per shared deletion\n"
        "reference: the same construction on Kemmeren vs Sameith",
        ha="center",
        va="center",
        fontsize=5,
    )

    # b. per-reporter r
    ax = axes[0, 1]
    setup(ax, "per-reporter r: one gene across the shared deletions")
    matrix(ax, 0.08, 0.34, 0.34, 0.42, "kem", "Kemmeren", hl_col=2)
    matrix(ax, 0.58, 0.34, 0.34, 0.42, "nad", "Nadal-Ribelles A", rows="", hl_col=2)
    arrow(ax, (0.43, 0.55), (0.57, 0.55), "same gene", dy=0.03)
    ax.text(
        0.5,
        0.15,
        "Pearson across the shared deletions; one r per reporter\n"
        "(test-retest reliability of that gene's response)",
        ha="center",
        va="center",
        fontsize=5,
    )

    # c. cross-batch replication
    ax = axes[0, 2]
    setup(ax, "cross-batch replication within Nadal-Ribelles")
    cells(ax, 0.06, 0.62, 0.24, 0.24, "nad", title="genotype g, batch i", seed=1)
    cells(ax, 0.68, 0.62, 0.24, 0.24, "nad", title="genotype g, batch j", seed=3)
    cells(ax, 0.06, 0.06, 0.24, 0.24, "wt", n=12, title="WT cells, batch i", seed=2)
    cells(ax, 0.68, 0.06, 0.24, 0.24, "wt", n=12, title="WT cells, batch j", seed=4)
    ax.text(0.18, 0.46, "log2 g / WT", ha="center", va="center", fontsize=5)
    ax.text(0.80, 0.46, "log2 g / WT", ha="center", va="center", fontsize=5)
    arrow(ax, (0.33, 0.52), (0.65, 0.52), "r between the two profiles", dy=0.04)
    ax.text(
        0.5,
        0.18,
        "null: batch j of\nanother genotype",
        ha="center",
        va="center",
        fontsize=5,
    )

    # d. split-half with the WT split
    ax = axes[1, 0]
    setup(ax, "split-half within one batch, reference split too")
    cells(
        ax, 0.04, 0.58, 0.40, 0.28, "nad", n=20, title="genotype g, one batch", seed=5
    )
    ax.plot([0.24, 0.24], [0.58, 0.86], color="black", lw=0.6, ls="--")
    ax.text(0.14, 0.55, "half A", ha="center", va="top", fontsize=5)
    ax.text(0.34, 0.55, "half B", ha="center", va="top", fontsize=5)
    cells(ax, 0.04, 0.14, 0.40, 0.26, "wt", n=14, title="WT cells, same batch", seed=6)
    ax.plot([0.24, 0.24], [0.14, 0.40], color="black", lw=0.6, ls="--")
    ax.text(0.14, 0.11, "WT half A", ha="center", va="top", fontsize=5)
    ax.text(0.34, 0.11, "WT half B", ha="center", va="top", fontsize=5)
    ax.text(
        0.72,
        0.70,
        "profile A = log2 (half A / WT half A)\nprofile B = log2 (half B / WT half B)\n"
        "r between profile A and profile B",
        ha="center",
        va="center",
        fontsize=5,
    )
    ax.text(
        0.72,
        0.30,
        "a shared reference would put its own\nsampling noise into both profiles\n"
        "(0.37 with a shared WT, 0.08 split)",
        ha="center",
        va="center",
        fontsize=5,
    )

    # e. gene co-variation
    ax = axes[1, 1]
    setup(ax, "gene co-variation: no shared strains needed")
    matrix(
        ax, 0.06, 0.58, 0.20, 0.26, "mes", "Messner", rows="deletions", cols="proteins"
    )
    matrix(ax, 0.06, 0.12, 0.20, 0.26, "cau", "Caudal", rows="isolates", cols="genes")
    arrow(ax, (0.28, 0.71), (0.40, 0.71))
    arrow(ax, (0.28, 0.25), (0.40, 0.25))
    matrix(ax, 0.42, 0.58, 0.22, 0.26, "mes", "gene x gene r", rows="", cols="genes")
    matrix(ax, 0.42, 0.12, 0.22, 0.26, "cau", "gene x gene r", rows="", cols="genes")
    arrow(ax, (0.66, 0.66), (0.74, 0.55))
    arrow(ax, (0.66, 0.30), (0.74, 0.41))
    ax.text(
        0.86,
        0.48,
        "Spearman\nbetween the\nupper triangles",
        ha="center",
        va="center",
        fontsize=5,
    )

    # f. the paper's own comparison
    ax = axes[1, 2]
    setup(ax, "the paper's own comparison (its Supp. Fig. 1i)")
    matrix(ax, 0.08, 0.46, 0.30, 0.34, "kem", "Kemmeren", hl_row=2)
    matrix(ax, 0.58, 0.46, 0.30, 0.34, "nad", "Nadal (stored logFC)", rows="", hl_row=2)
    yr = 0.46 + 0.34 * 2.5 / 6
    arrow(ax, (0.39, yr), (0.57, yr))
    ax.text(
        0.5,
        0.91,
        "profile vs profile: computed, not shown; median 0.013",
        ha="center",
        va="center",
        fontsize=5,
    )
    ax.text(
        0.23, 0.33, "genes at |M| > 1, p < 0.05", ha="center", va="center", fontsize=5
    )
    ax.text(
        0.73,
        0.33,
        "genes at |logFC| >= 1, p < 0.05",
        ha="center",
        va="center",
        fontsize=5,
    )
    arrow(ax, (0.39, 0.22), (0.57, 0.22))
    ax.text(
        0.5,
        0.10,
        "count vs count: the published panel, Spearman 0.23",
        ha="center",
        va="center",
        fontsize=5,
    )

    for ax in axes.flat:
        for s in ax.spines.values():
            s.set_visible(True)
    fig.subplots_adjust(
        left=0.03, right=0.99, bottom=0.03, top=0.92, wspace=0.12, hspace=0.3
    )
    for ax, letter in zip(axes.flat, "abcdef"):
        panel_label(ax, letter)
    stem = osp.join(images, "comparison_designs")
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"figure: {stem}.svg")


if __name__ == "__main__":
    main()
