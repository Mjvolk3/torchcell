# experiments/019-perturb-seq-costing/scripts/plot_economics.py
# [[experiments.019-perturb-seq-costing.scripts.plot_economics]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/plot_economics
"""Three panels on the economics of a genome-scale yeast CRISPRi perturb-seq.

(a) Cells needed per perturbation, by how deep the method reads each cell. Two
    independent constraints; the taller bar is the one that binds.
(b) Where the money goes at genome scale. The story is that the dominant term
    moves between platforms: sequencing for split-pool, library prep for droplet.
(c) What multiplexing buys. Main effects get k times cheaper; named pairwise
    interactions do not become reachable genome-wide at any k.

Output: $ASSET_IMAGES_DIR/019-perturb-seq-costing/economics.svg
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter

import cost_model as CM
import method_data as MD
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-perturb-seq-costing")


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "hatch.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )


def box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def place_panel_letters(fig, axes, letters) -> None:
    """Bold panel letters at the true top-left of each panel's full extent.

    House rule: the letter must sit in clear whitespace -- above the title and
    left of the y-axis label -- so a crop taken at the letter's corner slices
    only background. An axes-fraction offset cannot guarantee that, because the
    y-label's width depends on the tick text and changes with the data; a letter
    placed at a fixed -0.22 will sometimes clear it and sometimes graze it.

    So measure instead. ``get_tightbbox`` returns the panel's extent INCLUDING
    its y-label, tick labels and title; anchoring to the top-left of that box and
    stepping a little further out puts the letter outside everything the panel
    draws, whatever the numbers happen to be. This is also what creates visible
    separation between adjacent panels.
    """
    fig.canvas.draw()  # tightbbox needs a renderer
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(
            bb.x0 - 0.012, bb.y1 + 0.020, letter,
            fontsize=8, fontweight="bold", ha="left", va="bottom", zorder=20,
        )


def panel_a(ax) -> None:
    """Cells per perturbation: biological floor vs depth requirement."""
    depths = [
        ("SPLiT-seq\npublished", 410),
        ("SPLiT-seq\n+depl.", 861),
        ("10x v2", 894),
        ("10x v3", 2000),
    ]
    target = CM.PSEUDOBULK_TIERS["standard (Brettner's own 500-cell heuristic)"]
    x = np.arange(len(depths))
    w = 0.38

    depth_req = [target / d for _, d in depths]
    floor = [CM.CELLS_FLOOR] * len(depths)

    ax.bar(
        x - w / 2,
        depth_req,
        w,
        color=PLOT_PALETTE[0],
        edgecolor="black",
        linewidth=0.5,
        # Plain text, not mathtext: a 5 pt mathtext superscript does not survive
        # Arial + svg.fonttype:none through rsvg-convert and renders as garbage
        # in the PDF, even though it looks fine in a rasterised preview.
        label="depth: 200,000 pseudobulk UMIs",
    )
    ax.bar(
        x + w / 2,
        floor,
        w,
        color=PLOT_PALETTE[4],
        edgecolor="black",
        linewidth=0.5,
        label="biology: 100-cell floor",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in depths], fontsize=5)
    ax.set_ylabel("Cells per perturbation")
    ax.set_yscale("log")
    # Headroom above the tallest bar so the legend never sits on the data or on
    # the y-axis tick labels.
    ax.set_ylim(30, 4000)
    ax.legend(
        frameon=False,
        loc="upper center",
        fontsize=5,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    ax.set_title("Two constraints; the taller one binds", loc="left", fontsize=6)
    box(ax)


def panel_b(ax) -> None:
    """Cost stack at 250 cells/gene, 6,000 genes, one environment."""
    design = CM.ScreenDesign(cells_per_gene=250)
    budgets = [CM.budget_for(design, p) for p in CM.PLATFORMS]
    # The asterisk is the same marker the budget tables use for a projected row,
    # and it has to be IN the panel: a screenshotted figure loses the caption.
    labels = ["SPLiT-seq\npublished", "SPLiT-seq\n+rRNA depl.", "10x\nChromium X",
              "10x + scifi\npreindex.*"]
    x = np.arange(len(budgets))
    w = 0.55

    protocol = np.array([b.protocol_usd for b in budgets]) / 1e3
    sublib = np.array([b.sublibrary_usd for b in budgets]) / 1e3
    seq = np.array([b.sequencing_usd for b in budgets]) / 1e3

    ax.bar(x, protocol, w, color=PLOT_PALETTE[1], edgecolor="black", lw=0.5,
           label="barcoding / library prep")
    ax.bar(x, sublib, w, bottom=protocol, color=PLOT_PALETTE[3], edgecolor="black",
           lw=0.5, label="sublibrary prep")
    ax.bar(x, seq, w, bottom=protocol + sublib, color=PLOT_PALETTE[2],
           edgecolor="black", lw=0.5, label="sequencing (NovaSeq X 25B)")

    for xi, b in zip(x, budgets):
        ax.text(xi, b.recurring_usd / 1e3 + 12, f"${b.recurring_usd/1e3:.0f}k",
                ha="center", fontsize=5.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Recurring cost per screen ($ thousands)")
    # Headroom, not a shorter label. The legend is three rows deep and its widest
    # entry runs most of the panel width, so at ylim=430 the "sequencing" row
    # crossed the 10x bar, whose $364k top sits at 87% of the axis. Ending the
    # axis at 560 drops that bar to ~65% and leaves the legend a clear band. The
    # tick labels stay at round hundreds either way.
    ax.set_ylim(0, 560)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, handletextpad=0.4,
              labelspacing=0.35, borderaxespad=0.3)
    ax.set_title("6,000 genes, 250 cells per target gene", loc="left", fontsize=6)
    box(ax)


def panel_c(ax) -> None:
    """Multiplexing: main effects scale, named interactions do not."""
    plexes = list(range(1, 11))
    base = 250
    cells = [CM.cells_for_main_effects_kplex(base, 6000, k) for k in plexes]
    pair_focus = [
        CM.cells_per_named_pair(n, 200, k) for n, k in zip(cells, plexes)
    ]

    ax.plot(plexes, np.array(cells) / 1e6, marker="o", ms=3, lw=0.8,
            color=PLOT_PALETTE[0], markeredgecolor="black", markeredgewidth=0.4,
            label="cells for main effects (left)")
    # "plex k" spelled out -- the parenthetical was colliding with the italic k.
    ax.set_xlabel("Guides per cell, $k$")
    ax.set_ylabel("Total cells needed (millions)")
    ax.set_ylim(0, 1.7)
    ax.set_xticks(plexes)
    ax.set_xticklabels([str(k) for k in plexes])
    ax.set_xlim(0.4, 10.6)
    box(ax)

    ax2 = ax.twinx()
    ax2.plot(plexes, pair_focus, marker="s", ms=3, lw=0.8, ls="--",
             color=PLOT_PALETTE[4], markeredgecolor="black", markeredgewidth=0.4,
             label="cells per gene pair, 200-gene panel (right)")
    # The 100-cell floor is the minimum cells a perturbation needs to be callable
    # at all; a pair below it is not measurable however many cells the screen has.
    ax2.axhline(CM.CELLS_FLOOR, color="#666666", lw=0.5, ls=":")
    ax2.text(10.5, CM.CELLS_FLOOR * 1.12, "100-cell floor", fontsize=5,
             color="#666666", ha="right")
    ax2.set_ylabel("Cells per gene pair")
    ax2.set_ylim(0, 400)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=5,
              handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Main effects get cheaper with plex; pairs do not",
                 loc="left", fontsize=6)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    # Reserve headroom at the top and a strip at the left so the measured letter
    # positions land inside the canvas rather than being clipped.
    fig.tight_layout(pad=0.4, w_pad=2.6, rect=(0.012, 0.0, 1.0, 0.945))
    place_panel_letters(fig, axes, ["a", "b", "c"])
    out = osp.join(OUT_DIR, "economics.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
