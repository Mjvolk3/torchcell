# experiments/024-perturb-seq-costing/scripts/plot_sequencers.py
# [[experiments.024-perturb-seq-costing.scripts.plot_sequencers]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_sequencers
"""Sequencing capacity and price, and where a genome-scale screen lands on it.

Two panels:

(a) Output per flow cell, split into reads-per-lane and lanes, so the two ways a
    platform gets big are visible separately. Open bars are instruments the UIUC
    core does NOT have -- they are manufacturer specifications included only for
    comparison, and they carry no price.
(b) Price per million read pairs for what the core actually sells, with the
    screen's read requirement marked. This is the panel that decides which
    configuration to buy.

The gap the core's inventory leaves is the point of panel (a): between the MiSeq
i100 at 50M reads and the NovaSeq X 1.5B at 1.6B there is nothing, so a run is
either a QC-scale experiment or a production one with no middle option.

Output: $ASSET_IMAGES_DIR/024-perturb-seq-costing/sequencers.svg
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

import cost_model as CM
from figure_checks import assert_legible
import uiuc_core_data as UC
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")

INSTRUMENT_COLOR = {
    "MiSeq i100": PLOT_PALETTE[0],      # amber
    "NovaSeq X Plus": PLOT_PALETTE[1],  # brick
    "NovaSeq 6000": PLOT_PALETTE[2],    # lilac
    "NextSeq 2000": PLOT_PALETTE[3],    # wheat
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 6, "axes.linewidth": 0.5,
            "xtick.major.width": 0.5, "ytick.major.width": 0.5,
            "svg.fonttype": "none",
        }
    )


def box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def place_panel_letters(fig, axes, letters) -> None:
    """Panel letters anchored to each panel's measured full extent.

    Same rule as plot_economics.py: the letter goes above the title and left of
    the y-axis labels, so a crop at its corner takes only whitespace. These
    panels have long categorical y-tick labels whose width is data-dependent,
    which is exactly the case a fixed axes-fraction offset gets wrong.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0 - 0.010, bb.y1 + 0.020, letter, fontsize=8,
                 fontweight="bold", ha="left", va="bottom", zorder=20)


def panel_a(ax) -> None:
    fcs = sorted(UC.FLOW_CELLS, key=lambda f: f.total_read_pairs)
    y = np.arange(len(fcs))
    for i, f in enumerate(fcs):
        c = INSTRUMENT_COLOR[f.instrument]
        # Filled = we can buy it here; open = manufacturer spec, not at the core.
        ax.barh(i, f.total_read_pairs / 1e9, height=0.7,
                color=c if f.available_at_uiuc else "white",
                edgecolor=c if f.available_at_uiuc else c,
                linewidth=0.6, zorder=3)
        ax.text(f.total_read_pairs / 1e9 * 1.15, i,
                f"{f.lanes}×{f.read_pairs_per_lane/1e6:,.0f}M",
                va="center", fontsize=4.6, color="#444444")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{f.instrument}  {f.flow_cell}" for f in fcs], fontsize=5)
    ax.set_xscale("log")
    ax.set_xlim(3e-3, 200)
    ax.set_xlabel("Read pairs per flow cell (billions)")
    box(ax)
    ax.set_title("Capacity; labels are lanes × pairs per lane",
                 loc="left", fontsize=6)

    handles = [
        plt.Line2D([], [], marker="s", ls="", markerfacecolor="#888888",
                   markeredgecolor="#888888", markersize=4, label="at UIUC"),
        plt.Line2D([], [], marker="s", ls="", markerfacecolor="white",
                   markeredgecolor="#888888", markersize=4,
                   label="not at UIUC (spec only)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              handletextpad=0.3, borderaxespad=0.3)


def panel_b(ax) -> None:
    opts = sorted(UC.NOVASEQ_X + UC.MISEQ_I100,
                  key=lambda o: o.usd_per_million_read_pairs)
    y = np.arange(len(opts))
    colors = [
        INSTRUMENT_COLOR["NovaSeq X Plus"] if "Nova" in o.instrument
        else INSTRUMENT_COLOR["MiSeq i100"]
        for o in opts
    ]
    ax.barh(y, [o.usd_per_million_read_pairs for o in opts], height=0.7,
            color=colors, edgecolor="black", linewidth=0.4, zorder=3)
    for i, o in enumerate(opts):
        ax.text(o.usd_per_million_read_pairs * 1.06, i,
                f"${o.usd_per_lane:,.0f}/lane", va="center", fontsize=4.6,
                color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([o.label.replace(", per lane", "") for o in opts], fontsize=5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 700)
    ax.set_xlabel("Cost per million read pairs (USD)")
    box(ax)
    ax.set_title("Price, on-campus rate", loc="left", fontsize=6)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0))
    )
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout(pad=0.4, w_pad=2.0, rect=(0.010, 0.0, 1.0, 0.945))
    place_panel_letters(fig, axes, ["a", "b"])
    # Legibility gate; see figure_checks.py.
    assert_legible(fig, axes=list(axes))

    out = osp.join(OUT_DIR, "sequencers.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
