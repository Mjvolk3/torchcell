# experiments/024-perturb-seq-costing/scripts/plot_environments.py
# [[experiments.024-perturb-seq-costing.scripts.plot_environments]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_environments
"""Environments as the other axis: linear, nearly free in split-pool, not in droplet.

Sec. 7.3 makes three claims about the environment axis, and each panel draws
one of them:

    (a) Environments multiply cell demand LINEARLY -- 600,000 cells per
        condition, so 4 environments are 2.4 million cells and 12 are 7.2
        million -- where named gene pairs grow QUADRATICALLY at Yao et al.'s
        400 cells per pair.
    (b) A pooled single-condition screen uses ONE round-1 well for sample
        identity, so 95 of the 96 round-1 barcodes are idle. Growing 96
        conditions in a deep-well plate consumes capacity that is otherwise
        wasted.
    (c) The same multi-environment screen on droplet needs one channel per
        condition, and channels dominate that platform's cost; split-pool
        pools every condition into shared runs and shared lanes.

Two honesty constraints.

THE INTERCEPTS IN (a) ARE NOT THE CLAIM. One environment costs 600,000 cells
and one pair costs 400, so across the whole drawn range the linear line sits
ABOVE the quadratic curve; the two only cross near 3,000, off-panel. What the
panel argues is the GROWTH SHAPE -- slope 1 against slope 2 on log axes -- and
what an environment buys is also not what a pair buys: one environment adds a
full 6,000-gene condition profile, one pair adds a single combination.

THE COST CURVES IN (c) INHERIT THE SEC. 5 MODEL'S ASSUMPTIONS. Every dollar
comes from ``scaling_analysis.environment_cost``, which composes
``cost_model.budget_for`` along the environment axis: the split-pool platform
is SPLiT-seq + rRNA depletion, whose depth is a transfer from an E. coli
result, and the droplet channel rate is a UIUC list price. What does NOT
depend on any price is the structure -- a droplet channel holds one condition
while the round-1 plate holds 96 -- and that structure is the panel's point.

Output: $ASSET_IMAGES_DIR/024-perturb-seq-costing/environments.svg
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from figure_checks import assert_legible
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results")

# One color per SERIES, held across panels. The two platform colors are PINNED:
# plot_economics established SPLiT-seq + rRNA depletion on slot 1 and the 10x
# Chromium X on slot 2, and a platform must keep its color across the document.
# The two non-platform series then take the first free slots. Environments get
# slot 0; named pairs take slot 4 rather than slot 3, because slots 0 and 3
# (amber and wheat) are the pair plot_economics documented as inseparable at
# line weight, and here they would share panel (a).
C_ENV = PLOT_PALETTE[0]    # the environment axis: (a) line, (b) filled wells
C_PAIRS = PLOT_PALETTE[4]  # named gene pairs, (a)
C_SPLIT = PLOT_PALETTE[1]  # SPLiT-seq + rRNA depletion, (c), as in plot_economics
C_DROP = PLOT_PALETTE[2]   # 10x Chromium X, (c), as in plot_economics
C_PI = PLOT_PALETTE[3]     # 10x + preindexing, (c), as in plot_economics
C_REF = PLOT_PALETTE[5]    # reference ink: grid lines on empty wells


def style() -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
        "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "svg.fonttype": "none",
    })


def box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def place_panel_letters(fig, axes, letters) -> None:
    """Bold panel letters outside each panel's full extent. See plot_economics."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(bb.x0 - 0.010, bb.y1 + 0.012, letter, fontsize=8,
                 fontweight="bold", ha="left", va="bottom", zorder=20)


def load() -> dict:
    with open(osp.join(RESULTS, "scaling_analysis.json")) as fh:
        return json.load(fh)


# =============================================================================
# (a) linear against quadratic
# =============================================================================


def panel_a(ax, data) -> None:
    """Cells against axis size: environments (slope 1) and named pairs (slope 2).

    Both curves are drawn from the constants scaling_analysis writes -- 600,000
    cells per condition and 400 cells per pair -- and the three annotated
    anchors are read back from its tables, not recomputed here: the 4- and
    12-environment cell counts from the "environments" key, and the all-pairs
    cost of the 200-target panel from the recovery table at k = 2, which is the
    same 400 x C(200, 2) the curve passes through.
    """
    per_cond = data["environments"]["1"]
    cpp = data["constants"]["cells_per_pair"]
    pairs_200 = next(
        r for r in data["recovery"] if r["n_targets"] == 200 and r["k"] == 2
    )["cells_for_all_pairs"]

    s = np.arange(1, 401, dtype=float)
    ax.plot(s, per_cond * s, lw=1.0, color=C_ENV,
            label=f"environments, {per_cond:,.0f} cells each")
    sp = np.arange(2, 401, dtype=float)
    ax.plot(sp, cpp * sp * (sp - 1) / 2, lw=1.0, color=C_PAIRS,
            label=f"named gene pairs, {cpp} cells per pair")

    for e in (4, 12):
        cells = data["environments"][str(e)]
        ax.plot([e], [cells], marker="o", ms=2.6, color=C_ENV,
                markeredgecolor="black", markeredgewidth=0.3, zorder=5)
    # The three anchors sit on opposite sides of their curves so no label box
    # can contain a curve segment. The 4-environment label hangs BELOW-RIGHT,
    # where a slope-1 line leaves open space (hanging left of x = 4 left the
    # frame on an axis that starts at 1). The 12-environment label sits
    # ABOVE-LEFT, where the line has not yet arrived -- below-right put it at
    # the same height as the pairs anchor, since both land near 7-8 million.
    ax.annotate(f"4 environments:\n{data['environments']['4'] / 1e6:.1f} million cells",
                (4, data["environments"]["4"]), xytext=(4, -2),
                textcoords="offset points", fontsize=4.5, ha="left",
                va="top", color=C_ENV)
    ax.annotate(f"12: {data['environments']['12'] / 1e6:.1f} million",
                (12, data["environments"]["12"]), xytext=(-3, 2),
                textcoords="offset points", fontsize=4.5, ha="right",
                va="bottom", color=C_ENV)
    ax.plot([200], [pairs_200], marker="o", ms=2.6, color=C_PAIRS,
            markeredgecolor="black", markeredgewidth=0.3, zorder=5)
    # ABOVE-LEFT and kept narrow: the wedge between the two curves is the one
    # region neither crosses, but it closes leftward -- a line wider than about
    # twelve characters reaches back far enough for the environment line to run
    # through its top-left corner, which is exactly what the first draft shipped.
    ax.annotate(f"200 targets:\n{pairs_200 / 1e6:.1f} million",
                (200, pairs_200), xytext=(-4, 2), textcoords="offset points",
                fontsize=4.5, ha="right", va="bottom", color=C_PAIRS)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 400)
    ax.set_ylim(2e2, 4e8)
    ax.set_xlabel("Size of the axis: environments, or targets")
    ax.set_ylabel("Cells required")
    ax.legend(frameon=False, loc="lower right", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Environments are the linear axis", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (b) the idle round-1 plate
# =============================================================================


def _plate(ax, x0: int, filled: int) -> None:
    """One 96-well round-1 plate as a 12 x 8 grid of unit squares.

    ``filled`` wells carry a condition and are drawn in the environment color;
    the rest are the capacity split-pool has already paid for whether or not a
    sample occupies it. Filling proceeds from the top-left, the way a plate is
    loaded, so the one-condition plate shows a single occupied corner well.
    """
    for i in range(96):
        row, col = i // 12, i % 12
        used = i < filled
        ax.add_patch(Rectangle(
            (x0 + col, 7 - row), 1.0, 1.0,
            facecolor=C_ENV if used else "white",
            edgecolor="black" if used else C_REF, lw=0.35, zorder=3 if used else 2,
        ))


def panel_b(ax) -> None:
    """Round-1 barcode capacity used, at 1 condition and at 96.

    Split-pool spends the round-1 96-well plate on sample identity. A pooled
    single-condition screen is one sample, so it uses one well and idles the
    other 95 -- the plate is bought, stamped, and mostly unread. The same
    plate fully loaded is 96 conditions in one run, which is the capacity
    Sec. 7.3 argues a deep-well plate of environments should consume. Drawn as
    two capacity grids rather than a chart because the quantity is literal:
    each square IS a round-1 well.
    """
    _plate(ax, 0, filled=1)
    _plate(ax, 15, filled=96)
    ax.text(6, -0.9, "1 condition\n1 of 96 wells used", fontsize=5,
            ha="center", va="top")
    ax.text(21, -0.9, "96 conditions\ncapacity fully used", fontsize=5,
            ha="center", va="top")
    ax.set_xlim(-0.7, 27.7)
    ax.set_ylim(-3.6, 9.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A single condition idles the round-1 plate",
                 loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (c) what the platforms charge for the same axis
# =============================================================================


def panel_c(ax, data) -> None:
    """Recurring cost against environments, split-pool and droplet.

    Both curves come straight from the "environment_costs" table
    scaling_analysis writes with the Sec. 5 cost model. Both are linear -- runs,
    sublibraries, channels and lanes all scale with cells -- so the story is
    the slope: droplet pays one channel per condition and channels dominate
    that platform's cost, where split-pool adds a condition by occupying a
    round-1 well it has already paid for and sharing runs and lanes. The
    marginal cost per added environment is annotated from the same file rather
    than recomputed.
    """
    rows = data["environment_costs"]
    summ = data["environment_cost_summary"]
    e = np.array([r["n_env"] for r in rows], dtype=float)
    sp = np.array([r["splitpool_usd"] for r in rows])
    dr = np.array([r["droplet_usd"] for r in rows])
    pi = np.array([r["droplet_preindexed_usd"] for r in rows])

    ax.plot(e, dr, lw=1.0, color=C_DROP,
            label="10x Chromium X, one channel per condition")
    ax.plot(e, pi, lw=1.0, color=C_PI, ls=(0, (4, 1.5)),
            label="10x + preindexing, still one channel per condition")
    ax.plot(e, sp, lw=1.0, color=C_SPLIT,
            label="SPLiT-seq + rRNA depletion, shared runs")

    m_dr = summ["marginal_usd_per_env_droplet"]
    m_pi = summ["marginal_usd_per_env_droplet_preindexed"]
    m_sp = summ["marginal_usd_per_env_splitpool"]
    i60 = int(np.argmin(np.abs(e - 60)))
    ax.annotate(f"${m_dr:,.0f} per added\nenvironment", (60, dr[i60]),
                xytext=(-4, 3), textcoords="offset points", fontsize=4.5,
                ha="right", va="bottom", color=C_DROP)
    # The two cheap curves run close together at the bottom of the panel, so
    # their labels go on opposite sides of the pair rather than both above.
    ax.annotate(f"${m_pi:,.0f}", (84, pi[int(np.argmin(np.abs(e - 84)))]),
                xytext=(0, 3), textcoords="offset points", fontsize=4.5,
                ha="center", va="bottom", color=C_PI)
    ax.annotate(f"${m_sp:,.0f} per added environment", (60, sp[i60]),
                xytext=(0, -4), textcoords="offset points", fontsize=4.5,
                ha="center", va="top", color=C_SPLIT)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, dr[-1] * 1.08)
    ax.set_xticks([1, 12, 24, 48, 96])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "$0" if v == 0 else f"${v / 1e6:g}M")
    )
    ax.set_xlabel("Environments")
    ax.set_ylabel("Recurring cost")
    ax.legend(frameon=False, loc="upper left", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Droplet pays per condition", loc="left", fontsize=6)
    box(ax)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    data = load()

    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(58.0))
    )
    flat = list(axes.flat)
    panel_a(flat[0], data)
    panel_b(flat[1])
    panel_c(flat[2], data)
    fig.tight_layout(pad=0.4, w_pad=2.6, rect=(0.012, 0.0, 1.0, 0.93))
    place_panel_letters(fig, flat, ["a", "b", "c"])

    assert_legible(fig, axes=flat)

    out = osp.join(OUT_DIR, "environments.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    summ = data["environment_cost_summary"]
    print(f"\nper added environment: split-pool "
          f"${summ['marginal_usd_per_env_splitpool']:,.0f}, droplet "
          f"${summ['marginal_usd_per_env_droplet']:,.0f} "
          f"({summ['droplet_channels_per_condition']} channels at "
          f"${summ['droplet_channel_usd']:,.0f})")


if __name__ == "__main__":
    main()
