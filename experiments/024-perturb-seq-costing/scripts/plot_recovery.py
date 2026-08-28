# experiments/024-perturb-seq-costing/scripts/plot_recovery.py
# [[experiments.024-perturb-seq-costing.scripts.plot_recovery]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_recovery
"""Do you have to see a combination twice? Two answers, and only one lever.

Sec. 7.2 asks one question that has opposite answers depending on the estimand,
and the split is the whole content of the section:

    main effects        A cell carrying k guides contributes to k different main
                        effects, so no combination has to recur and multiplexing
                        simply divides the cell count by k. Panel (a).
    a NAMED pair        Estimating the cell state under one specific double
                        needs cells carrying exactly that pair, and the chance a
                        random k-subset of T targets contains it is k(k-1)/(T(T-1)).
                        At the budget that powers main effects, that expectation
                        is far below one at genome scale. Panel (b).

Panel (c) is the keybox: both k and T sit in the same expression, so on the
arithmetic they are interchangeable, and what separates them is that one is
capped and the other is not. Array construction holds k to two-to-four guides
(Sec. 7.1) while T is a free choice spanning a factor of thirty here, so
restricting the panel is what moves a named pair across the one-observation line
and raising the plex is not.

Two honesty constraints.

EVERY NUMBER COMES FROM scaling_analysis.py. The discrete points in (a) and (b)
are read from its results file, and the continuous sweep in (c) calls its
``recovery`` directly rather than restating the formula here. The one-observation
crossings marked in (c) are ``max_panel_for_one_observation``, which is in that
module for the same reason: a second copy of an expression is a second place for
it to drift.

THE BUDGET IS FIXED, AND THAT IS THE POINT. Nothing in (b) or (c) says a named
pair cannot be measured. It says that at the cell count which powers main
effects, it is not measured, and Eq. 3 in the section gives the far larger budget
that would be needed instead. The one-observation line is a threshold on
presence in the dataset, not on statistical power.

Output: $ASSET_IMAGES_DIR/024-perturb-seq-costing/recovery.svg
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from design_equation import box, place_panel_letters, style
from figure_checks import assert_legible
from scaling_analysis import max_panel_for_one_observation, recovery
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results")

# Five series, so the FIRST FIVE palette slots: orange, red, purple, yellow,
# blue. The figure carries two families rather than one, because the panels ask
# two different questions of the same expression -- (a) and (b) fix the library
# and sweep the plex, so their series is the library; (c) fixes the plex and
# sweeps the library, so its series is the plex. Each color means one thing
# throughout, and no color is reused across the two families.
#
# The reference grey is PLOT_PALETTE[5], not a typed "#666666". Same value, but
# a typed hex is a color that has left the palette: nothing updates it if the
# palette moves, and nothing flags a near-miss.
C_GENOME = PLOT_PALETTE[0]  # T = 6,000, the yeast genome
C_PANEL = PLOT_PALETTE[1]   # T = 200, the focused panel
C_K2 = PLOT_PALETTE[2]      # k = 2, the smallest plex that makes a pair
C_K4 = PLOT_PALETTE[3]      # k = 4, the array-construction ceiling
C_K8 = PLOT_PALETTE[4]      # k = 8, past what an array delivers
C_REF = PLOT_PALETTE[5]     # reference lines, annotations, non-series marks

N_PANEL = 200
N_GENOME = 6000

# Plex values scaling_analysis.py tabulates. k = 1 makes no pair at all, so the
# pair panels start at 2 rather than plotting a zero on a log axis.
PLEXES = [1, 2, 3, 5, 8]
PLEXES_PAIR = [2, 3, 5, 8]

# The plexes panel (c) draws, chosen to bracket the construction ceiling rather
# than to fill the palette: 2 is the smallest plex that makes a pair, 4 is the
# most an array reliably carries, and 8 is past it and drawn anyway so the
# reader can see how little the extra plex buys.
PLEXES_SWEEP = [(2, C_K2, "k = 2"),
                (4, C_K4, "k = 4, array ceiling"),
                (8, C_K8, "k = 8, past the ceiling")]


def load() -> dict[tuple[int, int], dict]:
    """Recovery points keyed by (T, k), as scaling_analysis.py wrote them."""
    with open(osp.join(RESULTS, "scaling_analysis.json")) as fh:
        rec = json.load(fh)["recovery"]
    return {(r["n_targets"], r["k"]): r for r in rec}


# =============================================================================
# (a) main effects
# =============================================================================


def panel_a(ax, rec) -> None:
    """Cells needed for main effects, against plex, at both library sizes.

    The curve is floor * T / k, so on a log axis it is a straight fall of one
    decade per decade of plex, and its message is the one the section opens
    with: no combination has to recur for a main effect, so every guide a cell
    carries is a guide whose effect that cell reports on.

    Both library sizes are drawn because the whole figure turns on the gap
    between them, and a reader who meets T only in (c) has no sense of scale
    for it.
    """
    for T, color, label in ((N_GENOME, C_GENOME, "genome, T = 6,000"),
                            (N_PANEL, C_PANEL, "panel, T = 200")):
        y = [rec[(T, k)]["cells_for_main_effects"] for k in PLEXES]
        ax.plot(PLEXES, y, lw=1.0, color=color, marker="o", ms=2.4,
                markeredgecolor="black", markeredgewidth=0.3, label=label)

    # The two numbers the section quotes, at the two ends of the genome curve.
    # Written with thousands separators rather than as powers of ten: Arial has
    # no superscript digits on this machine, and a tofu box on a print figure is
    # worse than four extra characters.
    for k, dx, dy, ha, va in ((1, 3, 3, "left", "bottom"), (8, -3, -3, "right", "top")):
        ax.annotate(f"{rec[(N_GENOME, k)]['cells_for_main_effects']:,.0f}",
                    (k, rec[(N_GENOME, k)]["cells_for_main_effects"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=4.5,
                    color=C_GENOME, ha=ha, va=va)

    ax.set_yscale("log")
    ax.set_xlim(0.4, 8.6)
    ax.set_ylim(1.5e3, 2.5e6)
    ax.set_xticks(PLEXES)
    ax.set_xlabel("Guides per cell, k")
    ax.set_ylabel("Cells for main effects")
    ax.legend(frameon=False, loc="lower left", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Plex divides the cell count", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (b) a named pair, at the same budget
# =============================================================================


def panel_b(ax, rec) -> None:
    """Expected observations of one NAMED pair, at the budget panel (a) buys.

    Same libraries, same plexes, same cells: only the estimand changes. The
    horizontal rule at one observation is what makes the panel readable, because
    the quantity is an expectation and the reader's question is whether it
    clears the threshold of appearing at all. The genome curve does not come
    within a decade of it at any constructible plex, which is the section's
    answer.
    """
    for T, color, label in ((N_GENOME, C_GENOME, "genome, T = 6,000"),
                            (N_PANEL, C_PANEL, "panel, T = 200")):
        y = [rec[(T, k)]["expected_repeats_per_pair"] for k in PLEXES_PAIR]
        ax.plot(PLEXES_PAIR, y, lw=1.0, color=color, marker="o", ms=2.4,
                markeredgecolor="black", markeredgewidth=0.3, label=label)

    # zorder below the series: a reference rule is the backdrop a curve is read
    # against, so where the two cross it is the curve that has to survive. Drawn
    # after the series, an axhline shares their default zorder of 2 and wins the
    # tie, which put a grey dashed break straight through the panel curve at the
    # one place the panel is about -- where it crosses one observation.
    ax.axhline(1.0, color=C_REF, lw=0.6, ls="--", zorder=1)
    # Right end and above the rule: the panel curve passes through 1.0 on the
    # left half, so a label anchored there would sit on the data it explains.
    ax.annotate("seen once", (8.5, 1.0), xytext=(0, 2),
                textcoords="offset points", fontsize=4.5, color=C_REF,
                ha="right", va="bottom")

    for T, k, color, dx, dy, ha, va in (
        (N_GENOME, 2, C_GENOME, 3, -1, "left", "top"),
        (N_PANEL, 8, C_PANEL, -3, 2, "right", "bottom"),
    ):
        ax.annotate(f"{rec[(T, k)]['expected_repeats_per_pair']:.2g}",
                    (k, rec[(T, k)]["expected_repeats_per_pair"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=4.5,
                    color=color, ha=ha, va=va)

    ax.set_yscale("log")
    ax.set_xlim(1.4, 8.6)
    ax.set_ylim(6e-3, 30)
    ax.set_xticks(PLEXES_PAIR)
    ax.set_xlabel("Guides per cell, k")
    ax.set_ylabel("Observations per named pair")
    ax.legend(frameon=False, loc="upper left", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("A named pair is usually absent", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (c) which lever moves it
# =============================================================================


def panel_c(ax) -> None:
    """The same expectation against panel size, at fixed plex.

    Panels (a) and (b) sweep k; this sweeps T, and the two sweeps are what make
    the keybox's claim visible rather than asserted. Over the range a design can
    actually choose, T carries the curve across the one-observation line and off
    the bottom of the axis, while the whole constructible span of k -- 2 to 4 --
    moves the crossing from a 101-gene panel to a 301-gene one.

    The sweep calls scaling_analysis.recovery for every point rather than
    inlining floor * T / k times k(k-1)/(T(T-1)), so the curve and the table in
    the section are the same arithmetic by construction.
    """
    targets = np.unique(np.round(np.logspace(2, np.log10(6000), 200)).astype(int))
    for k, color, label in PLEXES_SWEEP:
        y = [recovery(int(T), k).expected_repeats_per_pair for T in targets]
        ax.plot(targets, y, lw=1.0, color=color, label=label)

    ax.axhline(1.0, color=C_REF, lw=0.6, ls="--", zorder=1)
    ax.annotate("seen once", (5800, 1.0), xytext=(0, 2),
                textcoords="offset points", fontsize=4.5, color=C_REF,
                ha="right", va="bottom")

    # Where each plex crosses the line, back-solved by scaling_analysis.py. The
    # three crossings are the panel's actual content: they span a factor of
    # seven while the axis spans a factor of sixty, and the constructible two of
    # them span a factor of three.
    for (k, color, _), dy, va in zip(PLEXES_SWEEP, (5, -5, 5), ("bottom", "top", "bottom")):
        t_cross = max_panel_for_one_observation(k)
        ax.plot([t_cross], [1.0], marker="o", ms=2.8, color=color,
                markeredgecolor="black", markeredgewidth=0.3, zorder=5)
        ax.annotate(f"{t_cross:,}", (t_cross, 1.0), xytext=(1, dy),
                    textcoords="offset points", fontsize=4.5, color=color,
                    ha="left", va=va)

    # The two library sizes panels (a) and (b) are evaluated at, drawn as rules
    # on the x axis rather than as markers on a curve: they are properties of
    # the panel size alone, and a marker sitting on the k = 2 line would read as
    # though it belonged to that plex.
    for x, lab, ha in ((N_PANEL, "panel", "left"), (N_GENOME, "genome", "right")):
        ax.axvline(x, color=C_REF, lw=0.5, ls=":", zorder=1)
        ax.annotate(lab, (x, 7e-3), xytext=(2 if ha == "left" else -2, 0),
                    textcoords="offset points", fontsize=4.5, color=C_REF,
                    ha=ha, va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Left edge below the smallest panel drawn, so the k = 2 crossing marker at
    # T = 101 sits clear of the spine rather than half under it.
    ax.set_xlim(85, 7000)
    ax.set_ylim(6e-3, 30)
    ax.set_xlabel("Genes in the panel, T")
    ax.set_ylabel("Observations per named pair")
    ax.legend(frameon=False, loc="upper right", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Panel size is the lever, not plex", loc="left", fontsize=6)
    box(ax)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    rec = load()

    fig, axes = plt.subplots(
        1, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0))
    )
    flat = list(axes.flat)
    panel_a(flat[0], rec)
    panel_b(flat[1], rec)
    panel_c(flat[2])
    fig.tight_layout(pad=0.4, w_pad=2.6, rect=(0.012, 0.0, 1.0, 0.935))
    place_panel_letters(fig, flat, ["a", "b", "c"])

    assert_legible(fig, axes=flat)

    out = osp.join(OUT_DIR, "recovery.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    for k in (2, 4, 8):
        t = max_panel_for_one_observation(k)
        print(f"  k = {k}: a named pair is seen once up to T = {t:,}")


if __name__ == "__main__":
    main()
