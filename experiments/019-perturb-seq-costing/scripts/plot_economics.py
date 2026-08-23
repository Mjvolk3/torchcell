# experiments/019-perturb-seq-costing/scripts/plot_economics.py
# [[experiments.019-perturb-seq-costing.scripts.plot_economics]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/plot_economics
"""Six panels on the economics of a genome-scale yeast CRISPRi perturb-seq.

The figure exists to support one decision -- which platform to run -- so it is
organized as the four questions that decision actually turns on, plus the two
that bound it:

(a) How many cells does a perturbation need? Two independent constraints, and
    the taller binds. This is where depth stops mattering.
(b) Where does the money go at genome scale? The dominant term MOVES between
    platforms, which is why no single optimization helps all of them.
(c) How does that scale with the precision target? Whether an ordering survives
    a change of ambition is a different question from the ordering at one point.
(d) What happens to a purchased read? The reagent comparison is not the cost
    comparison, and the gap between them lives entirely in this panel.
(e) What is the cost sensitive to? Each platform has exactly one unmeasured
    number that dominates it, and they are not equally worth measuring.
(f) What does multiplexing buy? Main effects get k times cheaper; named pairwise
    interactions do not become reachable genome-wide at any k.

Three platforms run through all six -- split-pool (Brettner), droplet (10x, as
Jariani adapted it for yeast), and preindexed droplet (scifi) -- plus split-pool
with rRNA depletion, which is a fourth column rather than a variant because its
cost structure genuinely differs.

COLOR IS PLATFORM, EVERYWHERE. One key for the whole figure. Where a panel needs
a second categorical dimension (the cost categories in b) it uses hatching, per
the repo rule that a second dimension is disambiguated with pattern rather than
with more color.

Output: $ASSET_IMAGES_DIR/019-perturb-seq-costing/economics.svg
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import cost_model as CM
from figure_checks import assert_legible
from torchcell.utils import (PANEL_WIDTHS_MM, PLOT_PALETTE, PLOT_PALETTE_FILL,
                            mm_to_in, savefig_true_size_svg)

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-perturb-seq-costing")

# One color per platform, in CM.PLATFORMS order, taken from the front of the
# repo palette as the ordering rule requires. Every panel that distinguishes
# platforms uses this and only this, so a reader learns the key once.
# Slots 0, 1, 2, 4 -- NOT 0..3. The repo rule is "a series of N takes the first
# N", which would put SPLiT-seq published in orange (#D79B00) and the scifi row
# in yellow (#D6B656). At 0.9 pt on a 57 mm panel those two are not separable,
# and this figure asks one key to hold across six panels, so it has to hold in
# the worst of them. Skipping to blue is the documented deviation; it costs one
# warm primary and buys a key a reader can actually use.
_SLOTS = (0, 1, 2, 4)
PLATFORM_COLOR = {p.name: PLOT_PALETTE[i] for p, i in zip(CM.PLATFORMS, _SLOTS)}
# The matched pale companion of each platform color, used only as the lighter
# member of a two-level bar in panel (b) -- which is the one job the standard
# reserves PLOT_PALETTE_FILL for. Never as a primary plot color.
PLATFORM_FILL = {p.name: PLOT_PALETTE_FILL[i] for p, i in zip(CM.PLATFORMS, _SLOTS)}

# Non-series ink, from the palette rather than typed: reference lines, grey
# annotations and the near-black used for a value printed on a bar.
C_REF = PLOT_PALETTE[5]
C_INK = "#333333"

# Legend swatches for a TREATMENT rather than a platform -- the reagents and
# sequencing key in panel (b), and the filled/open marker key in (e).
# Deliberately OUTSIDE the palette and deliberately neutral: drawn in any
# palette color they would read as a fifth platform, which is the one thing the
# figure's single color key must not allow. They hold the same dark/pale
# relation the bars do, so the key still teaches the encoding.
KEY_DARK = "#8C8C8C"
KEY_PALE = "#E8E8E8"

# Two-line labels for a 57 mm panel. The asterisk on the projected row is the
# same marker the budget tables use, and it has to be IN the panel: a
# screenshotted figure loses the caption.
PLATFORM_LABEL = {
    "SPLiT-seq (Brettner, as published)": "SPLiT-seq\npublished",
    "SPLiT-seq + rRNA depletion": "SPLiT-seq\n+rRNA depl.",
    "10x Chromium X (GEM-X 3')": "10x\nChromium X",
    "10x + scifi preindexing (projected)": "10x + scifi\npreindex.*",
}
PLATFORM_SHORT = {k: v.replace("\n", " ") for k, v in PLATFORM_LABEL.items()}

# The line style that means "projected". Coarse on purpose: at matplotlib's
# default "--" and 0.9 pt, a dashed line laid exactly over a solid one of another
# color reads as a single solid line, because the gaps are narrower than the line
# is wide. Panel (d) is where that matters -- the 10x and scifi curves coincide
# at every stage, so the only thing telling a reader both are there is purple
# showing through blue.
PROJECTED_LS = (0, (2.0, 3.5))

# The design every panel that needs one is priced at. Named once so panels
# cannot drift apart, which they did when 250 was typed into three of them.
BASE_CELLS_PER_GENE = 250


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
            bb.x0 - 0.010, bb.y1 + 0.012, letter,
            fontsize=8, fontweight="bold", ha="left", va="bottom", zorder=20,
        )


# =============================================================================
# (a) two constraints
# =============================================================================


def panel_a(ax) -> None:
    """Cells per perturbation: biological floor against depth requirement.

    Indexed by PLATFORM rather than by an abstract depth ladder, which is what
    the previous version did. The change matters for one reason: it makes the
    preindexed column's identity with the 10x column visible instead of hiding
    it. Preindexing buys cells per dollar, not UMIs per cell, so it cannot move
    a bar in this panel -- and the reader should be able to see that rather than
    wonder why the platform is missing.
    """
    target = CM.PSEUDOBULK_TIERS["standard (Brettner's own 500-cell heuristic)"]
    x = np.arange(len(CM.PLATFORMS))
    w = 0.38

    depth_req = [target / p.mrna_umis_per_cell for p in CM.PLATFORMS]
    floor = [CM.CELLS_FLOOR] * len(CM.PLATFORMS)

    # ONE bar per platform, and the floor drawn ONCE as a rule.
    #
    # It used to be a paired bar chart: a depth bar and a hatched floor bar for
    # every platform. The floor is 100 cells for all four, because it is a
    # property of yeast biology and not of any chemistry, so that drew the same
    # number four times and invited the reader to look for a difference between
    # the hatched bars that does not exist. A constant is a reference line.
    #
    # The reading also gets sharper: a bar above the rule is depth-limited, a
    # bar at it is floor-limited, and the two platforms whose bars sit exactly on
    # the rule are the ones where extra depth has stopped buying anything.
    ax.bar(x, depth_req, w * 1.5,
           color=[PLATFORM_COLOR[p.name] for p in CM.PLATFORMS],
           edgecolor="black", linewidth=0.5)
    ax.axhline(CM.CELLS_FLOOR, color="black", lw=0.7, ls=":")
    ax.annotate("100-cell biological floor", (len(CM.PLATFORMS) - 0.55,
                                              CM.CELLS_FLOOR * 1.12),
                fontsize=4.5, ha="right", va="bottom")
    ax.annotate("bars: cells to reach 200,000 pseudobulk UMIs",
                (-0.45, 3200), fontsize=4.5, ha="left", va="top",
                color=C_REF)

    ax.set_xticks(x)
    ax.set_xticklabels([PLATFORM_LABEL[p.name] for p in CM.PLATFORMS], fontsize=4.5)
    ax.set_ylabel("Cells per perturbation")
    ax.set_yscale("log")
    ax.set_ylim(30, 4000)
    # "Higher", not "taller": one constraint is a bar and the other is now
    # a rule, so they are compared by height on the axis rather than as two
    # bars side by side.
    ax.set_title("Two constraints; the higher binds", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (b) cost composition
# =============================================================================


def panel_b(ax) -> None:
    """Cost stack at the base design: reagents against sequencing.

    TWO segments, separated by tone rather than by hatch, and both changes were
    forced by the same measurement. The stack used to carry three categories
    distinguished by ``xxx`` and ``...`` hatching, and neither worked at this
    size for a reason arithmetic rather than aesthetic: sublibrary prep is
    $2,970 of a $156,670 screen, so its band is under 2% of the bar -- about
    four points tall. No pattern is legible in four points, and a pattern that
    cannot be read is worse than no category at all, because it still costs the
    reader an entry in the key. Folding it into reagents also makes this panel
    agree with \\cref{tab:budgets}, whose Reagents column has always been
    protocol plus sublibrary; the exact split is stated in the caption, which is
    a more precise place for it than a four-point band.

    With two categories the strongest available separation is lightness, and the
    repo palette already supplies it as a matched pair: PLOT_PALETTE is the line
    color and PLOT_PALETTE_FILL its pale companion, which is exactly the
    "lighter member of a two-level bar" the standard reserves fills for. Dark to
    pale bottom to top also happens to run bench to sequencer.
    """
    design = CM.ScreenDesign(cells_per_gene=BASE_CELLS_PER_GENE)
    budgets = [CM.budget_for(design, p) for p in CM.PLATFORMS]
    x = np.arange(len(budgets))
    w = 0.6

    reagents = np.array([b.protocol_usd + b.sublibrary_usd for b in budgets]) / 1e3
    seq = np.array([b.sequencing_usd for b in budgets]) / 1e3
    dark = [PLATFORM_COLOR[b.platform] for b in budgets]
    pale = [PLATFORM_FILL[b.platform] for b in budgets]

    ax.bar(x, reagents, w, color=dark, edgecolor="black", lw=0.5)
    ax.bar(x, seq, w, bottom=reagents, color=pale, edgecolor="black", lw=0.5)

    # Totals INSIDE the bars, at 92% of each bar's height. That fraction lands
    # in the pale sequencing segment for all four platforms -- checked, not
    # assumed: the reagent fraction is at most 81% (10x) -- so the label always
    # has a light ground under it and never straddles the boundary between the
    # two segments. Putting them inside also returns the headroom the old
    # above-bar placement needed, which is why ylim drops from 470 to 400 and
    # the tallest bar now fills the panel instead of floating in it.
    # Two numbers per bar, and they are the two a reader wants: the TOTAL above
    # the bar, and the REAGENT subtotal sitting on the line that divides the two
    # segments. The divider is where the split actually is, so a label there
    # needs no legend entry to be understood, and it turns a stacked bar from
    # something to be measured against the axis into something that states its
    # own decomposition.
    for xi, b, rg in zip(x, budgets, reagents):
        ax.text(xi, b.recurring_usd / 1e3 + 10, f"${b.recurring_usd/1e3:.0f}k",
                ha="center", va="bottom", fontsize=5, fontweight="bold")
        ax.text(xi, rg + 6, f"${rg:.0f}k", ha="center", va="bottom",
                fontsize=4.5, color=C_INK)

    ax.set_xticks(x)
    ax.set_xticklabels([PLATFORM_LABEL[b.platform] for b in budgets], fontsize=4.5)
    ax.set_ylabel("Recurring cost per screen ($ thousands)")
    ax.set_ylim(0, 430)
    # Category key only -- the color key is the x-axis, which names the platform
    # under every bar. Grey swatches so the legend cannot be read as a fifth
    # platform, and the same dark/pale relation the bars use.
    ax.legend(
        handles=[
            # Two words each. The long forms -- "reagents (barcoding, library,
            # sublibrary)" and "sequencing (NovaSeq X 25B)" -- made the legend
            # wide enough to reach the third bar and be drawn across its top.
            # What each term contains belongs in the caption, which has room for
            # it; the legend only has to distinguish two things.
            Patch(facecolor=KEY_DARK, edgecolor="black", lw=0.5,
                  label="reagents"),
            Patch(facecolor=KEY_PALE, edgecolor="black", lw=0.5,
                  label="sequencing"),
        ],
        frameon=False, loc="upper left", fontsize=4.5, handlelength=1.0,
        handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2,
    )
    ax.set_title(f"{BASE_CELLS_PER_GENE} cells per target gene", loc="left",
                 fontsize=6)
    box(ax)


# =============================================================================
# (c) scaling with the precision target
# =============================================================================


def panel_c(ax) -> None:
    """Total recurring cost against cells per target gene.

    Panel (b) prices one point. This asks whether the ordering at that point is
    a property of the platforms or of the point, which is the question a reader
    who wants a different design has to be able to answer. It is also where the
    two cost structures become visible as different SLOPES: a platform whose
    cost is reagents scales with batches (a step function, rounded up), and one
    whose cost is sequencing scales with reads (smooth).
    """
    tiers = [50, 100, 250, 500, 1000]
    for p in CM.PLATFORMS:
        ys = [CM.budget_for(CM.ScreenDesign(cells_per_gene=t), p).recurring_usd
              for t in tiers]
        ax.plot(tiers, ys, lw=0.9, color=PLATFORM_COLOR[p.name], marker="o",
                ms=2.2, markeredgecolor="black", markeredgewidth=0.3,
                ls=PROJECTED_LS if p.projected else "-",
                label=PLATFORM_SHORT[p.name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(tiers)
    ax.set_xticklabels([str(t) for t in tiers])
    ax.set_xlim(42, 1200)
    ax.set_ylim(8e3, 3e6)
    ax.set_xlabel("Cells per target gene")
    ax.set_ylabel("Recurring cost per screen")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1e3:,.0f}k"))
    # The field-standard entry point, marked because it is the design the
    # recommendation in Sec. 5.5 actually names.
    ax.axvline(100, color=C_REF, lw=0.4, ls=":")
    ax.text(104, 1.1e4, "field standard", fontsize=4.5, color=C_REF,
            ha="left", va="bottom")
    ax.legend(frameon=False, loc="upper left", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Does the ordering survive the target?", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (d) what happens to a purchased read
# =============================================================================

# The four multiplicative tolls between a purchased read and a usable mRNA read
# from a cell that survives. Order is the order they are levied in, which is
# also the order that makes the cumulative curve monotone and readable.
READ_STAGES = [
    ("purchased", lambda p: 1.0),
    ("not PhiX", lambda p: 1.0 - p.phix_fraction),
    ("barcode resolves", lambda p: p.valid_barcode_fraction),
    ("mRNA, not rRNA", lambda p: p.mrna_read_fraction),
    ("cell survives QC", lambda p: p.usable_fraction),
]


def panel_d(ax) -> None:
    """Cumulative surviving fraction of a purchased read, stage by stage.

    This panel is the answer to why the reagent comparison inverts. Split-pool
    reagents are ~22x cheaper per cell than droplet and it still loses end to
    end, and every bit of that reversal is here: 94 reads in 100 buy ribosomal
    RNA, and three quarters of the cells they came from are then discarded. A
    reader who only sees the totals in panel (b) has to take the inversion on
    trust; a reader who sees this can locate it.

    Log axis, because the endpoints span 1.0% to 37% and a linear axis would
    show the split-pool line as flat against zero -- which is the opposite of
    the point.
    """
    xs = np.arange(len(READ_STAGES))
    finals: dict[str, list] = {}
    for p in CM.PLATFORMS:
        y, acc = [], 1.0
        for _, f in READ_STAGES:
            acc *= f(p)
            y.append(acc)
        # Open markers for the projected platform, filled for the rest -- the
        # same convention panel (e) uses, and here it does double duty: an open
        # marker lets the coincident purple show through at every stage, not
        # just in the gaps between dashes.
        ax.plot(xs, y, lw=0.9, color=PLATFORM_COLOR[p.name], marker="o", ms=2.4,
                markeredgecolor="black", markeredgewidth=0.3,
                markerfacecolor="none" if p.projected else PLATFORM_COLOR[p.name],
                ls=PROJECTED_LS if p.projected else "-")
        finals.setdefault(f"{y[-1]*100:.1f}%", []).append((p, y[-1]))
    # ONE label per distinct endpoint, not one per platform, and INSIDE the axes.
    # The 10x and scifi curves are not merely close, they are identical at every
    # stage: preindexing changes cells per priced channel and touches no term in
    # this panel. Two labels printed on top of each other said that badly.
    #
    # Anchored above-left of each endpoint rather than to its right. To the right
    # is outside the frame -- the last stage sits on the axis edge by
    # construction, so any label there needs an exemption from the legibility
    # check, which is a way of saying it is outside.
    for txt, group in finals.items():
        # BELOW-left, not above-left. Every curve descends left to right into
        # its endpoint, so the space above and to the left of the last marker is
        # occupied by the curve's own final segment -- which is where the labels
        # landed, printed along the lines they name. Below-left is clear on all
        # three, and the endpoints are far enough apart on a log axis that no
        # label reaches the curve beneath it.
        ax.annotate(txt, (xs[-1], group[0][1]), xytext=(0, -6),
                    textcoords="offset points", fontsize=4.5, ha="center",
                    va="top", color=PLATFORM_COLOR[group[0][0].name])

    # Legend, because coincident curves cannot name themselves. The third entry
    # is a TUPLE handle -- purple solid and blue dashed drawn side by side under
    # one label -- which is the honest rendering of two platforms that share a
    # curve: neither is hidden, and the label says they are identical rather
    # than leaving a reader to wonder which color won.
    coincident = [p for p in CM.PLATFORMS
                  if p.name.startswith("10x")]
    handles = [
        Line2D([], [], color=PLATFORM_COLOR[p.name], lw=0.9,
               label=PLATFORM_SHORT[p.name])
        for p in CM.PLATFORMS if not p.name.startswith("10x")
    ]
    handles.append(
        tuple(
            Line2D([], [], color=PLATFORM_COLOR[p.name], lw=0.9,
                   ls=PROJECTED_LS if p.projected else "-", marker="o", ms=2.4,
                   markeredgecolor="black", markeredgewidth=0.3,
                   markerfacecolor="none" if p.projected
                   else PLATFORM_COLOR[p.name])
            for p in coincident
        )
    )
    labels = [h.get_label() for h in handles[:-1]] + ["10x and 10x + scifi (identical)"]
    ax.legend(handles, labels, frameon=False, loc="lower left", fontsize=4.5,
              handlelength=1.8, handletextpad=0.4, labelspacing=0.3,
              borderaxespad=0.2,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    ax.set_yscale("log")
    ax.set_xticks(xs)
    # Short forms, because five two-line category names do not fit across
    # 57 mm -- "mRNA, not rRNA" and "cell survives QC" ran into each other. The
    # full meaning of each stage is in the caption and in READ_STAGES.
    ax.set_xticklabels(["bought", "not\nPhiX", "barcode\nok", "mRNA\nnot rRNA",
                        "cell\npasses"], fontsize=4.5)
    ax.set_xlim(-0.35, len(READ_STAGES) - 0.25)
    ax.set_ylim(3.5e-3, 2.0)
    ax.set_ylabel("Fraction of purchased reads surviving")
    ax.set_title("Where a purchased read goes", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (e) sensitivity to the one number each platform rests on
# =============================================================================


def panel_e(ax) -> None:
    """Total cost against cells per batch, the dominant unmeasured parameter.

    Every platform in this model has exactly one number that is both uncertain
    and load-bearing, and for all of them it is the same field: cells barcoded
    per protocol run for split-pool, cells recovered per channel for droplet. So
    they share an x-axis and can be compared on the ONE thing worth measuring
    next.

    The panel's content is that the two platforms are not equally worth
    measuring. Droplet cost is reagents, and reagents are per batch, so its
    curve falls steeply and does not flatten. Split-pool cost is sequencing,
    which does not depend on this number at all, so its curve flattens almost
    immediately -- past a few hundred thousand cells per run, learning that
    yeast tolerates a denser RT well buys almost nothing.
    """
    design = CM.ScreenDesign(cells_per_gene=BASE_CELLS_PER_GENE)
    grid = np.logspace(np.log10(5e3), np.log10(1.5e6), 40)

    # ONE curve per distinct cost structure, not one per platform. 10x and its
    # preindexed version differ in exactly one field, and that field is this
    # panel's x-axis -- so they are the SAME curve, and preindexing is a move
    # along it rather than a move to a different one. Drawing both stacked a
    # dashed line on a solid one and hid the purple entirely, which said the
    # opposite of what is true. Curves are keyed on everything except
    # cells_per_batch; the operating points are drawn per platform afterwards.
    def signature(p):
        return (p.mrna_umis_per_cell, p.mrna_read_fraction, p.usable_fraction,
                p.reads_per_cell, p.cost_per_batch_usd, p.cells_per_sublibrary,
                p.cost_per_sublibrary_usd, p.phix_fraction,
                p.valid_barcode_fraction)

    drawn: set = set()
    for p in CM.PLATFORMS:
        if signature(p) in drawn:
            continue
        drawn.add(signature(p))
        ys = [
            CM.budget_for(
                design, p.model_copy(update={"cells_per_batch": int(g)})
            ).recurring_usd
            for g in grid
        ]
        ax.plot(grid, ys, lw=0.9, color=PLATFORM_COLOR[p.name], zorder=2)

    # The published (or projected) operating point of every platform, so each
    # curve is read as a sensitivity around a real value rather than as a free
    # parameter -- and so the two droplet points are visibly on one curve.
    for p in CM.PLATFORMS:
        b = CM.budget_for(design, p)
        ax.plot([p.cells_per_batch], [b.recurring_usd], marker="o", ms=3.4,
                color=PLATFORM_COLOR[p.name], markeredgecolor="black",
                markeredgewidth=0.4, zorder=4,
                fillstyle="none" if p.projected else "full")
    # Both notes go to the BOTTOM-LEFT, which is the only large empty region on
    # this panel: every curve enters at the top-left and falls to the right, so
    # the wedge under them at small x is clear. The coincidence note used to sit
    # at (2e4, 3.6e5), which is on the purple curve and beside its marker --
    # exactly where a note about a line must not be.
    ax.annotate("10x and scifi are one curve;\npreindexing moves along it",
                (5.2e3, 8.5e4), fontsize=4.5, color=C_REF,
                ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4e3, 2e6)
    ax.set_ylim(3e4, 3e6)
    ax.set_xlabel("Cells per batch (run, or channel)")
    ax.set_ylabel("Recurring cost per screen")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v/1e3:,.0f}k"))
    # A legend rather than a sentence: the marker convention is a key, and a key
    # drawn as its own marks is read at a glance where a sentence has to be
    # decoded. Grey swatches so it cannot be mistaken for a fifth platform.
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", ms=3.4, color=KEY_DARK,
                   markeredgecolor="black", markeredgewidth=0.4,
                   label="published"),
            Line2D([], [], marker="o", ls="", ms=3.4, markerfacecolor="none",
                   color=KEY_DARK, markeredgecolor=KEY_DARK,
                   markeredgewidth=0.8, label="projected"),
        ],
        frameon=False, loc="lower left", fontsize=4.5, handlelength=1.0,
        handletextpad=0.3, labelspacing=0.25, borderaxespad=0.2,
    )
    ax.set_title("What is worth measuring next", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (f) multiplexing
# =============================================================================


def panel_f(ax) -> None:
    """Multiplexing: main effects scale, named interactions do not."""
    plexes = list(range(1, 11))
    cells = [CM.cells_for_main_effects_kplex(BASE_CELLS_PER_GENE, 6000, k)
             for k in plexes]
    pair_focus = [CM.cells_per_named_pair(n, 200, k)
                  for n, k in zip(cells, plexes)]

    ax.plot(plexes, np.array(cells) / 1e6, marker="o", ms=2.6, lw=0.8,
            color=PLOT_PALETTE[0], markeredgecolor="black", markeredgewidth=0.4,
            label="cells for main effects (left)")
    ax.set_xlabel("Guides per cell, $k$")
    ax.set_ylabel("Total cells needed (millions)")
    ax.set_ylim(0, 1.7)
    ax.set_xticks(plexes)
    ax.set_xticklabels([str(k) for k in plexes])
    ax.set_xlim(0.4, 10.6)
    box(ax)

    ax2 = ax.twinx()
    ax2.plot(plexes, pair_focus, marker="s", ms=2.6, lw=0.8, ls="--",
             color=PLOT_PALETTE[4], markeredgecolor="black", markeredgewidth=0.4,
             label="cells per gene pair, 200-gene panel (right)")
    # The 100-cell floor is the minimum a perturbation needs to be callable at
    # all; a pair below it is not measurable however many cells the screen has.
    ax2.axhline(CM.CELLS_FLOOR, color=C_REF, lw=0.5, ls=":")
    ax2.text(10.5, CM.CELLS_FLOOR * 1.12, "100-cell floor", fontsize=4.5,
             color=C_REF, ha="right")
    ax2.set_ylabel("Cells per gene pair")
    ax2.set_ylim(0, 400)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=4.5,
              handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)
    ax.set_title("Main effects get cheaper with plex; pairs do not",
                 loc="left", fontsize=6)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    # 2 x 3 at full Nature width. Height is the free dimension (width is what
    # makes panels tile), and 118 mm keeps the whole figure inside the 170 mm
    # ceiling with room for a caption on the same page.
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(118.0))
    )
    flat = list(axes.flat)
    for fn, ax in zip((panel_a, panel_b, panel_c, panel_d, panel_e, panel_f), flat):
        fn(ax)
    fig.tight_layout(pad=0.4, w_pad=2.4, h_pad=2.8, rect=(0.012, 0.0, 1.0, 0.965))
    place_panel_letters(fig, flat, ["a", "b", "c", "d", "e", "f"])

    # Legibility gate. Six panels at 57 mm is where labels start colliding, and
    # every number in them moves when a rate card or a measured constant does.
    # Exemptions are labels deliberately at a panel edge.
    assert_legible(
        fig,
        axes=flat,
        # Only two exemptions left, both labels deliberately set against a
        # reference line at the panel edge. Every endpoint label and every note
        # that used to be exempt is now genuinely inside its axes.
        exempt={"field standard", "100-cell floor"},
    )

    out = osp.join(OUT_DIR, "economics.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    # Printed, not eyeballed: the caption quotes these and they must be
    # recomputed on every run so a rate-card change cannot leave a stale number
    # in the prose.
    design = CM.ScreenDesign(cells_per_gene=BASE_CELLS_PER_GENE)
    print(f"\nat {BASE_CELLS_PER_GENE} cells/gene")
    for p in CM.PLATFORMS:
        b = CM.budget_for(design, p)
        surv = 1.0
        for _, f in READ_STAGES:
            surv *= f(p)
        print(f"  {PLATFORM_SHORT[p.name]:26s} ${b.recurring_usd:>9,.0f}  "
              f"batches {b.n_batches:>4}  reagents "
              f"{100*(b.protocol_usd+b.sublibrary_usd)/b.recurring_usd:4.0f}%  "
              f"usable reads {100*surv:5.2f}%")


if __name__ == "__main__":
    main()
