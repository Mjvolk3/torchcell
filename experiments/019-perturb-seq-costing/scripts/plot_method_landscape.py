# experiments/019-perturb-seq-costing/scripts/plot_method_landscape.py
# [[experiments.019-perturb-seq-costing.scripts.plot_method_landscape]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/plot_method_landscape
"""The cells-versus-depth frontier for microbial single-cell RNA-seq.

One panel, log-log: cells profiled on x, mRNA UMIs recovered per cell on y.
The diagonals are iso-lines of constant *total* mRNA UMIs -- the product of the
two axes -- because in the shot-noise limit that product, not either axis alone,
is what sets how precisely a perturbation's mean expression can be estimated.

Reading the plot: methods sit on a frontier running from top-left (plate-based,
deep, few cells) to bottom-right (split-pool, shallow, many cells). Moving along
a diagonal is free; moving up-and-right is what costs money.

Output: $ASSET_IMAGES_DIR/019-perturb-seq-costing/method_landscape.svg
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.lines import Line2D

import method_data as MD
import uiuc_core_data as UC
from figure_checks import assert_legible
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()

# The channel baseline the arrow is measured against. From the rate card, not
# from Datlinger's own 15-fold, which is quoted against a ~10,000-cell standard
# run and would overstate the saving available here.
UC_TENX_BASELINE = UC.TENX_CELLS_PER_CHANNEL
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-perturb-seq-costing")

# Isolation principle -> palette slot. Order follows the repo palette rule:
# warm primaries first, blue/gray last.
ISOLATION_COLOR = {
    "plate": PLOT_PALETTE[0],  # amber
    "droplet": PLOT_PALETTE[1],  # brick
    "split_pool": PLOT_PALETTE[2],  # lilac
    "microwell": PLOT_PALETTE[3],  # wheat
    "preindexed_droplet": PLOT_PALETTE[4],  # blue
}
ISOLATION_LABEL = {
    "plate": "plate / FACS",
    "droplet": "droplet",
    "split_pool": "split-pool",
    "microwell": "microwell array",
    "preindexed_droplet": "preindexed droplet",
}


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
            "svg.fonttype": "none",
        }
    )


# A method needs BOTH coordinates to appear on this plot. Gasch (no UMI in the
# protocol at all) and Ma (reports genes only) have no depth value, and none is
# invented for them -- that substitution is the specific error this figure was
# audited for.
METHODS_PLOTTED = [m for m in MD.METHODS if m.mrna_umis_per_cell is not None]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()

    # FULL width, not half_plus. Thirteen studies each carrying a two-line
    # label do not fit in 88.5 mm without either colliding or dropping below
    # Nature's 5 pt type floor, and the floor is not negotiable. Widening is the
    # only lever left: the same 5 pt text occupies half the fraction of the
    # panel, which is what the eye reads as "smaller labels".
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(88.0))
    )

    # --- iso-lines of constant total mRNA UMIs -------------------------------
    # Axis limits are fixed FIRST, because each iso-line label is positioned as a
    # fraction along the segment of its line that is actually inside the axes.
    # A hand-picked absolute x cannot survive an axis-limit change: the 10^10 line
    # only enters the box in the top-right corner, so an x chosen for it once put
    # the label outside the frame entirely.
    # Whole decades on both axes. Limits like (80, 8e6) put the frame at an
    # arbitrary fraction of a decade, so the visual distance between gridlines
    # stops being a clean order of magnitude and the reader cannot step the axis
    # off by eye. 10^2..10^7 also buys the left margin that the leftmost label
    # (Gasch, at 163 cells) needs to clear the y-axis.
    # The x-axis reaches 10^7 again, but for a different and better reason than
    # before. It used to be stretched there by Gaisser's 1,000,000, which was a
    # protocol CAPACITY claim; that row is gone. Nadal-Ribelles et al. 2025
    # genuinely profiled 1,061,865 cells, so the decade is now carrying a
    # measurement. On y, Nadal-Ribelles 2019 rose to ~11,760 UMIs once its gene
    # count stopped being read as a molecule count, above the old 10^4 ceiling.
    XLIM = (1e2, 1e7)
    YLIM = (1e2, 1e5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)

    xs = np.logspace(2, 7, 100)
    # frac = how far along the VISIBLE segment the label sits, in log space.
    # Short segments (10^10, which clips the corner) take the midpoint so the
    # rotated text has room on both sides; long ones sit near the top-left end,
    # away from the data points.
    # The 10^6 line now runs straight through the shallow-and-few corner where
    # Urbonaite and McNulty sit, so its label is pushed to the very top-left end
    # of the visible segment rather than a quarter of the way along it.
    # `frac` is how far along the line's VISIBLE segment the label sits. The
    # 10^6 value used to be 0.04, which put the text on the y-axis spine: that
    # line only clips the bottom-left corner, so 4% along it is barely inside
    # the frame. These three values are held honest by `assert_legible` at the
    # end of main(), which fails the build if a label leaves the axes.
    for total, frac in [(1e6, 0.34), (1e8, 0.20), (1e10, 0.46)]:
        ax.plot(xs, total / xs, color="#BBBBBB", lw=0.4, ls=":", zorder=1)
        lo = max(XLIM[0], total / YLIM[1])
        hi = min(XLIM[1], total / YLIM[0])
        label_x = lo * (hi / lo) ** frac
        exp = int(np.log10(total))
        ax.text(
            label_x,
            total / label_x * 1.35,
            f"$10^{{{exp}}}$ UMIs",
            fontsize=5,
            color="#888888",
            rotation=-31,
            ha="center",
            va="bottom",
            zorder=1,
        )

    # --- methods -------------------------------------------------------------
    for m in MD.METHODS:
        if m.mrna_umis_per_cell is None:
            continue
        c = ISOLATION_COLOR.get(m.isolation, PLOT_PALETTE[5])
        # A perturbation readout is what makes a method usable for a screen at
        # all, so it gets the filled marker.
        ax.scatter(
            m.cells_profiled,
            m.mrna_umis_per_cell,
            s=16,
            facecolor=c if m.has_perturbation_readout else "white",
            edgecolor=c,
            linewidth=0.7,
            zorder=3,
        )
        # A bar whose meaning we cannot establish is not drawn. Gasch's is quoted
        # second-hand as "735 +/- 5437 mRNAs (median = 2351)", which cannot be a
        # dispersion statistic and is most likely a garbled range; drawing it
        # would assert a spread we are not in a position to assert. The numbers
        # stay in method_data.py with the quote, and the point is still plotted.
        if m.mrna_umis_low and m.mrna_umis_high and m.spread_kind != "unclear":
            ax.plot(
                [m.cells_profiled] * 2,
                [m.mrna_umis_low, m.mrna_umis_high],
                color=c,
                lw=0.5,
                alpha=0.65,
                zorder=2,
            )

    # Labels are ANCHORED, not offset by a guessed pixel width: each entry gives
    # a side, and matplotlib's alignment keeps the text touching its own marker.
    # Guessing dx from estimated text width is what left "microSPLiT
    # (B. subtilis)" floating a decade away from its point.
    #   side -> (dx, dy, horizontalalignment, verticalalignment)
    SIDES = {
        # 8 pt, not 5. A side-anchored label is now TWO lines split about the
        # marker, so the marker and its vertical range bar sit level with the gap
        # between them and read as though wedged into the text. Extra horizontal
        # clearance separates the block from the marker instead.
        "left": (-8, 0, "right", "center"),
        "right": (8, 0, "left", "center"),
        "above": (0, 5, "center", "bottom"),
        "below": (0, -5, "center", "top"),
        # Corner anchors exist for points close to an axis edge. A centered
        # ("above"/"below") label on the leftmost point runs off the frame, and a
        # "right" label on it collides with the next point up; anchoring at the
        # marker and growing down-and-right clears both.
        "below-right": (4, -4, "left", "top"),
        "above-right": (4, 4, "left", "bottom"),
    }
    placement = {
        # Gasch is no longer here: with no UMI measurement it has no y-value.
        # Nadal-Ribelles is now the leftmost AND topmost point, so it anchors
        # down-and-right to stay inside the frame.
        "Nadal-Ribelles 2019": "below-right",
        "Jackson 2020": "left",
        # Left, not below: Brettner's two-species organism line is wide and
        # centered, and it reaches back across this point.
        "Jariani 2020": "left",
        "McNulty 2023": "below",
        # Above, so its second line does not drop into the legend block below it.
        "Urbonaite 2021": "above",
        "Boocock 2025": "right",
        # Kuchina (25k, 397) and Brettner (43k, 410) are nearly the same height,
        # so they have to separate vertically or their labels interleave: one
        # takes the space above its marker, the other the space below.
        "Brettner 2024": "above",
        "Kuchina 2021": "below",
        # Right, not below, so it clears Kuchina's label -- the two split-pool
        # points sit at nearly the same depth and both cannot grow downward.
        "Brandner 2025": "right",
        "Nadal-Ribelles 2025": "left",
    }
    # Every point is named the same way: Method.study ("First-author Year") on
    # the first line, organism in italic on the second. The protocol names live
    # in Method.label and in the table, not here -- mixing "microSPLiT" with
    # "Jariani 2020" in one figure gave it two naming schemes at once.
    # Long binomials, abbreviated to keep the second line no wider than the first.
    ORG = {"S. cerevisiae + C. albicans": "S. cerevisiae, C. albicans"}
    for m in MD.METHODS:
        if m.mrna_umis_per_cell is None:
            continue
        dx, dy, ha, va = SIDES[placement.get(m.study, "right")]
        # Two independent flags, both of which have to survive into the FIGURE
        # rather than living only in the caption -- a screenshotted panel loses
        # anything that is not drawn.
        #   dagger  = primary paper not held; values from a review's summary table.
        #   ddagger = the bar spans separate EXPERIMENTS of differing depth rather
        #             than conditions within one run.
        marks = ""
        if m.secondhand:
            marks += "\\dagger"
        if m.spread_kind == "across_experiments":
            marks += "\\ddagger"
        label = m.study + (f"$^{{{marks}}}$" if marks else "")
        pt = (m.cells_profiled, m.mrna_umis_per_cell)
        common = dict(xycoords="data", textcoords="offset points", ha=ha,
                      fontsize=5, color="#333333")
        # Two annotations rather than one two-line string, because only the
        # second line is italic and matplotlib styles a Text object as a whole.
        # Mathtext (\it) would style within one string but does not survive
        # Arial + svg.fonttype:none through rsvg-convert -- the same failure that
        # garbled a superscript in the economics panel.
        # Two-line labels must show the SAME gap whichever side they sit on.
        # The three branches below anchor different edges, so a single "line
        # pitch" constant does not give a single visual gap: for the top/bottom
        # branches the pitch is baseline-to-baseline (gap = pitch - height), but
        # for the centered branch offsetting each line by pitch/2 from the marker
        # opens a gap of the FULL pitch. That is why the side-anchored labels
        # (Nadal-Ribelles, Jackson, Boocock, Brettner) read looser than the rest.
        # Work in the gap directly instead, and derive the pitch from it.
        GAP = 0.8   # visible space between the two lines, pt
        FS = 5.0    # font size, pt -- text height to a good approximation
        PITCH = FS + GAP
        if va == "top":  # block grows downward: name first, organism beneath
            ax.annotate(label, pt, xytext=(dx, dy), va="top", **common)
            ax.annotate(ORG.get(m.organism, m.organism), pt,
                        xytext=(dx, dy - PITCH), va="top", style="italic", **common)
        elif va == "bottom":  # grows upward: organism sits below the name
            ax.annotate(label, pt, xytext=(dx, dy + PITCH), va="bottom", **common)
            ax.annotate(ORG.get(m.organism, m.organism), pt,
                        xytext=(dx, dy), va="bottom", style="italic", **common)
        else:  # centered on the marker: split the pair about it by half the GAP,
               # not half the pitch, so the two lines end up GAP apart as above
            ax.annotate(label, pt, xytext=(dx, dy + GAP / 2), va="bottom", **common)
            ax.annotate(ORG.get(m.organism, m.organism), pt,
                        xytext=(dx, dy - GAP / 2), va="top", style="italic", **common)

    # --- the preindexing move, drawn as a vector rather than a point ---------
    # scifi-RNA-seq has no place on the depth axis, and NOT because the depth
    # was never measured -- it was. The article defers its performance metrics
    # to Supplementary Table 2, which is behind Nature's paywall and is not in
    # the mirror, while every UMI-per-cell mention in the article itself is a
    # plot axis. So there is no y-coordinate in hand and none is invented, which
    # is the same outcome as the rule that keeps Gasch and Ma off this axis but
    # a different reason. See the note on the Datlinger row in method_data.py.
    #
    # But the throughput move IS measurable and IS the point, so it is drawn as
    # a horizontal arrow at a height that carries no claim: well above every
    # data point, in the empty band between the top marker (11,760) and the
    # frame. Grey, no marker, and labelled with what it is -- the caption
    # repeats it. A filled circle here would assert a depth we do not have.
    y_arrow = 4.6e4
    ax.annotate(
        "", xy=(MD.SCIFI_RECOVERED_LARGE_RUN, y_arrow),
        xytext=(UC_TENX_BASELINE, y_arrow),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#666666",
                        shrinkA=0, shrinkB=0, mutation_scale=6),
        zorder=4,
    )
    for x in (UC_TENX_BASELINE, MD.SCIFI_RECOVERED_LARGE_RUN):
        ax.plot([x, x], [y_arrow / 1.35, y_arrow * 1.35], color="#666666",
                lw=0.7, zorder=4)
    ax.annotate(
        # Plain words, not "$\\rightarrow$": 5 pt mathtext does not survive
        # Arial + svg.fonttype:none through rsvg-convert and renders as a
        # broken glyph. Same failure as the economics panel's superscript.
        "preindexing: 20k to 152k cells per channel",
        (np.sqrt(UC_TENX_BASELINE * MD.SCIFI_RECOVERED_LARGE_RUN), y_arrow * 1.6),
        ha="center", va="bottom", fontsize=5, color="#333333",
    )
    ax.annotate(
        # "not retrieved", NOT "not published". The paper defers its performance
        # metrics to Supplementary Table 2, which is behind Nature's paywall and
        # is not in the mirror -- so a depth value exists and we do not hold it.
        "Datlinger 2021, human/mouse; depth in an unretrieved supplement",
        (np.sqrt(UC_TENX_BASELINE * MD.SCIFI_RECOVERED_LARGE_RUN), y_arrow / 1.7),
        ha="center", va="top", fontsize=5, color="#666666", style="italic",
    )

    ax.set_xlabel("Cells profiled in the published study")
    ax.set_ylabel("mRNA UMIs recovered per cell")

    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="white",
            markeredgecolor=ISOLATION_COLOR[k],
            markeredgewidth=0.7,
            markersize=4,
            label=ISOLATION_LABEL[k],
        )
        for k in ("plate", "droplet", "split_pool", "microwell")
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=4,
            label="has perturbation readout",
        )
    )
    # OUTSIDE the axes, under the x-label. Five isolation classes plus the
    # fill convention is six rows, and there is no longer a corner of this plot
    # big enough to hold them: the cloud reaches from (285, 11760) to
    # (1.06e6, 1200) and the two remaining gaps are both crossed by an iso-line
    # or a label. Panel WIDTH is the quantity that has to stay fixed for panels
    # to tile; height is free, so the legend buys its space vertically.
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.4)

    fig.tight_layout(pad=0.3)

    # Legibility gate. Every label position in this panel is a considered
    # choice, and every one of them is invalidated by a data change several
    # files away -- a new study, a corrected UMI count. Checking here means a
    # collision stops the SVG being written rather than surfacing three steps
    # later in a rendered PDF that nobody re-opens.
    #
    # The exemptions are the legend rows, which sit under the axes on purpose.
    assert_legible(
        fig,
        axes=[ax],
        exempt={ISOLATION_LABEL[k] for k in ISOLATION_LABEL}
        | {"has perturbation readout"},
    )

    out = osp.join(OUT_DIR, "method_landscape.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    # Frontier statistics, printed rather than eyeballed. The caption states how
    # far each axis and the PRODUCT span, and those three numbers are the whole
    # quantitative content of the "methods trade cells against depth" claim -- if
    # the product spanned as little as a strict frontier implies, it would be
    # near zero decades. Recomputed on every run so a new row cannot leave a
    # stale span in the prose.
    pts = [(m.study, m.cells_profiled, m.mrna_umis_per_cell)
           for m in METHODS_PLOTTED]
    xs_, ys_ = [p[1] for p in pts], [p[2] for p in pts]
    prod = [x * y for _, x, y in pts]
    def span(v):
        return np.log10(max(v)) - np.log10(min(v))
    print(f"\n{len(pts)} plotted studies")
    print(f"  cells      {min(xs_):,} .. {max(xs_):,}      {span(xs_):.2f} decades")
    print(f"  UMIs/cell  {min(ys_):,.0f} .. {max(ys_):,.0f}        {span(ys_):.2f} decades")
    print(f"  product    {min(prod):.2e} .. {max(prod):.2e}  {span(prod):.2f} decades")
    lo = min(pts, key=lambda p: p[1])
    hi = max(pts, key=lambda p: p[1])
    print(f"  {lo[0]} -> {hi[0]}: cells x{hi[1]/lo[1]:,.0f}, depth /{lo[2]/hi[2]:.1f}")


if __name__ == "__main__":
    main()
