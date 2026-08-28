# experiments/024-perturb-seq-costing/scripts/plot_effect_size.py
# [[experiments.024-perturb-seq-costing.scripts.plot_effect_size]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_effect_size
"""What differential expression actually looks like after a gene deletion.

Three panels, from the Kemmeren and Sameith compendia via
``effect_size_analysis.py``. The point of the figure is that "a two-fold change"
is a badly chosen design target for this biology, and each panel says so a
different way:

(a) The distribution of |log2 FC| over all genes and strains. It is a steep decay
    with almost all of its mass below the thresholds screens are usually designed
    around, so the choice of threshold decides how much of the response is
    visible at all.
(b) How many genes respond, per strain, and how variable that is between strains.
    A median deletion is not a typical deletion: the spread across strains is
    larger than the difference between singles and doubles.
(c) Responders against the threshold itself -- the design curve. Reading off how
    fast the count collapses from 1.25x to 2x is what makes the cost of a
    two-fold target concrete.

Output: $ASSET_IMAGES_DIR/024-perturb-seq-costing/effect_size.svg
"""

from __future__ import annotations

import json
import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_checks import assert_legible
from dotenv import load_dotenv

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")
RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results"
)

# Display order and color. Singles first, double last, so the eye reads
# "one perturbation, one perturbation, two" rather than by study.
SERIES = [
    ("kemmeren2014_single", "Kemmeren, single", PLOT_PALETTE[0]),
    ("sameith2015_single", "Sameith, single", PLOT_PALETTE[1]),
    ("sameith2015_double", "Sameith, double", PLOT_PALETTE[2]),
]

# The thresholds the document argues about, marked on every panel where they
# apply so the three panels can be read against each other.
MARKS = [(1.25, "1.25$\\times$"), (2.0, "2$\\times$")]


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
    """Panel letters measured from each panel's true extent (house rule)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(max(bb.x0 - 0.012, 0.002), min(bb.y1 + 0.020, 0.985), letter,
                 fontsize=8, fontweight="bold", ha="left", va="bottom", zorder=20)


def panel_a(ax) -> None:
    """Distribution of |log2 FC| over every gene in every strain."""
    h = pd.read_csv(osp.join(RESULTS, "effect_size_histogram.csv"))
    for key, lab, color in SERIES:
        d = h[h.dataset == key]
        mid = (d.lo + d.hi) / 2
        ax.plot(mid, d.frac, lw=0.9, color=color, label=lab)
    for fold, lab in MARKS:
        x = math.log2(fold)
        ax.axvline(x, color="#666666", lw=0.4, ls=":", zorder=1)
        ax.text(x + 0.03, 0.5, lab, fontsize=5, color="#666666",
                rotation=90, va="top", ha="left")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.6)
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlabel("Response magnitude in log2 units, $|\\Delta|$")
    ax.set_ylabel("Fraction of gene-strain pairs")
    ax.set_title("Most genes barely move", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="lower left", fontsize=5, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def panel_b(ax) -> None:
    """Responders per strain at 1.25x: the spread between strains.

    Violins, not boxes, and the change is not cosmetic. This panel's whole claim
    is about spread, and a boxplot was making that claim badly in two ways. It
    showed no density, so "most deletions are quiet, a few are enormous" had to be
    taken on trust; and with ``showfliers=False`` it clipped the whiskers at
    1.5 IQR, hiding the very tail the claim rests on -- 12% of Kemmeren's strains
    move more than 1,000 genes against a median of 269, and the largest moves
    4,810. Those strains were invisible.

    The density is computed on LOG10 counts and drawn on a linear axis relabelled
    in decades, rather than by handing raw counts to a violin and setting a log
    y-scale. matplotlib builds the KDE in data space, so the latter smooths in
    linear counts and then squashes the result -- which fattens the low end and
    flattens the tail, i.e. distorts exactly what the panel is about. Log space is
    also where these distributions are nearly symmetric (skew 1.3--3.3 raw,
    0.08--0.56 logged), so it is where a KDE is a fair summary at all.
    """
    df = pd.read_csv(osp.join(RESULTS, "effect_size_per_strain.csv"))
    data, labels, colors = [], [], []
    for key, lab, color in SERIES:
        v = df[df.dataset == key]["n_resp_1.25x"].to_numpy(dtype=float)
        data.append(np.log10(np.clip(v, 1, None)))
        labels.append(lab.replace(", ", "\n"))
        colors.append(color)

    parts = ax.violinplot(data, positions=range(1, len(data) + 1), widths=0.72,
                          showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(1.0)

    # Quartile box + full range inside each violin, so the summary statistics the
    # text quotes are readable off the same mark that shows the shape.
    for i, v in enumerate(data, start=1):
        lo, q1, med, q3, hi = np.percentile(v, [0, 25, 50, 75, 100])
        ax.vlines(i, lo, hi, color="black", lw=0.5, zorder=3)
        ax.vlines(i, q1, q3, color="black", lw=2.4, zorder=4)
        ax.plot([i], [med], marker="o", ms=2.2, color="white",
                markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    # Sample size under each violin. Without it the two Sameith violins read as
    # though their upper values were CAPPED: both end in a blunt horizontal cut
    # rather than tapering. They are not capped. A violin body is clipped at the
    # observed extremes, and with 72-82 strains the kernel density is still
    # appreciable when it reaches the largest strain, so it stops flat; Kemmeren
    # has 1,484, so its density has decayed to nothing by its maximum and comes
    # to a point. The flat edge is where the data stops, not where it was cut,
    # and n is what makes that legible without reading a caption.
    for i, v in enumerate(data, start=1):
        ax.text(i, 0.78, f"n = {len(v):,}", ha="center", va="bottom",
                fontsize=4.5, color="#666666")

    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylim(0.7, 3.9)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["$10^1$", "$10^2$", "$10^3$"])
    ax.set_ylabel("Genes responding at 1.25$\\times$")
    ax.set_title("A median deletion is not a typical one", loc="left", fontsize=6)
    box(ax)


def panel_c(ax) -> None:
    """The design curve: responders against the threshold you choose."""
    lad = pd.read_csv(osp.join(RESULTS, "effect_size_threshold_ladder.csv"))
    for key, lab, color in SERIES:
        d = lad[lad.dataset == key].sort_values("fold")
        ax.plot(d.fold, d.median_responders, lw=0.9, color=color, marker="o",
                ms=2.2, markeredgecolor="black", markeredgewidth=0.3, label=lab)
        ax.fill_between(d.fold, d.q25_responders, d.q75_responders,
                        color=color, alpha=0.18, linewidth=0)
    for fold, lab in MARKS:
        ax.axvline(fold, color="#666666", lw=0.4, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.08, 6.5)
    ax.set_ylim(1, 3e3)
    ax.set_xticks([1.25, 1.5, 2, 3, 4, 6])
    ax.set_xticklabels(["1.25", "1.5", "2", "3", "4", "6"])
    ax.set_xlabel("Fold-change threshold")
    ax.set_ylabel("Genes responding (median)")
    ax.set_title("The threshold decides the screen", loc="left", fontsize=6)

    # The second reading of this panel, and the reason it exists in this form.
    # Sec. 4.4 quotes "the median responding gene moves 1.34x" and feeds that
    # into the power calculation, where it costs a factor of 5.6 in cells. It is
    # worth seeing that 1.34 is not a measured property of a yeast deletion: the
    # |log2 FC| distribution falls off monotonically, so the median of whatever
    # upper tail you select sits just above the cut that selected it. Plotted
    # against the cut, the relation is close to a straight line through the
    # identity -- which is what "the threshold decides the screen" means
    # quantitatively, and what makes the 1.25x choice, not the biology, the thing
    # to argue about.
    ax2 = ax.twinx()
    ref = lad[lad.dataset == "kemmeren2014_single"].sort_values("fold")
    ax2.plot(ref.fold, ref.median_fold_responders, lw=0.9, ls="--",
             color="#666666", marker="s", ms=2.0, markeredgecolor="black",
             markeredgewidth=0.3, zorder=1)
    ax2.plot(ref.fold, ref.fold, lw=0.5, ls=":", color="#BBBBBB", zorder=0)
    ax2.set_yscale("log")
    ax2.set_ylim(1.05, 20)
    ax2.set_yticks([1.25, 2, 5, 10])
    ax2.set_yticklabels(["1.25", "2", "5", "10"], fontsize=5)
    ax2.set_ylabel("Median fold change among responders", fontsize=5,
                   color="#666666")
    ax2.tick_params(axis="y", colors="#666666")
    # Anchored to the last point of the dashed series and grown down-and-left
    # into the empty top-right corner. Anything nearer the middle of the panel
    # lands on the shaded interquartile bands, where grey-on-grey is unreadable.
    # ABOVE the last point, not below it. Anchored with va="top" the block grew
    # downward from (6, 13) and lay along the dashed series it names, which is
    # the one place a label for a line must not be. Growing upward puts it in
    # the empty top-right corner instead.
    ax2.annotate("median responder\n(right axis)", (6.0, 13.0),
                 xytext=(-2, 4), textcoords="offset points", fontsize=4.5,
                 ha="right", va="bottom", color="#666666")
    # "threshold", not "threshold itself". The dotted line IS y = the threshold,
    # so the word "itself" was doing the work of a comparison the reader has to
    # make anyway -- the dashed series sits above this line, and that gap is the
    # whole point. Naming the line for what it plots is enough.
    ax2.annotate("threshold", (3.0, 3.0), xytext=(4, -3),
                 textcoords="offset points", fontsize=4.5, ha="left",
                 va="top", color="#BBBBBB")

    # NO legend on this panel. The three colored series are the same three as
    # panel a, which carries the key; repeating it here costs the only clear
    # corner left, and the dashed grey series is named by its annotation.
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    box(ax)


def panel_d(ax) -> None:
    """Extrapolating responders to more perturbations, and by which model."""
    u = pd.read_csv(osp.join(RESULTS, "union_extrapolation.csv"))
    u = u[u.dataset == "kemmeren2014_single"].sort_values("k")
    with open(osp.join(RESULTS, "union_extrapolation_fit.json")) as fh:
        fit = json.load(fh)
    k = u.k.to_numpy(dtype=float)

    ax.fill_between(k, u.q25, u.q75, color=PLOT_PALETTE[0], alpha=0.18, linewidth=0)
    ax.plot(k, u["median"], lw=0.9, color=PLOT_PALETTE[0], marker="o", ms=2.2,
            markeredgecolor="black", markeredgewidth=0.3,
            label="union of k singles (null)")

    kk = np.linspace(1, 10, 200)
    g, pp = fit["G_eff"], fit["p"]
    ax.plot(kk, g * (1 - (1 - pp) ** kk), lw=0.8, ls="-", color=PLOT_PALETTE[1],
            label="saturating fit")
    ax.plot(kk, u["median"].iloc[0] * kk, lw=0.8, ls="--", color=PLOT_PALETTE[5],
            label="linear in k")
    # The asymptote is the model's substantive claim: a ceiling on how much of the
    # transcriptome any number of perturbations can move.
    ax.axhline(g, color=PLOT_PALETTE[1], lw=0.4, ls=":", zorder=1)
    ax.text(10.2, g * 1.04, f"ceiling {g:.0f}", fontsize=5,
            color=PLOT_PALETTE[1], ha="right", va="bottom")

    # Sameith's observed doubles, the only real multi-perturbation point.
    ax.plot([2], [fit["sameith_observed_k2"]], marker="D", ms=3.4,
            color=PLOT_PALETTE[2], markeredgecolor="black", markeredgewidth=0.4,
            ls="", label="Sameith doubles (observed)", zorder=6)

    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.6)
    # Headroom above the ceiling, and the legend pushed below it. At ylim 5800
    # the 5,264 rule sat a whisker under the frame and the legend, anchored to
    # the top-left corner, was drawn straight through it -- a dotted red line
    # crossing four legend rows. Raising the limit opens a clear band for the
    # "ceiling" label, and anchoring the legend at 0.80 of the axes height puts
    # its top under the rule rather than across it. The region it now occupies
    # is empty: the union curve does not reach 2,500 until k = 10.
    ax.set_ylim(0, 6600)
    ax.set_xlabel("Perturbations per cell, $k$")
    ax.set_ylabel("Genes responding at 1.25$\\times$")
    ax.set_title("Extrapolating past two perturbations", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80),
              fontsize=5, handlelength=1.4, handletextpad=0.4,
              borderaxespad=0.2, labelspacing=0.3)
    box(ax)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    fig, axes = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(108.0))
    )
    flat = axes.ravel()
    for fn, ax in zip((panel_a, panel_b, panel_c, panel_d), flat):
        fn(ax)
    fig.tight_layout(pad=0.4, w_pad=2.6, h_pad=2.4, rect=(0.012, 0.0, 1.0, 0.962))
    place_panel_letters(fig, flat, ["a", "b", "c", "d"])
    # Legibility gate; see figure_checks.py. Panel c is the risky one -- it
    # carries two y-axes, three shaded bands and two grey annotations, and its
    # label positions were chosen against one version of the ladder.
    assert_legible(fig, axes=list(flat), exempt={"threshold itself"})

    out = osp.join(OUT_DIR, "effect_size.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    lad = pd.read_csv(osp.join(RESULTS, "effect_size_threshold_ladder.csv"))
    for key, lab, _ in SERIES:
        d = lad[lad.dataset == key].set_index("fold")["median_responders"]
        print(f"{lab:20s} 1.25x={d.loc[1.25]:6.0f}  2x={d.loc[2.0]:5.0f}  "
              f"4x={d.loc[4.0]:4.0f}   ratio 1.25x/2x = {d.loc[1.25]/d.loc[2.0]:.0f}")

    # Panel (b) shape statistics. The caption quotes the skew pair to justify
    # taking the density in log space, and the range to make the point that the
    # old boxplot's hidden fliers were the whole tail -- so both are printed here
    # rather than read off the picture.
    from scipy import stats
    per = pd.read_csv(osp.join(RESULTS, "effect_size_per_strain.csv"))
    print("\npanel b -- responders per strain at 1.25x")
    for key, lab, _ in SERIES:
        v = per[per.dataset == key]["n_resp_1.25x"].to_numpy(dtype=float)
        lv = np.log10(np.clip(v, 1, None))
        print(f"  {lab:20s} n={len(v):5d}  range {v.min():.0f}-{v.max():.0f}  "
              f"median {np.median(v):.0f}  skew {stats.skew(v):.2f} raw / "
              f"{stats.skew(lv):.2f} logged  frac>1000 {np.mean(v > 1000):.2f}  "
              f"max/median {v.max()/np.median(v):.0f}x")


if __name__ == "__main__":
    main()
