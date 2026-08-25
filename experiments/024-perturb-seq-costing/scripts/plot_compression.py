# experiments/024-perturb-seq-costing/scripts/plot_compression.py
# [[experiments.024-perturb-seq-costing.scripts.plot_compression]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/024-perturb-seq-costing/scripts/plot_compression
"""Conventional, compressed and m-perturbation compressed screens, in yeast.

Sec. 3.4 states compressed Perturb-seq's premise -- that the number of composite
samples needed scales as ``(q + r) log n`` rather than n -- and leaves n, q and r
as quantities nobody could plan. Two of them are measurable from compendia
already in the mirror, and ``compression_analysis.py`` measures them. This draws
what they imply.

The three designs on one axis, all of them "m perturbations per composite
measurement":

    conventional        m = 1.  One perturbation per cell, one column of the
                        effect matrix per group of cells. Samples scale as n.
    guide-pooled        m > 1, decoded by averaging. Every cell reports on m
                        perturbations, so a main effect accumulates m times
                        faster and samples scale as n/m. No sparsity assumed.
    compressed          m > 1, decoded by sparse recovery. Samples scale as
                        (q + r) log n -- nearly FLAT in n, which is the whole
                        claim and the reason the comparison is worth drawing.

Two honesty constraints run through every panel.

THE CONSTANT IS UNKNOWN. Yao et al. state an ORDER, and a compressed-sensing
bound's constant depends on the measurement ensemble and on the recovery
guarantee wanted. So no panel here reports an absolute sample count as a budget.
What survives an unknown constant is the SHAPE -- flat in n against linear in n
-- and the library size at which the two cross, which moves with the constant but
whose existence does not.

THE ASSUMPTION IS TESTABLE HERE AND DOES NOT HOLD CLEANLY. Panel (c) is that
test, and it is the reason this figure exists in a document about yeast rather
than being a summary of somebody else's method.

Output: $ASSET_IMAGES_DIR/024-perturb-seq-costing/compression.svg
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D

import cost_model as CM
from figure_checks import assert_legible
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "024-perturb-seq-costing")
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "024-perturb-seq-costing", "results")

# One color per DESIGN, held across every panel that distinguishes them, the
# same discipline the economics figure uses for platforms.
#
# Slots 0, 1, 2 -- the FIRST THREE, which is the palette rule for a series of
# three and needs no deviation here. An earlier version took 0, 1, 4, copying
# the economics figure's documented skip to blue; that skip exists because that
# figure has FOUR series and slots 0 and 3 (amber and wheat) are not separable
# at 0.9 pt. With three series the fourth slot is never reached, so there is
# nothing to avoid and the deviation was inherited rather than justified.
#
# The reference grey is PLOT_PALETTE[5], not a typed "#666666". Same value,
# but a typed hex is a color that has left the palette: nothing updates it if
# the palette moves, and nothing flags it if someone types a near-miss.
C_CONV = PLOT_PALETTE[0]   # conventional, m = 1
C_POOL = PLOT_PALETTE[1]   # guide-pooled, averaged
C_COMP = PLOT_PALETTE[2]   # compressed, sparse recovery
C_REF = PLOT_PALETTE[5]    # reference lines, annotations, non-series marks

# Library sizes the document actually contemplates: the focused metabolic panel
# of Sec. 4.5 and the yeast genome.
N_PANEL = 200
N_GENOME = 6000


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


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    spec = pd.read_csv(osp.join(RESULTS, "compression_spectrum.csv"))
    spar = pd.read_csv(osp.join(RESULTS, "compression_sparsity.csv"))
    add = pd.read_csv(osp.join(RESULTS, "compression_additivity.csv"))
    with open(osp.join(RESULTS, "compression_summary.json")) as fh:
        summ = json.load(fh)
    return spec, spar, add, summ


# =============================================================================
# (a) r
# =============================================================================


def panel_a(ax, spec, summ) -> None:
    """Singular spectrum of the Kemmeren effect matrix, and where r sits.

    This is the panel that makes r a measured quantity rather than a symbol.
    Two readings are wanted at once -- how fast the spectrum falls, and how much
    is accounted for by the time you have taken k components -- so the singular
    values go on the left axis and the cumulative variance on the right.
    """
    k = spec.component.to_numpy()
    ax.plot(k, spec.singular_value, lw=0.9, color=C_COMP)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, len(k))
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")

    ax2 = ax.twinx()
    ax2.plot(k, spec.cumulative_variance, lw=0.9, ls="--", color=C_REF)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Cumulative variance", fontsize=5, color=C_REF)
    ax2.tick_params(axis="y", colors=C_REF, labelsize=5)
    for frac, key in ((0.5, "rank_50pct"), (0.9, "rank_90pct")):
        ax2.plot([summ[key]], [frac], marker="o", ms=2.6, color=C_REF,
                 markeredgecolor="black", markeredgewidth=0.3, zorder=5)
        ax2.annotate(f"{int(frac*100)}%: {summ[key]}", (summ[key], frac),
                     xytext=(3, -1), textcoords="offset points", fontsize=4.5,
                     ha="left", va="top", color=C_REF)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title("The response matrix is low rank", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (b) q
# =============================================================================


def panel_b(ax, spar, summ) -> None:
    """Two sparsities, and only one of them is the one the method needs.

    Row sparsity -- genes moved by ONE perturbation -- needs no basis and no
    threshold beyond the Sec. 4.4 responder cut, and it is what the sample
    complexity is evaluated at. The SVD module support is drawn beside it to
    show what it is NOT: an SVD component is dense by construction, so its
    support is an upper bound, and the gap between the two bars is exactly the
    work a sparse factorization has to do. That gap is why FR-Perturb uses
    sparse PCA rather than PCA, which the method description states as a design
    choice without saying what it buys.
    """
    q = summ["q_median_genes_moved_per_perturbation"]
    lo, hi = summ["q_iqr"]
    masses = sorted(summ["svd_support_median_by_mass"].items(), key=lambda kv: kv[0])
    labels = ["one\nperturbation"] + [f"SVD module\n{int(float(m)*100)}% mass"
                                      for m, _ in masses]
    vals = [q] + [v for _, v in masses]
    colors = [C_COMP] + [C_REF] * len(masses)

    x = np.arange(len(vals))
    ax.bar(x, vals, 0.62, color=colors, edgecolor="black", lw=0.5)
    ax.errorbar([0], [q], yerr=[[q - lo], [hi - q]], fmt="none", ecolor="black",
                elinewidth=0.6, capsize=1.6, capthick=0.6)
    ax.axhline(summ["n_genes"], color=C_REF, lw=0.5, ls=":")
    ax.text(len(vals) - 0.55, summ["n_genes"] * 0.88,
            f"all {summ['n_genes']:,} genes", fontsize=4.5, color=C_REF,
            ha="right", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=4.5)
    ax.set_yscale("log")
    ax.set_ylim(50, 1.5e4)
    ax.set_ylabel("Genes carrying the signal")
    ax.set_title("Sparse per perturbation, dense per component",
                 loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (c) the assumption
# =============================================================================


def panel_c(ax, add) -> None:
    """Do two deletions add? The test compressed recovery depends on.

    Yao et al. do not claim interactions are absent; they claim interactions
    CANCEL, because each cell draws a random combination and there are as many
    positive as negative. That is a claim about the mean, and it is falsifiable
    with a compendium of doubles: if the slope of observed on additively
    predicted is 1, they cancel; if it is below 1, they do not, and the bias is
    toward buffering.

    Plotted as the distribution of that slope over every double whose two
    singles were profiled by the same lab, because a single scatter of one pair
    would show only that pair.
    """
    slopes = add.slope.dropna().to_numpy()
    ax.hist(slopes, bins=np.linspace(0, 1.6, 25), color=C_POOL,
            edgecolor="black", lw=0.4)
    med = float(np.median(slopes))
    ax.axvline(1.0, color=C_REF, lw=0.7, ls="--")
    ax.annotate("additive", (1.0, ax.get_ylim()[1] * 0.94), xytext=(3, 0),
                textcoords="offset points", fontsize=4.5, color=C_REF,
                ha="left", va="top")
    ax.axvline(med, color="black", lw=0.7)
    # Top-left corner, not beside the rule: the median of a unimodal histogram
    # is by construction where the tallest bars are, so a label anchored to it
    # is a label on the data. The rule already marks the position; the text only
    # has to state the value.
    ax.annotate(f"median {med:.2f}", (0.03, ax.get_ylim()[1] * 0.97),
                xytext=(0, 0), textcoords="offset points", fontsize=4.5,
                ha="left", va="top")
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("Observed / additively predicted")
    ax.set_ylabel(f"Double deletions (n = {len(slopes)})")
    ax.set_title("Doubles are quieter than additive", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (d) samples against library size
# =============================================================================


def samples(design: str, n: np.ndarray, q: float, r: float, m: int) -> np.ndarray:
    """Composite measurements needed, per design, in units of the design's own.

    Every constant is 1. The point of the panel is the exponent, not the
    intercept, and a fabricated constant would put a budget on an axis that
    cannot carry one.
    """
    if design == "conventional":
        return n.astype(float)
    if design == "pooled":
        return n / m
    if design == "compressed":
        return np.full_like(n, (q + r), dtype=float) * np.log(n)
    raise ValueError(design)


def panel_d(ax, summ) -> None:
    """Conventional, pooled and compressed against library size."""
    q = summ["q_median_genes_moved_per_perturbation"]
    r = summ["rank_90pct"]
    m = 10
    n = np.logspace(1.3, 4.2, 200)
    ax.plot(n, samples("conventional", n, q, r, m), lw=1.0, color=C_CONV,
            label="conventional, m = 1")
    ax.plot(n, samples("pooled", n, q, r, m), lw=1.0, color=C_POOL,
            label=f"guide-pooled, averaged, m = {m}")
    ax.plot(n, samples("compressed", n, q, r, m), lw=1.0, color=C_COMP,
            label="compressed, sparse recovery")

    # Where compression starts paying: the crossing of the flat curve with the
    # linear one. Marked because it is the only thing on this panel that does
    # not move with the unknown constant in an obvious direction, and it is the
    # number a design has to clear.
    cross = n[np.argmin(np.abs(samples("compressed", n, q, r, m)
                               - samples("conventional", n, q, r, m)))]
    ax.axvline(cross, color=C_REF, lw=0.5, ls=":")
    # LEFT of the rule and at the floor: to the right is the corner where the
    # conventional and compressed curves converge, and no mathtext -- a 4.5 pt
    # \approx does not survive Arial + svg.fonttype:none through rsvg-convert.
    ax.annotate(f"breaks even\nnear n = {cross:,.0f}", (cross, 19),
                xytext=(-3, 0), textcoords="offset points", fontsize=4.5,
                color=C_REF, ha="right", va="bottom")
    for x, lab in ((N_PANEL, "panel"), (N_GENOME, "genome")):
        ax.plot([x], [x], marker="v", ms=3, color=C_REF,
                markeredgecolor="black", markeredgewidth=0.3, zorder=5)
        ax.annotate(lab, (x, x), xytext=(0, 4), textcoords="offset points",
                    fontsize=4.5, color=C_REF, ha="center", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 1.6e4)
    ax.set_ylim(15, 2e4)
    ax.set_xlabel("Library size, n targets")
    ax.set_ylabel("Composite samples needed")
    ax.legend(frameon=False, loc="upper left", fontsize=4.5, handlelength=1.3,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("Compression pays only above a library size",
                 loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (e) what m buys, at each library size
# =============================================================================


def panel_e(ax, summ) -> None:
    """Samples against perturbations per composite, at both library sizes.

    Panel (d) fixes m and sweeps n; this fixes n and sweeps m, which is the
    knob an experiment actually turns. The compressed curves are horizontal
    because sparse recovery's sample requirement does not depend on m at all --
    only on the sparsity and rank of what is being recovered. That is the whole
    difference between the two decodings, and it is easier to see here than
    anywhere else in the figure.
    """
    q = summ["q_median_genes_moved_per_perturbation"]
    r = summ["rank_90pct"]
    ms = np.arange(1, 21)
    for n, ls in ((N_GENOME, "-"), (N_PANEL, "--")):
        ax.plot(ms, np.full_like(ms, n, dtype=float) / ms, lw=1.0, ls=ls,
                color=C_POOL)
        ax.plot(ms, np.full_like(ms, (q + r) * np.log(n), dtype=float), lw=1.0,
                ls=ls, color=C_COMP)
        # Anchored at m = 1, the LEFT end, where the pooled curve sits at n
        # itself. Anchoring at m = 20 put the n = 200 label at n/20 = 10, below
        # the axis floor -- the label was placed by a formula whose value at one
        # end of the sweep leaves the panel.
        ax.annotate(f"n = {n:,}", (1, n), xytext=(3, 2),
                    textcoords="offset points", fontsize=4.5, ha="left",
                    va="bottom", color=C_POOL)
    ax.set_yscale("log")
    ax.set_xlim(0.5, 20.5)
    # Floor at 5, not 30: the pooled curve for the 200-gene panel reaches 10 at
    # m = 20, and a limit that clips a drawn curve is worse than empty space.
    ax.set_ylim(5, 3e4)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Perturbations per composite, m")
    ax.set_ylabel("Composite samples needed")
    ax.legend(handles=[
        Line2D([], [], color=C_POOL, lw=1.0, label="pooled, averaged"),
        Line2D([], [], color=C_COMP, lw=1.0, label="compressed"),
        Line2D([], [], color=C_REF, lw=1.0, ls="-", label="genome, n = 6,000"),
        Line2D([], [], color=C_REF, lw=1.0, ls="--", label="panel, n = 200"),
    ], frameon=False, loc="upper right", fontsize=4.5, handlelength=1.3,
        handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2, ncol=1)
    ax.set_title("Sparse recovery does not care about m", loc="left", fontsize=6)
    box(ax)


# =============================================================================
# (f) the three parameters, and how well each is pinned
# =============================================================================


def panel_f(ax, summ) -> None:
    """Ranges for n, q and r, drawn against how each one is known.

    This panel exists because the three symbols were the reason the compressed
    design could not be planned, and they are not equally unknown. One is a
    choice, one is measured with a spread, and one has no single value at all
    because the spectrum it comes from has no knee. Drawing them on one log axis
    with the basis named under each is the honest summary, and it is what a
    reader should carry away rather than any single number from (d) or (e).

    The bar is the range and the dot is the working value the other panels use.
    """
    r_lo, r_hi = summ["rank_50pct"], summ["rank_95pct"]
    q_lo, q_hi = summ["q_iqr"]
    rows = [
        ("n, library size", N_PANEL, N_GENOME, N_GENOME, C_CONV,
         "a design choice, not a measurement"),
        # Displayed as nu, not q: q is the per-guide detection probability
        # throughout the document. The RESULTS keys stay q_* -- they are what
        # compression_analysis.py writes, and renaming them would orphan every
        # results file already on disk.
        ("$\\nu$, genes moved\nper perturbation", q_lo, q_hi,
         summ["q_median_genes_moved_per_perturbation"], C_POOL,
         f"IQR over {summ['n_strains']:,} Kemmeren deletions"),
        ("r, components", r_lo, r_hi, summ["rank_90pct"], C_COMP,
         "50% to 95% of variance; tick = participation ratio"),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (_, lo, hi, mid, color, _) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], lw=3.0, color=color, solid_capstyle="butt",
                zorder=2)
        ax.plot([mid], [yi], marker="o", ms=3.6, color="white",
                markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        # ABOVE the bar's midpoint, not off its right end. The n row runs to
        # 6,000 against an axis ending at 30,000, and a label hung to the right
        # of it left the frame -- a placement that works for two rows and fails
        # for the third is not a placement.
        ax.annotate(f"{lo:,.0f}-{hi:,.0f}", (np.sqrt(lo * hi), yi),
                    xytext=(0, 5), textcoords="offset points", fontsize=4.5,
                    ha="center", va="bottom", color=color)
    # The participation ratio, a cut-off-free reading of the same spectrum,
    # marked so the r bar is not read as though 50-95% were the only options.
    ax.plot([summ["effective_rank_participation_ratio"]], [y[-1]], marker="|",
            ms=7, color="black", markeredgewidth=0.8, zorder=5)
    # The tick is left unlabelled and named in the basis line below instead:
    # the only clear position for a caption here is directly under the mark,
    # which is where that basis line already sits.
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=4.5)
    ax.set_xscale("log")
    ax.set_xlim(2, 3e4)
    ax.set_ylim(-0.75, len(rows) - 0.4)
    ax.set_xlabel("Value")
    ax.set_title("What each parameter rests on", loc="left", fontsize=6)
    for yi, (_, _, _, _, color, basis) in zip(y, rows):
        ax.annotate(basis, (2.6, yi - 0.30), fontsize=4.0, color=C_REF,
                    ha="left", va="center")
    box(ax)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    spec, spar, add, summ = load()

    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(112.0))
    )
    flat = list(axes.flat)
    panel_a(flat[0], spec, summ)
    panel_b(flat[1], spar, summ)
    panel_c(flat[2], add)
    panel_d(flat[3], summ)
    panel_e(flat[4], summ)
    panel_f(flat[5], summ)
    fig.tight_layout(pad=0.4, w_pad=2.6, h_pad=2.8, rect=(0.012, 0.0, 1.0, 0.965))
    place_panel_letters(fig, flat, ["a", "b", "c", "d", "e", "f"])

    assert_legible(
        fig, axes=flat,
        exempt={"additive", "panel", "genome",
                f"all {summ['n_genes']:,} genes"},
    )

    out = osp.join(OUT_DIR, "compression.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    q = summ["q_median_genes_moved_per_perturbation"]
    r = summ["rank_90pct"]
    print(f"\nq = {q:.0f} genes per perturbation, r = {r} components")
    for n in (N_PANEL, 1000, N_GENOME):
        m = (q + r) * np.log(n)
        print(f"  n = {n:>5}: compressed needs {m:>7.0f} samples = {m/n:5.2f} x n")


if __name__ == "__main__":
    main()
