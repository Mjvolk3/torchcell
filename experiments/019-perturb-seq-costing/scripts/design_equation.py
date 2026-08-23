# experiments/019-perturb-seq-costing/scripts/design_equation.py
# [[experiments.019-perturb-seq-costing.scripts.design_equation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-perturb-seq-costing/scripts/design_equation
"""A single design equation for multiplexed perturb-seq, and its consequences.

Sections 4.3 and 4.5 currently carry three separate rules of thumb -- a shot-noise
requirement, a 100-cell biological floor taken as an independent constraint, and a
combinatorial factor for multiplexing -- and combine the first two with a max().
They are not independent. All three fall out of one expression once the pseudobulk
count is modeled as negative binomial rather than Poisson.

THE EQUATION
------------
Pool n cells carrying a given perturbation, each read to depth d (mRNA UMIs per
cell), and look at gene j whose share of the transcriptome is p_j. The pseudobulk
count is K = n d p_j. Under a negative binomial with gene-level biological
overdispersion phi_j (the squared coefficient of variation of gene j's expression
BETWEEN cells, over and above sampling),

    Var(log K) ~= 1/(n d p_j)  +  phi_j / n
                   ^shot noise     ^biology

Requiring a two-sample log2 fold change Delta to be resolvable at level alpha and
power 1-beta gives the cells-per-perturbation requirement

    n*  =  A(Delta) * [ 1/(d p_j) + phi_j ]                                  (1)
    A(Delta) = 2 (z_{1-alpha/2} + z_{1-beta})^2 / (Delta ln 2)^2

The factor 2 is the two-sample comparison against controls.

Equation (1) is the whole point: the two constraints the document currently takes
a max() over are the two LIMITS of one curve.

  * phi_j -> 0 recovers the pure shot-noise rule, n = A/(d p_j), i.e. a required
    pseudobulk count K = n d p_j = A. With kappa = 1 that is EXACTLY the
    right-hand side of Eq. (power) in Sec. 4.3, (z/(Delta ln 2))^2 = 16.3 at
    two-fold. The two sections agree by construction.
  * d p_j -> infinity gives n -> A phi_j, a floor that no amount of sequencing
    can cross. THAT is the "100-cell biological floor", and (1) says what it
    means quantitatively: at two-fold and 80% power A = 16.3, so a 100-cell floor
    is the statement phi_j ~= 6.1. The field's rule of thumb is a claim about
    yeast gene-level overdispersion, and it is measurable.

Everything else is bookkeeping on top of (1). With k guides per cell drawn from a
panel of T targets, a cell informs an order-r effect only if it carries all r of
those guides. That probability is exactly C(k,r)/C(T,r), so

    N*  =  rho_r * n* * C(T,r) / C(k,r) / ( s * q^r )                        (2)

    rho_r  contrast variance inflation for order r
    s      fraction of sequenced cells surviving QC
    q      per-guide detection probability; q^r for the r guides of the contrast

WHERE THIS COMES FROM, AND WHAT IS ASSUMED
------------------------------------------
Equation (1) is not new. It is the standard negative-binomial sample-size result
from the bulk RNA-seq literature, applied to a pseudobulk pool:

  * Hart et al. 2013, J Comput Biol 20:970, doi:10.1089/cmb.2012.0283 -- gives
    n >= (z_a + z_b)^2 (1/mu + sigma^2) / (log FC)^2, which is Eq. (1) in another
    notation. The closest published statement of this equation.
  * McCarthy, Chen & Smyth 2012, NAR 40:4288, doi:10.1093/nar/gks042 -- defines
    the biological coefficient of variation, BCV = sqrt(phi). Our phi_j IS the
    edgeR NB dispersion, so published BCV values transfer directly.
  * Robinson & Smyth 2008, Biostatistics 9:321, doi:10.1093/biostatistics/kxm030
    -- NB dispersion estimation, the Var = mu + phi mu^2 parameterization.
  * Grun, Kester & van Oudenaarden 2014, Nat Methods 11:637, doi:10.1038/nmeth.2930
    -- the single-cell UMI noise model: Poisson sampling on top of biological
    variation, which is the decomposition Eq. (1) rests on.
  * Svensson 2020, Nat Biotechnol 38:147, doi:10.1038/s41587-019-0379-5 -- UMI
    counts are NB, not zero-inflated. This is what licenses using a plain NB and
    not a ZINB here.
  * Zhang, Ntranos & Tse 2020, Nat Commun 11:774, doi:10.1038/s41467-020-14482-y
    -- the cells-versus-depth optimum for a fixed budget; the published treatment
    of what panel (a) shows.
  * Squair et al. 2021, Nat Commun 12:5692, doi:10.1038/s41467-021-25960-2 --
    why the estimand is pseudobulk rather than per-cell.
  * Replogle et al. 2022, Cell 185:2559, doi:10.1016/j.cell.2022.05.013 --
    genome-scale Perturb-seq; practical cells-per-perturbation in the largest
    published screen.

NONE of these are in our Zotero collection yet, so the document cites them by DOI
rather than by \cite. Add them and they become proper citations.

Assumptions that are ours, and are not free:

  1. rho_2 = 4 is QUOTED from Yao et al., not derived. It corresponds to a full
     2x2 factorial contrast (AB, A, B, neither) with four equally sized groups,
     whose variance is 4 sigma^2. With a large control pool -- the same kappa = 1
     assumption used for main effects -- the exact factor is 3, not 4. We keep
     Yao's 4 because the 400-cells-per-pair figure in Sec. 4.5 is built on it and
     the two must not disagree; it is conservative by ~33%.
  2. The delta method Var(log K) ~= Var(K)/E[K]^2 degrades at small counts, and
     A ~ 16 counts is small. Treat Eq. (1) near its shot-noise limit as an
     order-of-magnitude statement; an exact NB test is the right tool there.
  3. Equal per-cell depth d. Real libraries vary and are handled with size
     factors; d should be read as the mean depth, correct to first order.
  4. Cells are independent, and phi_j is constant across the compared groups.

WHAT IS ACTUALLY MISSING
------------------------
Three inputs are assumed rather than measured, and they are the honest answer to
"what would it take":

  1. phi_j, gene-level biological overdispersion in yeast. THE critical one: it
     alone sets the floor. Estimable from any yeast scRNA-seq count matrix, and
     published BCV values are directly reusable (see McCarthy above).
  2. The Delta distribution -- how large real knockdown responses are. A enters
     as Delta^-2, so this is the most violent term in the equation.
  3. q, per-guide detection probability, which needs the guide-barcode construct
     to exist before it can be measured.

Output: $ASSET_IMAGES_DIR/019-perturb-seq-costing/design_equation.svg
"""

from __future__ import annotations

import math
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_checks import assert_legible
from dotenv import load_dotenv
from matplotlib.ticker import LogLocator

import method_data as MD
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

load_dotenv()
OUT_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "019-perturb-seq-costing")
RESULTS = osp.join(os.environ["EXPERIMENT_ROOT"], "019-perturb-seq-costing", "results")

# Normal quantiles for alpha = 0.05 two-sided and power 0.8. Same convention as
# Eq. (power) in Sec. 4.3, so the two agree by construction.
Z = 2.80

# Expression share of a typical gene: 3.5 molecules per gene per cell against a
# working total of 30,000 mRNA per cell, both from method_data.py's sourced
# constants. A typical gene, not a highly expressed one -- the design has to work
# for the median target, not the best case.
P_TYPICAL = 3.5 / MD.YEAST_MRNA_PER_CELL_WORKING

# Contrast variance inflation by interaction order. rho_2 = 4 is quoted from Yao
# et al. ("four times as many cells are needed to estimate a second-order
# interaction effect as a first-order effect with the same magnitude") and is the
# same constant behind the 400-cells-per-pair figure used in Sec. 4.5.
RHO = {1: 1.0, 2: 4.0}

# Control-pool factor. The variance of an estimated log fold change is the sum of
# the perturbed and control pseudobulk variances. Which value is right depends on
# a design fact, so it is a parameter rather than a hidden 2:
#   KAPPA = 1  the non-targeting control pool is large enough that its variance
#              is negligible -- the usual Perturb-seq case, since NTCs are pooled
#              over many guides. This is ALSO what Eq. (power) in Sec. 4.3
#              assumes, so keeping 1 as the default makes Eq. (1) reduce to that
#              equation exactly in the shot-noise limit instead of differing by 2.
#   KAPPA = 2  control of the same size as the perturbation.
# An earlier version of this module hard-coded 2 and then claimed Eq. (1)
# "recovers the shot-noise expression exactly", which it did not -- it was a
# factor of two above Sec. 4.3. Caught on review.
KAPPA = 1.0


def power_coefficient(delta_log2: float, z: float = Z,
                      kappa: float = KAPPA) -> float:
    """A(Delta) in Eq. (1). Cells per unit of [1/(d p) + phi].

    With kappa = 1 this is exactly (z / (Delta ln 2))^2, i.e. the right-hand side
    of Eq. (power) in Sec. 4.3 -- so the two sections agree by construction
    rather than by coincidence.
    """
    return kappa * z**2 / (delta_log2 * math.log(2.0)) ** 2


def cells_per_perturbation(depth: float, phi: float,
                           delta_log2: float | None = None,
                           p: float = P_TYPICAL) -> float:
    """Eq. (1): n*, cells that must share a perturbation.

    ``delta_log2`` defaults to the MEASURED median response, not to a
    nominal two-fold. Designing at two-fold understates every cell count
    by ~5.7x, and that factor is not a detail: it is the difference
    between "one lane will do" and "a whole flow cell will not".
    """
    d = DELTA_MEASURED if delta_log2 is None else delta_log2
    return power_coefficient(d) * (1.0 / (depth * p) + phi)


def depth_sufficiency(phi: float, p: float = P_TYPICAL) -> float:
    """Depth at which shot noise stops dominating: d* = 1/(phi p).

    At d = d* the two terms of Eq. (1) are equal, so n* is exactly twice its
    floor. Past it, further depth is buying at most a factor of two and the
    budget is better spent on cells or on more perturbations.
    """
    return 1.0 / (phi * p)


def total_cells(n_star: float, n_targets: int, plex: int, order: int = 1,
                survival: float = 1.0, q: float = 1.0) -> float:
    """Eq. (2): N*, total cells to sequence."""
    if plex < order:
        return math.inf
    return (
        RHO[order] * n_star
        * math.comb(n_targets, order) / math.comb(plex, order)
        / (survival * q**order)
    )


def min_detectable_delta(n: float, depth: float, phi: float,
                         p: float = P_TYPICAL, kappa: float = KAPPA) -> float:
    """Eq. (1) solved for Delta: the smallest log2 fold change n cells can resolve.

    Inverting the design equation is what makes it usable as a design tool rather
    than a checker -- the question in practice is never "is 2-fold powered?" but
    "what can I actually see for the budget I have?".
    """
    z = Z
    return (z / math.log(2.0)) * math.sqrt(kappa * (1.0 / (depth * p) + phi) / n)


# --- sequencing scenarios ----------------------------------------------------
# The two configurations worth designing against at the UIUC core, per
# uiuc_core_data.py. One lane is the smallest sensible unit on the biggest flow
# cell; eight lanes is a whole 25B flow cell, which is also where the volume rate
# applies -- $3,180/lane against $3,380, i.e. $0.994 vs $1.056 per million read
# pairs. The discount is only ~6%, so the reason to run a full flow cell is
# throughput, not price per read; the figure makes that visible by plotting what
# each scenario BUYS rather than what it costs.
LANE_READ_PAIRS = 3.2e9
SCENARIOS = [
    ("1 lane", 1, 3380.0),
    ("8 lanes (full flow cell)", 8, 3180.0),
]

# Reads per USABLE cell, per platform, derived from the genome-scale budget the
# cost model already produces (read pairs / cells surviving QC with a guide call).
# Read off that table rather than restated, so a change to the cost model
# propagates here instead of silently disagreeing with it.
def reads_per_usable_cell() -> dict[str, float]:
    df = pd.read_csv(osp.join(RESULTS, "genome_scale_budgets.csv"))
    df = df[df.cells_per_gene == df.cells_per_gene.iloc[0]]
    return {
        r["platform"]: r["read_pairs_billions"] * 1e9 / r["usable_cells"]
        for _, r in df.iterrows()
    }


PLATFORM_DEPTH = {
    "SPLiT-seq (Brettner, as published)": 410.0,
    "SPLiT-seq + rRNA depletion": 861.0,
    "10x Chromium X (GEM-X 3')": 2000.0,
}
PLATFORM_SHORT = {
    "SPLiT-seq (Brettner, as published)": "SPLiT-seq",
    "SPLiT-seq + rRNA depletion": "SPLiT-seq + depl.",
    "10x Chromium X (GEM-X 3')": "10x Chromium X (and + scifi)",
}

# --- figure ------------------------------------------------------------------
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


def box(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)


def place_panel_letters(fig, axes, letters) -> None:
    """Panel letters in clear whitespace at each panel's true top-left.

    Measured from get_tightbbox rather than offset by a fixed axes fraction, so
    the letter clears the y-label whatever the tick text turns out to be. Same
    routine as plot_economics.py; the house rule is that a crop taken at the
    letter's corner must slice only background.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(r).transformed(inv)
        # Clamped: the layout reserves room for this offset, but a clamp means a
        # letter can never be cropped off the canvas entirely if it does not.
        y = min(bb.y1 + 0.020, 0.985)
        fig.text(max(bb.x0 - 0.012, 0.002), y, letter, fontsize=8,
                 fontweight="bold", ha="left", va="bottom", zorder=20)


# phi is swept, not fixed, because it is the parameter we have not measured. The
# top value is not a guess: it is the phi that the field's own 100-cell floor
# implies, so the sweep brackets "what a bursty yeast gene plausibly does" against
# "what the convention asserts". If the truth is nearer 0.5-2, the 100-cell rule
# is far more conservative than variance alone requires -- which would mean it is
# really protecting against cell-cycle and state confounding, not sampling error.
# Either way the sweep, not a fixed value, is the honest object to plot.
# The MEASURED median response, read from effect_size_analysis.py's output rather
# than assumed. Across all three compendia the median |log2 FC| among genes
# responding to a deletion is ~0.42, i.e. a 1.34-fold change -- not the two-fold
# the rest of the document assumed. Because A goes as Delta^-2 this is the single
# largest correction in the analysis: it multiplies every cell requirement by
# (1/0.42)^2 ~ 5.7.
def measured_delta() -> float:
    df = pd.read_csv(osp.join(RESULTS, "effect_size_summary.csv"))
    return float(df["median_abs_log2fc_responders"].median())


DELTA_MEASURED = measured_delta()

# phi implied by the 100-cell convention, evaluated at BOTH effect sizes. At the
# nominal two-fold it is ~6.1, an implausibly bursty gene; at the measured 1.34-
# fold it is ~1.1, an entirely ordinary single-cell overdispersion. The tension
# flagged in Sec. 4.7 -- "either yeast genes are very bursty or the convention is
# doing something other than variance control" -- dissolves once the effect size
# is measured instead of assumed. The convention was right; the assumed Delta
# was wrong.
PHI_IMPLIED_BY_100_CELL_RULE = 100.0 / (Z**2 / math.log(2.0) ** 2)
PHI_IMPLIED_AT_MEASURED_DELTA = 100.0 / power_coefficient(DELTA_MEASURED)
PHI_SWEEP = [0.5, round(PHI_IMPLIED_AT_MEASURED_DELTA, 1), 6.0]


def panel_a(ax) -> None:
    """n* against depth: one curve, two regimes, and where the knee sits."""
    d = np.logspace(1, 4.3, 300)
    for i, phi in enumerate(PHI_SWEEP):
        n = [cells_per_perturbation(x, phi) for x in d]
        ax.plot(d, n, lw=0.9, color=PLOT_PALETTE[i],
                label=(f"$\\varphi={phi:g}$ (100-cell rule)" if i == 1
                       else f"$\\varphi={phi:g}$"))
        floor = power_coefficient(DELTA_MEASURED) * phi
        ax.axhline(floor, color=PLOT_PALETTE[i], lw=0.4, ls=":")
        # Labelled at the LEFT edge: at this end of the axis every curve is far
        # above its own floor, so the three labels sit in clear space.
        # Left edge: at d=10 every curve is ~2 decades above its own floor, so
        # this strip is empty. At the right edge the labels landed on the curve
        # endpoints they exist to distinguish.
        ax.text(11.5, floor * 1.13, f"{floor:.0f} cells", fontsize=4.6,
                color=PLOT_PALETTE[i], ha="left", va="bottom")
        ds = depth_sufficiency(phi)
        if d[0] < ds < d[-1]:
            ax.plot([ds], [cells_per_perturbation(ds, phi)], marker="o", ms=2.5,
                    color=PLOT_PALETTE[i], markeredgecolor="black",
                    markeredgewidth=0.4, zorder=5)

    # Where the two live platforms actually sit on this axis.
    for depth, name in ((410, "SPLiT-seq"), (2000, "10x v3")):
        # vlines, not axvline: a full-height rule runs straight through the
        # legend in the upper right.
        ax.vlines(depth, 30, 2.0e4, color="#666666", lw=0.4, ls="--")
        ax.text(depth * 1.12, 38.0, name, fontsize=5, color="#666666",
                rotation=90, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e1, 2e4)
    ax.set_ylim(30, 1e5)
    # $d$ dropped from the axis label rather than fixed: a mathtext block
    # mid-phrase eats the following space ("depth d(mRNA"). The symbol is
    # defined in the caption and in the panel title, which ends in mathtext
    # and is therefore safe.
    ax.set_xlabel("Sequencing depth (mRNA UMIs per cell)")
    ax.set_ylabel("Cells per perturbation, $n^{*}$")
    ax.set_title("Depth stops buying precision at $d^{*}$", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="upper right", fontsize=5, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def panel_b(ax) -> None:
    """The current max() of two rules against the single NB curve."""
    d = np.logspace(1, 4.3, 300)
    phi = PHI_IMPLIED_AT_MEASURED_DELTA
    A = power_coefficient(DELTA_MEASURED)

    smooth = [cells_per_perturbation(x, phi) for x in d]
    # What Secs. 4.3/4.5 do today: the larger of a pure shot-noise requirement
    # and a flat 100-cell floor.
    piecewise = [max(A / (x * P_TYPICAL), 100.0) for x in d]

    ax.plot(d, piecewise, lw=0.9, ls="--", color=PLOT_PALETTE[1],
            label="max(shot noise, 100-cell floor)")
    ax.plot(d, smooth, lw=0.9, color=PLOT_PALETTE[0],
            label=f"Eq. (1), $\\varphi={phi:g}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e1, 2e4)
    ax.set_ylim(30, 1e5)
    # $d$ dropped from the axis label rather than fixed: a mathtext block
    # mid-phrase eats the following space ("depth d(mRNA"). The symbol is
    # defined in the caption and in the panel title, which ends in mathtext
    # and is therefore safe.
    ax.set_xlabel("Sequencing depth (mRNA UMIs per cell)")
    ax.set_ylabel("Cells per perturbation, $n^{*}$")
    ax.set_title("The two rules are one curve", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="lower left", fontsize=5, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def panel_c(ax) -> None:
    """Total cells against plex, for main effects and for pairs, by panel size."""
    ks = np.arange(1, 11)
    n_star = cells_per_perturbation(410.0, PHI_IMPLIED_AT_MEASURED_DELTA)  # split-pool depth, phi = 3

    series = [
        # Colour by panel size, linestyle by interaction order. Four distinct
        # colors put two yellows side by side; two colors x two dash patterns
        # reads at a glance and spends less of the palette.
        (6000, 1, PLOT_PALETTE[0], "-", "main effects, $T=6{,}000$"),
        (6000, 2, PLOT_PALETTE[0], "--", "all pairs, $T=6{,}000$"),
        (200, 1, PLOT_PALETTE[1], "-", "main effects, $T=200$"),
        (200, 2, PLOT_PALETTE[1], "--", "all pairs, $T=200$"),
    ]
    for T, r, color, ls, lab in series:
        y = [total_cells(n_star, T, int(k), order=r) for k in ks]
        ax.plot(ks, y, lw=0.9, ls=ls, color=color, marker="o", ms=2.2,
                markeredgecolor="black", markeredgewidth=0.3, label=lab)

    # One split-pool run is ~480,000 cells; anything above a few runs is not a
    # budget question but a feasibility one.
    ax.axhline(4.8e5, color="#666666", lw=0.5, ls=":")
    # Clear of both the rule it annotates and the k=1 end of the nearest
    # curve; on an eight-decade axis a "just above" offset is invisible.
    # Sits in the empty band between the T=200 main-effects curve below and
    # the rule it annotates above, clear of the left spine and of the k=1
    # markers.
    ax.text(5.9, 1.4e5, "one protocol run", fontsize=5, color="#666666",
            ha="left", va="center")
    # The droplet unit of batch, for comparison on the same axis. Without it
    # this panel implicitly prices feasibility in split-pool runs only, and the
    # two units are a factor of three apart rather than the order of magnitude
    # the un-preindexed 20,000-cell channel would suggest.
    ax.axhline(MD.SCIFI_RECOVERED_LARGE_RUN, color="#666666", lw=0.5, ls=":")
    ax.text(5.9, 4.4e4, "one preindexed channel", fontsize=5, color="#666666",
            ha="left", va="center")

    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(1e4, 1e13)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.set_xlabel("Guides per cell, $k$")
    ax.set_ylabel("Total cells needed, $N^{*}$")
    ax.set_title("Panel size beats plex", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="upper right", fontsize=5, handlelength=1.4,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.3)
    box(ax)


# Reference effect sizes, drawn on every Delta panel so the three are readable
# against each other: a two-fold change is the nominal target Sec. 4.3 uses, and
# 1.25-fold is nearer what a regulatory response actually looks like.
DELTA_REFS = [(1.0, "2-fold"), (math.log2(1.25), "1.25-fold")]


def _delta_axis(ax, label_x: float | None = None) -> None:
    """Reference effect sizes, with the label position optional.

    Defaults to the right edge. Panel (e) overrides it: that panel carries a
    vertical "genome scale" rule near the right edge, and a right-aligned label
    lands in the narrow strip beside it.
    """
    x = ax.get_xlim()[1] if label_x is None else label_x
    for dv, lab in DELTA_REFS:
        ax.axhline(dv, color="#666666", lw=0.4, ls=":")
        ax.text(x, dv * 1.06, lab, fontsize=5, color="#666666",
                ha="right", va="bottom")


def panel_d(ax) -> None:
    """Smallest resolvable effect against cells per perturbation, by platform."""
    n = np.logspace(1, 4, 200)
    phi = PHI_IMPLIED_AT_MEASURED_DELTA
    for i, (plat, depth) in enumerate(PLATFORM_DEPTH.items()):
        y = [min_detectable_delta(x, depth, phi) for x in n]
        ax.plot(n, y, lw=0.9, color=PLOT_PALETTE[i], label=PLATFORM_SHORT[plat])
    # No fourth line for the preindexed droplet, and its absence is the finding
    # rather than an omission: preindexing changes cells per priced channel, not
    # UMIs per cell, so it lies exactly on top of the 10x curve here. Every
    # statistical panel of this figure is blind to it. What it changes is the
    # cost of reaching a given n, which is Fig. 9.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e1, 1e4)
    ax.set_ylim(0.1, 10)
    _delta_axis(ax)
    ax.set_xlabel("Cells per perturbation, $n$")
    # Mathtext ONLY at the end of the string: two blocks mid-phrase lost
    # both adjacent spaces and rendered as "|Delta 0g)".
    ax.set_ylabel("Smallest resolvable effect, $|\\Delta|$")
    ax.set_title("What a given cell count can see", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="lower left", fontsize=5, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def panel_e(ax) -> None:
    """Effect-size resolution against panel size, for the two lane budgets."""
    rpc = reads_per_usable_cell()
    T = np.logspace(1.3, 3.9, 200)
    phi = PHI_IMPLIED_AT_MEASURED_DELTA
    plat = "SPLiT-seq + rRNA depletion"
    for i, (name, lanes, _usd) in enumerate(SCENARIOS):
        cells = LANE_READ_PAIRS * lanes / rpc[plat]
        y = [min_detectable_delta(cells / t, PLATFORM_DEPTH[plat], phi) for t in T]
        ax.plot(T, y, lw=0.9, color=PLOT_PALETTE[i],
                label=f"{name.split(' (')[0]}, {cells/1e6:.2f}M cells")
    ax.axvline(6000, color="#666666", lw=0.4, ls="--")
    # Horizontal and at the top, not rotated at the right edge, where it
    # ran into the "1.25-fold" reference label.
    ax.text(5600, 6.0, "genome scale", fontsize=5, color="#666666",
            ha="right", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 8e3)
    ax.set_ylim(0.1, 10)
    _delta_axis(ax, label_x=5200)
    ax.set_xlabel("Target genes in the panel, $T$")
    # Mathtext ONLY at the end of the string: two blocks mid-phrase lost
    # both adjacent spaces and rendered as "|Delta 0g)".
    ax.set_ylabel("Smallest resolvable effect, $|\\Delta|$")
    ax.set_title("One lane is a panel; a flow cell is a genome",
                 loc="left", fontsize=6)
    ax.legend(frameon=False, loc="upper left", fontsize=5, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def panel_f(ax) -> None:
    """Multiplexing converts a fixed lane budget into effect-size resolution."""
    rpc = reads_per_usable_cell()
    ks = np.arange(1, 11)
    phi = PHI_IMPLIED_AT_MEASURED_DELTA
    plat = "SPLiT-seq + rRNA depletion"
    for i, (name, lanes, _usd) in enumerate(SCENARIOS):
        cells = LANE_READ_PAIRS * lanes / rpc[plat]
        y = [min_detectable_delta(cells * k / 6000, PLATFORM_DEPTH[plat], phi)
             for k in ks]
        ax.plot(ks, y, lw=0.9, color=PLOT_PALETTE[i], marker="o", ms=2.2,
                markeredgecolor="black", markeredgewidth=0.3, label=name)
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.1, 10)
    _delta_axis(ax)
    ax.set_xlabel("Guides per cell, $k$")
    # Mathtext ONLY at the end of the string: two blocks mid-phrase lost
    # both adjacent spaces and rendered as "|Delta 0g)".
    ax.set_ylabel("Smallest resolvable effect, $|\\Delta|$")
    ax.set_title("Plex at genome scale, $T=6{,}000$", loc="left", fontsize=6)
    ax.legend(frameon=False, loc="lower left", fontsize=5, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.2)
    box(ax)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    style()
    # 2x3: the top row is the structure of the equation, the bottom row is what
    # it says for the two sequencing configurations actually on offer. Height is
    # 112 mm, inside the 170 mm cap.
    fig, axes = plt.subplots(
        2, 3, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(112.0))
    )
    flat = axes.ravel()
    for fn, ax in zip((panel_a, panel_b, panel_c, panel_d, panel_e, panel_f), flat):
        fn(ax)
    fig.tight_layout(pad=0.4, w_pad=2.6, h_pad=2.4, rect=(0.012, 0.0, 1.0, 0.955))
    place_panel_letters(fig, flat, ["a", "b", "c", "d", "e", "f"])
    # Legibility gate; see figure_checks.py. Six panels, and every reference
    # line in them is labelled by hand at a position that depends on the axis
    # limits, so a limit change is exactly what this catches.
    assert_legible(
        fig, axes=list(flat),
        exempt={"one protocol run", "one preindexed channel", "genome scale",
                "2-fold", "1.25-fold"},
    )

    out = osp.join(OUT_DIR, "design_equation.svg")
    savefig_true_size_svg(fig, out)
    print(f"wrote {out}")

    A = power_coefficient(1.0)
    print(f"\nA(1.0)              = {A:.1f}")
    print(f"p_typical           = {P_TYPICAL:.2e}")
    Am = power_coefficient(DELTA_MEASURED)
    for phi in PHI_SWEEP:
        print(f"phi={phi:<5g} floor n = {Am*phi:6.0f}   "
              f"d* = {depth_sufficiency(phi):8.0f}")
    rpc = reads_per_usable_cell()
    print("\nusable cells per scenario (SPLiT-seq + rRNA depletion):")
    for name, lanes, usd in SCENARIOS:
        c = LANE_READ_PAIRS * lanes / rpc["SPLiT-seq + rRNA depletion"]
        d_gs = min_detectable_delta(c / 6000, 861.0, PHI_IMPLIED_AT_MEASURED_DELTA)
        print(f"  {name:26s} {c:>10,.0f} cells  ${usd*lanes:>8,.0f}  "
              f"genome-scale |Delta| = {d_gs:.2f} log2 ({2**d_gs:.1f}-fold)")
    print(f"\nmeasured median |log2 FC| among responders: {DELTA_MEASURED:.3f} "
          f"({2**DELTA_MEASURED:.2f}-fold)")
    print(f"A at measured Delta                       : "
          f"{power_coefficient(DELTA_MEASURED):.1f}  (vs {A:.1f} at two-fold)")
    print(f"phi implied by 100-cell rule, two-fold    : "
          f"{PHI_IMPLIED_BY_100_CELL_RULE:.2f}")
    print(f"phi implied by 100-cell rule, measured    : "
          f"{PHI_IMPLIED_AT_MEASURED_DELTA:.2f}")
    print(f"shot-noise limit K = A = {A:.1f}  (Sec. 4.3 Eq. gives "
          f"{(Z/math.log(2.0))**2:.1f})")


if __name__ == "__main__":
    main()
