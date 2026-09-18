# experiments/008-xue-ffa/scripts/perspective_figure_panels.py
# [[experiments.008-xue-ffa.scripts.perspective_figure_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/perspective_figure_panels
#
# Every publication panel for the epistasis-in-metabolic-engineering perspective, built to
# the repo figure standard: torchcell palette, Arial 6 pt, boxed axes, one standard panel
# width per file, true-size SVG. Each panel is written as its OWN file with NO panel letter
# and NO shared legend, because the arrangement and the a/b/c lettering are draw.io
# decisions and baking either one in here removes the flexibility that storyboarding needs.
#
# The script reads only committed result CSVs under experiments/008-xue-ffa/results/. It
# fits nothing and estimates nothing; the four epistasis models and the path enumeration
# are upstream, in free_fatty_acid_interactions.py, additive_free_fatty_acid_interactions.py,
# glm_log_link_epistatic_interactions.py, log_ols_wt_differencing_epistatic_interactions.py,
# ffa_epistatic_path_panels.py and linear_vs_log_scale_sign_test.py. Keeping the panel code
# separate from the model code is what lets a restyle happen without a refit.
#
# WHICH READOUT. Total FFA titer unless a panel says otherwise. It is the quantity a
# metabolic engineer optimizes, and it is the only readout whose label agrees across all
# four model families on disk (see canonical_readout in notable_interaction_selection.py).
#
# STATISTICS, stated once so no caption has to restate it:
#   - tau is the Kuzmin tau-SGA trigenic score on the linear scale, epsilon the digenic one.
#   - p-values come from a t test on the delta-method standard error, referred to the
#     Welch-Satterthwaite effective df of the linear combination (median 4.28 here). The
#     earlier df = min(n) - 1 was wrong and suppressed nearly every call.
#   - FDR is Benjamini-Hochberg. The stored fdr_corrected_p pools all six readouts
#     (714 tests); panels that claim over one readout recompute BH within that readout.

import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator, MultipleLocator
from statsmodels.stats.multitest import multipletests

import torchcell
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
plt.style.use(osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle"))
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.7,
        "savefig.bbox": "standard",
        # Math text in Arial as well, so a panel ships one typeface (PAPER_RC does the
        # same repo-wide; restated here because this script sets its own rc block).
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)

EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
ENRICH_DIR = osp.join(RESULTS_DIR, "graph_enrichment")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")

TOTAL = "Total Titer"
MODELS = {
    "multiplicative": "multiplicative_trigenic_interactions_3_delta_normalized.csv",
    "additive": "additive_trigenic_interactions_3_delta_normalized.csv",
    "glm_log_link": "glm_log_link/glm_log_link_trigenic_interactions.csv",
    "log_ols": "glm_models/log_ols_trigenic_interactions.csv",
}
MODEL_LABELS = {
    "multiplicative": "multiplicative",
    "additive": "additive",
    "glm_log_link": "GLM log-link",
    "log_ols": "log-OLS",
}
MODEL_COLORS = {
    # The same four as epistasis_model_intuition_panels: multiplicative blue, additive
    # brick, so the two nulls every panel contrasts are the pair that separates best;
    # log-OLS takes the orange (author review, 2026.09.18).
    "multiplicative": PLOT_PALETTE[4],
    "additive": PLOT_PALETTE[1],
    "glm_log_link": PLOT_PALETTE[2],
    "log_ols": PLOT_PALETTE[0],
}
# Sign encoding, held fixed across every panel in the document: blue = positive
# interaction, brick = negative (review 2026.09.16: positive moved from amber to blue, and
# amber is now free for objects such as the pathway genes in the network figure). Third
# category takes lilac; the two series of the score histogram take amber and lilac, since
# digenic-versus-trigenic is not a sign.
C_POS = PLOT_PALETTE[4]
C_NEG = PLOT_PALETTE[1]
C_THIRD = PLOT_PALETTE[2]
C_DIGENIC = PLOT_PALETTE[2]
C_TRIGENIC = PLOT_PALETTE[0]
# The best-combination line shares an axes with brick bars and nothing else. It was amber,
# the other member of the two-series palette prefix, and amber type on white does not read
# at 6 pt: the right axis label and its tick numbers took the line's color and disappeared
# (author review, 2026.09.18). Blue carries the same series at a lightness that prints;
# lilac was tried and orange before it, and neither was kept.
C_BEST = PLOT_PALETTE[4]
# The base strain a campaign starts from, as the pale companion of the campaign's brick.
# The star that marks the best strain takes the same pale fill: the two endpoints of the
# panel are one kind of thing, and a blue star was a third color on a brick panel.
C_START = PLOT_PALETTE_FILL[1]
C_GRAY = PLOT_PALETTE[5]

# Reserved band above the tallest bar, so a value label never lands on the title and an
# in-axes legend never lands on a bar.
LABEL_HEADROOM = 1.30

# The ten interaction graphs the enrichment test scans, in the order they are reported.
GRAPH_LABELS = {
    "physical": "physical",
    "regulatory": "regulatory",
    "genetic": "genetic",
    "tflink": "TFLink",
    "string12_0_neighborhood": "STRING neighborhood",
    "string12_0_fusion": "STRING fusion",
    "string12_0_cooccurence": "STRING co-occurrence",
    "string12_0_coexpression": "STRING coexpression",
    "string12_0_experimental": "STRING experimental",
    "string12_0_database": "STRING database",
}


def canonical_readout(label):
    """`C14:0` in the linear-scale tables, `C140` in the regression tables."""
    return str(label).replace(":", "").replace(" ", "_")


def canonical_triple(gene_set):
    """`RPD3_SPT3_YAP6` in one model family, `FKH1:GCN5:MED4` in the other."""
    return "-".join(sorted(re.split(r"[_:|-]", str(gene_set))))


def load_model(rel, readout=TOTAL):
    """One model's trigenic table for one readout, with BH recomputed within it."""
    df = pd.read_csv(osp.join(RESULTS_DIR, rel))
    df = df[
        (df["ffa_type"].map(canonical_readout) == canonical_readout(readout))
        & df["p_value"].notna()
    ].copy()
    df["triple"] = df["gene_set"].map(canonical_triple)
    reject, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh", alpha=0.05)
    df["fdr_within"] = fdr
    df["fdr_sig"] = reject
    return df


def style_axes(ax):
    ax.grid(axis="both", which="major", color="0.88", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")


def bar_headroom(ax, totals):
    ymax = max(totals) if len(totals) else 1
    ax.set_ylim(0, ymax * LABEL_HEADROOM)
    return ymax


# ---------------------------------------------------------------------------------------
# The interaction landscape
# ---------------------------------------------------------------------------------------

def panel_interaction_distribution(ax):
    """Digenic and trigenic interaction scores on total titer, on one axis.

    Both are deviations from the same multiplicative null, so they are directly
    comparable, and the comparison is the point: adding a third deletion does not damp the
    interaction, it widens it.
    """
    di = pd.read_csv(
        osp.join(RESULTS_DIR, "multiplicative_digenic_interactions_3_delta_normalized.csv")
    )
    tri = pd.read_csv(
        osp.join(RESULTS_DIR, "multiplicative_trigenic_interactions_3_delta_normalized.csv")
    )
    di = di[di["ffa_type"] == TOTAL]
    tri = tri[tri["ffa_type"] == TOTAL]

    lo = min(di["interaction_score"].min(), tri["interaction_score"].min())
    hi = max(di["interaction_score"].max(), tri["interaction_score"].max())
    bins = np.linspace(lo, hi, 26)
    # Two outline (step) histograms in two colors, the form the repo uses to compare
    # distributions across datasets (e.g. the 025 recapitulation residuals): the two
    # distributions overlap heavily, and an outline in each color shows both through the
    # overlap where a fill-versus-hatch pair (the earlier form) hid one behind the other.
    for sub, color, label in (
        (di, C_DIGENIC, f"digenic $\\varepsilon$ (n={len(di)})"),
        (tri, C_TRIGENIC, f"trigenic $\\tau$ (n={len(tri)})"),
    ):
        ax.hist(sub["interaction_score"], bins=bins, histtype="step", color=color,
                linewidth=0.9, label=label, zorder=3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel("interaction score (total FFA titer)")
    ax.set_ylabel("gene combinations")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.28)
    ax.set_title(
        "adding a third deletion moves the median\n"
        f"from {di['interaction_score'].median():+.2f} to "
        f"{tri['interaction_score'].median():+.2f}",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.3,
              borderaxespad=0.3)
    return {"digenic_n": len(di), "trigenic_n": len(tri)}


def panel_volcano(ax):
    """Trigenic tau against evidence, with the within-readout BH threshold drawn.

    The FDR line is drawn at the largest p that BH rejects rather than at a fixed alpha,
    because BH has no fixed p threshold: the cutoff depends on the whole p distribution.
    """
    tri = load_model(MODELS["multiplicative"])
    sig = tri[tri["fdr_sig"]]
    p_cut = sig["p_value"].max() if len(sig) else np.nan

    pos = tri[tri["interaction_score"] > 0]
    neg = tri[tri["interaction_score"] <= 0]
    for sub, color, label in (
        (neg, C_NEG, f"negative $\\tau$ (n={len(neg)})"),
        (pos, C_POS, f"positive $\\tau$ (n={len(pos)})"),
    ):
        ax.scatter(sub["interaction_score"], -np.log10(sub["p_value"]), s=5,
                   facecolor=color, edgecolor="black", linewidth=0.2, zorder=3,
                   label=label)
    if np.isfinite(p_cut):
        ax.axhline(-np.log10(p_cut), color="black", linewidth=0.5, linestyle="--",
                   zorder=2)
        # Below the line at the left: above it, the label sat on the cloud of called
        # negatives, and below it on the right it sat on the uncalled positives. Every
        # uncalled triple has |tau| < 0.7, so the region below the line at the far left
        # is empty. The extra two points of drop keep the cap heights of the text clear
        # of the dashes rather than touching them.
        ax.annotate(" BH FDR < 0.05", xy=(0.02, -np.log10(p_cut)),
                    xycoords=ax.get_yaxis_transform(), xytext=(0, -2),
                    textcoords="offset points", va="top", ha="left", fontsize=5)
    ax.axvline(0, color=C_GRAY, linewidth=0.4, zorder=1)
    ax.set_xlabel("$\\tau$ (trigenic interaction)")
    ax.set_ylabel("$-\\log_{10}$ $P$")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.24)
    n_sig_pos = int((sig["interaction_score"] > 0).sum())
    ax.set_title(
        f"{len(sig)}/{len(tri)} trigenic interactions at FDR < 0.05\n"
        f"{n_sig_pos} of them positive",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.3)
    return {"n_sig": len(sig), "n_total": len(tri), "n_sig_pos": n_sig_pos,
            "p_cut": float(p_cut)}


# ---------------------------------------------------------------------------------------
# Model dependence
# ---------------------------------------------------------------------------------------

def panel_model_agreement(ax_top, ax_dot):
    """Which models call which interactions, as an intersection plot.

    Replaces the stock UpSet render, which arrives in pure black with 20 pt type and no
    relation to the palette. Same data and same reading order; only the drawing changes.
    """
    calls = {}
    for name, rel in MODELS.items():
        m = load_model(rel)
        calls[name] = set(m.loc[m["fdr_sig"], "triple"])
    universe = sorted(set().union(*calls.values()))

    combos = {}
    for t in universe:
        key = tuple(name in calls and t in calls[name] for name in MODELS)
        combos.setdefault(key, []).append(t)
    order = sorted(combos, key=lambda k: -len(combos[k]))

    x = np.arange(len(order))
    sizes = [len(combos[k]) for k in order]
    ax_top.bar(x, sizes, color=C_NEG, edgecolor="black", linewidth=0.4, width=0.62)
    ymax = bar_headroom(ax_top, sizes)
    for xi, s in zip(x, sizes):
        ax_top.text(xi, s + ymax * 0.03, str(s), ha="center", fontsize=5)
    ax_top.set_ylabel("interactions")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.set_xlim(-0.6, len(order) - 0.4)

    names = list(MODELS)
    for row, name in enumerate(names):
        y = len(names) - 1 - row
        for xi, key in enumerate(order):
            on = key[row]
            ax_dot.scatter([xi], [y], s=14, zorder=3,
                           facecolor=MODEL_COLORS[name] if on else "0.86",
                           edgecolor="black" if on else "0.7", linewidth=0.25)
        ons = [xi for xi, key in enumerate(order) if key[row]]
        del ons
    for xi, key in enumerate(order):
        ys = [len(names) - 1 - r for r, on in enumerate(key) if on]
        if len(ys) > 1:
            ax_dot.plot([xi, xi], [min(ys), max(ys)], color="black", linewidth=0.5,
                        zorder=2)
    ax_dot.set_yticks(range(len(names)))
    ax_dot.set_yticklabels([MODEL_LABELS[n] for n in reversed(names)])
    ax_dot.set_xticks(x)
    ax_dot.set_xticklabels([])
    ax_dot.set_xlim(-0.6, len(order) - 0.4)
    ax_dot.set_ylim(-0.6, len(names) - 0.4)
    ax_dot.set_xlabel("models calling the interaction at FDR < 0.05")
    return {"n_union": len(universe), "n_all_four": len(combos.get((True,) * 4, []))}


def panel_scale_scatter(ax):
    """tau on the linear scale against tau on the log scale, one point per triple."""
    st = pd.read_csv(osp.join(RESULTS_DIR, "linear_vs_log_scale_sign_test.csv"))
    groups = [
        ("negative_both", C_NEG, "negative on both"),
        ("positive_both", C_POS, "positive on both"),
        ("positive_to_nonpositive", C_THIRD, "positive $\\rightarrow$ non-positive"),
    ]
    for key, color, label in groups:
        sub = st[st["sign_class"] == key]
        ax.scatter(sub["tau_linear"], sub["tau_log"], s=6, facecolor=color,
                   edgecolor="black", linewidth=0.2, zorder=3,
                   label=f"{label} (n={len(sub)})")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xlabel("$\\tau$ linear scale (multiplicative)")
    ax.set_ylabel("$\\tau$ log scale (GLM log-link)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.34)
    ax.set_title(
        "the upper-left quadrant is empty:\nno negative ever becomes positive",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.3)
    return {k: int((st["sign_class"] == k).sum()) for k, _, _ in groups}


def panel_scale_slope(ax):
    """Every linear-scale positive, carried onto the log scale."""
    st = pd.read_csv(osp.join(RESULTS_DIR, "linear_vs_log_scale_sign_test.csv"))
    pos = st[st["tau_linear"] > 0]
    for _, row in pos.iterrows():
        keeps = row["tau_log"] > 0
        ax.plot([0, 1], [row["tau_linear"], row["tau_log"]],
                color=C_POS if keeps else C_THIRD, linewidth=0.5, zorder=2)
        ax.scatter([0, 1], [row["tau_linear"], row["tau_log"]], s=5,
                   facecolor=C_POS if keeps else C_THIRD, edgecolor="black",
                   linewidth=0.2, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["linear", "log"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("$\\tau$")
    n_lost = int((pos["tau_log"] <= 0).sum())
    ax.set_title(
        f"every triple positive on the linear scale\n"
        f"{n_lost} of {len(pos)} lose that sign on the log scale",
        fontsize=6, pad=3,
    )
    return {"n_pos_linear": len(pos), "n_lost": n_lost}


# ---------------------------------------------------------------------------------------
# The engineering consequence
# ---------------------------------------------------------------------------------------

def panel_path_accessibility(ax):
    """The thesis panel: how many triples a stepwise campaign could actually have found.

    A path is one of the six orders in which a triple's three deletions can be introduced.
    It is monotone when every intermediate strain is at least as good as the one before it,
    which is what a greedy build-and-screen campaign requires: an engineer who screens
    after each round keeps only what improved, so a route through a worse intermediate is
    never taken. Of the 120 triples, 51 beat the base strain; only 2 are reachable by any
    monotone route.
    """
    acc = pd.read_csv(osp.join(RESULTS_DIR, "ffa_epistatic_path_accessibility.csv"))
    beats = acc["f_triple"] > 1.0
    reachable = beats & (acc["n_monotone"] >= 1)
    cats = [
        ("does not beat\nbase strain", int((~beats).sum()), C_GRAY),
        ("beats base,\nno monotone route", int((beats & ~reachable).sum()), C_NEG),
        ("beats base,\nreachable", int(reachable.sum()), C_POS),
    ]
    x = np.arange(len(cats))
    heights = [c[1] for c in cats]
    ax.bar(x, heights, color=[c[2] for c in cats], edgecolor="black", linewidth=0.4,
           width=0.62)
    ymax = bar_headroom(ax, heights)
    for xi, h in zip(x, heights):
        ax.text(xi, h + ymax * 0.03, str(h), ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in cats])
    ax.set_ylabel("triple deletions")
    ax.set_title(
        f"{int(reachable.sum())} of {int(beats.sum())} improving triples\n"
        "sit at the end of a monotone route",
        fontsize=6, pad=3,
    )
    return {"n_triples": len(acc), "n_beats": int(beats.sum()),
            "n_reachable": int(reachable.sum())}


def _combination_titers():
    """Every measured strain's normalized titer, keyed by its frozenset of deletions.

    The path table stores one row per (triple, order) with that order's rungs, so the same
    single and double strains appear many times over. Collapsing on the gene set recovers
    the 10 + 45 + 120 distinct strains the design actually contains.
    """
    p = pd.read_csv(osp.join(RESULTS_DIR, "ffa_epistatic_paths.csv"))
    singles = {r.gene_1: r.f_single for r in p.drop_duplicates("gene_1").itertuples()}
    doubles = {frozenset([r.gene_1, r.gene_2]): r.f_double for r in p.itertuples()}
    triples = {frozenset(r.triple.split("-")): r.f_triple
               for r in p.drop_duplicates("triple").itertuples()}
    assert len(singles) == 10 and len(doubles) == 45 and len(triples) == 120, (
        f"design is not complete: {len(singles)}/{len(doubles)}/{len(triples)}")
    return singles, doubles, triples


def _greedy_walk(singles, doubles, triples):
    """The strain a stepwise campaign reaches, taking the best strictly improving step.

    One deletion per round, always the best available, stopping when no remaining deletion
    improves on the strain in hand. This is the mildest possible formalization of
    build-and-screen: it is allowed to see every candidate at every round, which no real
    campaign is, and it still stops early.
    """
    # `singles` is keyed by gene name, the other two by frozenset; key everything the same
    # way so one lookup covers all three rounds.
    by_set = {frozenset([g]): v for g, v in singles.items()}
    by_set.update(doubles)
    by_set.update(triples)
    current, f = frozenset(), 1.0
    trace = [(current, f)]
    while len(current) < 3:
        options = [(by_set[current | {g}], g) for g in singles
                   if g not in current and (current | {g}) in by_set]
        best_f, best_g = max(options)
        if best_f <= f:
            break
        current, f = current | {best_g}, best_f
        trace.append((current, f))
    return trace


def panel_improving_by_order(ax):
    """What fraction of combinations beats the base strain, at one, two and three deletions.

    The shape is the obstacle: single deletions almost all lose titer, so the first step of
    a stepwise campaign is nearly always downhill even though half the doubles and 4 in 10
    of the triples are improvements.
    """
    singles, doubles, triples = _combination_titers()
    tables = [("1 deletion", singles), ("2 deletions", doubles), ("3 deletions", triples)]
    x = np.arange(len(tables))
    fracs, labels = [], []
    for _, t in tables:
        n_up = sum(v > 1.0 for v in t.values())
        fracs.append(100.0 * n_up / len(t))
        labels.append(f"{n_up}/{len(t)}")
    ax.bar(x, fracs, color=C_NEG, edgecolor="black", linewidth=0.4, width=0.62)
    ymax = bar_headroom(ax, fracs)
    for xi, (fr, lab) in enumerate(zip(fracs, labels)):
        ax.text(xi, fr + ymax * 0.03, lab, ha="center", fontsize=5)
    best = [max(t.values()) for _, t in tables]
    ax2 = ax.twinx()
    ax2.plot(x, best, color=C_BEST, marker="o", markersize=2.6, linewidth=0.7,
             markeredgecolor="black", markeredgewidth=0.25, zorder=4,
             label="best combination")
    ax2.set_ylabel("best titer (rel. base strain)", color=C_BEST)
    ax2.tick_params(axis="y", colors=C_BEST, labelsize=6)
    ax2.set_ylim(0.9, max(best) * 1.30)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in tables])
    ax.set_ylabel("combinations beating the base strain (%)")
    ax.set_title(
        f"{labels[0]} single deletions improve titer,\n"
        f"but the best triple reaches {max(best):.2f}$\\times$",
        fontsize=6, pad=3,
    )
    # Upper right: the reserved headroom leaves that band clear, while lower right sits
    # inside the three-deletion bar.
    ax2.legend(loc="upper right", frameon=False, handlelength=1.2, fontsize=5,
               borderaxespad=0.3)
    return {"fracs": [round(f, 1) for f in fracs], "labels": labels,
            "best": [round(b, 3) for b in best]}


def panel_greedy_walk(ax):
    """Where a stepwise campaign stops, next to the best strain the design contains.

    Gray lines are all six deletion orders of the highest-titer triple. Every one of them
    passes through an intermediate below the base strain, so a campaign that keeps only
    improvements can never take any of them.
    """
    singles, doubles, triples = _combination_titers()
    trace = _greedy_walk(singles, doubles, triples)
    paths = pd.read_csv(osp.join(RESULTS_DIR, "ffa_epistatic_paths.csv"))
    best_set, best_f = max(triples.items(), key=lambda kv: kv[1])
    best_name = "-".join(sorted(best_set))

    rows = paths[paths["triple"] == best_name]
    for i, r in enumerate(rows.itertuples()):
        ax.plot([0, 1, 2, 3], [r.f_base, r.f_single, r.f_double, r.f_triple],
                color=C_GRAY, linewidth=0.6, marker="o", markersize=2.2,
                markerfacecolor="white", markeredgecolor=C_GRAY, markeredgewidth=0.4,
                zorder=2, label="routes to the best triple" if i == 0 else None)

    xs = [len(s) for s, _ in trace]
    ys = [f for _, f in trace]
    ax.plot(xs, ys, color=C_NEG, linewidth=1.1, marker="o", markersize=3.4,
            markeredgecolor="black", markeredgewidth=0.3, zorder=4,
            label="greedy campaign")
    ax.scatter([xs[-1]], [ys[-1]], s=26, marker="X", color=C_NEG, edgecolor="black",
               linewidth=0.3, zorder=5)
    # Where the campaign starts, in the pale companion of the campaign's own brick: the
    # base strain is a member of that series rather than a third thing, and filled brick
    # it read as one more step of the walk.
    ax.scatter([xs[0]], [ys[0]], s=22, marker="o", facecolor=C_START, edgecolor=C_NEG,
               linewidth=0.6, zorder=6, label="base strain")
    # Above the marker, not beside it: beside, the text ran into the X it names.
    ax.annotate("stops here", (xs[-1], ys[-1]), fontsize=5, va="bottom", ha="center",
                xytext=(0, 5), textcoords="offset points")
    # The star is the destination the panel is about, so it is drawn larger than the X that
    # marks where the campaign stops; at equal area the X dominated it.
    ax.scatter([3], [best_f], s=52, marker="*", facecolor=C_START, edgecolor=C_NEG,
               linewidth=0.6, zorder=5, label="best strain in the design")

    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["base", "1 KO", "2 KO", "3 KO"])
    ax.set_xlim(-0.25, 3.55)
    ax.set_ylabel("titer (rel. base strain)")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.30)
    ax.set_title(
        f"greedy stops at {ys[-1]:.2f}$\\times$; the best strain\n"
        f"reaches {best_f:.2f}$\\times$ only through a loss",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, fontsize=5,
              labelspacing=0.25, borderaxespad=0.3)
    return {"greedy_end": round(ys[-1], 3), "greedy_genes": sorted(trace[-1][0]),
            "best_triple": best_name, "best_f": round(best_f, 3),
            "worst_rung_on_best_routes": round(float(rows[["f_single", "f_double"]]
                                                     .to_numpy().min()), 3)}


def panel_path_valley(ax):
    """How deep the best available route has to dip before it comes back up.

    Valley depth is the shortfall of the worst intermediate on a triple's shallowest route,
    as a fraction of the base strain. It is the quantity a stop-early screening rule pays:
    a campaign that discards any strain below its parent never sees the far side.
    """
    acc = pd.read_csv(osp.join(RESULTS_DIR, "ffa_epistatic_path_accessibility.csv"))
    beats = acc["f_triple"] > 1.0
    bins = np.linspace(0, acc["max_valley_depth"].max() * 1.02, 22)
    ax.hist(acc.loc[~beats, "max_valley_depth"], bins=bins, color=C_GRAY,
            edgecolor="black", linewidth=0.3,
            label=f"does not beat base (n={int((~beats).sum())})")
    ax.hist(acc.loc[beats, "max_valley_depth"], bins=bins,
            bottom=np.histogram(acc.loc[~beats, "max_valley_depth"], bins=bins)[0],
            color=C_NEG, edgecolor="black", linewidth=0.3,
            label=f"beats base (n={int(beats.sum())})")
    med = acc["max_valley_depth"].median()
    ax.axvline(med, color="black", linewidth=0.5, linestyle="--", zorder=3)
    ax.set_xlabel("valley depth of the shallowest route\n(fraction of base strain titer)")
    ax.set_ylabel("triple deletions")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.30)
    ax.set_title(
        f"median route dips {med * 100:.0f}% below the base strain\n"
        "before the third deletion recovers it",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, labelspacing=0.3,
              borderaxespad=0.3)
    return {"median_valley": float(med), "max_valley": float(acc["max_valley_depth"].max())}


# ---------------------------------------------------------------------------------------
# Can any existing network predict the interactions?
# ---------------------------------------------------------------------------------------

def panel_graph_enrichment(ax):
    """Fold enrichment of each interaction graph among the significant trigenic triples.

    A triple counts as overlapping a graph when its three genes are connected in that
    graph. Fold enrichment is the ratio of the overlapping fraction among significant
    triples to the overlapping fraction among the rest, over all six readouts pooled at
    p < 0.05, which is the family the enrichment scripts test. A value of 1 means the graph
    carries no information about which triples interact.
    """
    frames = {}
    for name in MODELS:
        p = osp.join(ENRICH_DIR, f"{name}_trigenic_connected_enrichment.csv")
        d = pd.read_csv(p)
        d["key"] = d["graph_type"].str.replace("_connected$", "", regex=True)
        frames[name] = d.set_index("key")

    # Graphs where no triple in the study is connected at all carry no test; dropping them
    # is a statement about the graph's coverage of these ten TFs, not about epistasis.
    keys = [k for k in GRAPH_LABELS
            if frames["multiplicative"].loc[k, "fold_enrichment"] == frames[
                "multiplicative"].loc[k, "fold_enrichment"]]
    y = np.arange(len(keys))
    h = 0.19
    for i, name in enumerate(MODELS):
        vals = [frames[name].loc[k, "fold_enrichment"] for k in keys]
        ps = [frames[name].loc[k, "p_value"] for k in keys]
        offs = y + (i - 1.5) * h
        ax.barh(offs, vals, height=h, color=MODEL_COLORS[name], edgecolor="black",
                linewidth=0.3, label=MODEL_LABELS[name], zorder=3)
        for yy, v, pv in zip(offs, vals, ps):
            if pv < 0.05:
                ax.text(v + 0.03, yy, "*", va="center", fontsize=6)
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([GRAPH_LABELS[k] for k in keys])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlabel("fold enrichment among significant trigenic triples")
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    xmax = max(
        frames[n].loc[k, "fold_enrichment"] for n in MODELS for k in keys
    )
    ax.set_xlim(0, xmax * 1.28)
    hits = [(n, k) for n in MODELS for k in keys if frames[n].loc[k, "p_value"] < 0.05]
    n_excess = sum(1 for n, k in hits if frames[n].loc[k, "fold_enrichment"] > 1.0)
    # Derived, never asserted: the direction of the significant tests is the finding, so it
    # is read off the same numbers the bars are drawn from.
    direction = "every one of them a depletion" if n_excess == 0 else (
        f"{n_excess} of them an excess")
    ax.set_title(
        f"{len(hits)} of {len(keys) * len(MODELS)} graph-by-model tests reach $P$ < 0.05,\n"
        f"{direction} (* marks $P$ < 0.05)",
        fontsize=6, pad=3,
    )
    ax.legend(loc="lower right", frameon=False, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.3)
    return {"n_graphs": len(keys), "n_sig_tests": len(hits), "n_excess": n_excess}


# ---------------------------------------------------------------------------------------

def emit(name, width_key, height_mm, draw, nrows=1, height_ratios=None):
    """One panel to one pair of files, at a standard width and true size."""
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]), mm_to_in(height_mm)),
        gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
        sharex=(nrows > 1),
    )
    axes = np.atleast_1d(axes)
    info = draw(*axes)
    for ax in axes:
        style_axes(ax)
    fig.tight_layout(pad=0.4)
    if nrows > 1:
        fig.subplots_adjust(hspace=0.06)
    stem = osp.join(IMAGES_DIR, f"panel_{name}")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)
    print(f"wrote panel_{name}  ({PANEL_WIDTHS_MM[width_key]} x {height_mm} mm)  {info}")
    return info


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    emit("interaction_distribution", "third", 62.0, panel_interaction_distribution)
    emit("volcano_total_titer", "third", 62.0, panel_volcano)
    emit("model_agreement", "half", 66.0, panel_model_agreement, nrows=2,
         height_ratios=[2.1, 1.0])
    emit("scale_scatter", "third", 62.0, panel_scale_scatter)
    emit("scale_slope", "third", 62.0, panel_scale_slope)
    emit("improving_by_order", "third", 62.0, panel_improving_by_order)
    emit("greedy_walk", "third", 62.0, panel_greedy_walk)
    emit("path_accessibility", "third", 62.0, panel_path_accessibility)
    emit("path_valley", "third", 62.0, panel_path_valley)
    emit("graph_enrichment", "half_plus", 66.0, panel_graph_enrichment)


if __name__ == "__main__":
    main()
