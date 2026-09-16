# experiments/008-xue-ffa/scripts/supplementary_figure_panels.py
# [[experiments.008-xue-ffa.scripts.supplementary_figure_panels]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/supplementary_figure_panels
#
# The supplementary panels of the epistasis-in-metabolic-engineering perspective: the
# DIGENIC layer, which the main text passes over in one sentence, and the FIVE SPECIES
# under the summed total titer the main text reports. Same construction as
# perspective_figure_panels.py -- one panel per file, no letters, no shared legend, read
# from committed result CSVs only, fit nothing.
#
# WHY THESE TWO AXES. The main text makes one claim on one readout under one null: the
# third deletion interacts on total titer under the multiplicative model. Both of the
# obvious "does that survive?" questions are answered by tables the four model scripts
# already wrote and nothing plotted:
#
#   - the digenic layer. tau removes the digenic contributions before asking what the
#     third deletion adds (Eq. 2 of the document), so what the PAIRS do is the reference
#     the trigenic claim is made against, and it is the opposite of the trigenic one:
#     43 of 45 pairs are positive on total titer.
#   - the five species. Total titer is a sum, and summing is a modeling choice. C14:0 is
#     the clearest case: every digenic call on it is negative and every trigenic call is
#     positive, so the sign of the interaction depends on which molecule is counted.
#
# STATISTICS, stated once. tau is the Kuzmin linear-scale trigenic score and epsilon the
# digenic one; p-values are a t test on the delta-method standard error referred to the
# Welch-Satterthwaite effective df; FDR is Benjamini-Hochberg recomputed WITHIN each
# readout, which is the family a per-readout count claims over (the stored
# fdr_corrected_p pools all six).

import os
import os.path as osp
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.multitest import multipletests

import torchcell
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
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
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)

EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")

TOTAL = "Total Titer"
# Species in chain-length order, the order a lipid reader expects, with the summed total
# last because it is derived from the five.
READOUTS = ["C14:0", "C16:0", "C18:0", "C16:1", "C18:1", TOTAL]
READOUT_LABELS = {r: ("total" if r == TOTAL else r) for r in READOUTS}
TF_GENES = ["FKH1", "GCN5", "MED4", "OPI1", "RFX1", "RGR1", "RPD3", "SPT3", "TFC7", "YAP6"]

DIGENIC_CSV = "multiplicative_digenic_interactions_3_delta_normalized.csv"
TRIGENIC_CSV = "multiplicative_trigenic_interactions_3_delta_normalized.csv"

# Sign encoding, the same as the main figures: blue positive, brick negative. Order
# (digenic versus trigenic) is not a sign, so it takes amber and lilac.
C_POS = PLOT_PALETTE[4]
C_NEG = PLOT_PALETTE[1]
C_DIGENIC = PLOT_PALETTE[2]
C_TRIGENIC = PLOT_PALETTE[0]
C_GRAY = PLOT_PALETTE[5]


def canonical_genes(gene_set):
    """`FKH1_MED4` and `FKH1:MED4` both to a sorted tuple."""
    return tuple(sorted(re.split(r"[_:|-]", str(gene_set))))


def load_scores(csv_name, readout):
    """One order's interaction table for one readout, with BH recomputed within it."""
    df = pd.read_csv(osp.join(RESULTS_DIR, csv_name))
    df = df[(df["ffa_type"] == readout) & df["p_value"].notna()].copy()
    df["genes"] = df["gene_set"].map(canonical_genes)
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


# ---------------------------------------------------------------------------------------
# The digenic layer
# ---------------------------------------------------------------------------------------

def panel_digenic_volcano(ax):
    """Digenic epsilon against evidence on total titer, with the within-readout BH line.

    The trigenic volcano of the main text has its mass at negative tau. This one is its
    mirror image, and that is the point: the pairs are overwhelmingly positive, so the
    negative trigenic scores are not a continuation of a trend the pairs set.
    """
    di = load_scores(DIGENIC_CSV, TOTAL)
    sig = di[di["fdr_sig"]]
    p_cut = sig["p_value"].max() if len(sig) else np.nan
    for sub, color, label in (
        (di[di["interaction_score"] <= 0], C_NEG,
         f"negative $\\varepsilon$ (n={int((di['interaction_score'] <= 0).sum())})"),
        (di[di["interaction_score"] > 0], C_POS,
         f"positive $\\varepsilon$ (n={int((di['interaction_score'] > 0).sum())})"),
    ):
        ax.scatter(sub["interaction_score"], -np.log10(sub["p_value"]), s=5,
                   facecolor=color, edgecolor="black", linewidth=0.2, zorder=3,
                   label=label)
    if np.isfinite(p_cut):
        ax.axhline(-np.log10(p_cut), color="black", linewidth=0.5, linestyle="--",
                   zorder=2)
        ax.text(0.02, -np.log10(p_cut), " BH FDR < 0.05",
                transform=ax.get_yaxis_transform(), va="top", ha="left", fontsize=5)
    ax.axvline(0, color=C_GRAY, linewidth=0.4, zorder=1)
    ax.set_xlabel("$\\varepsilon$ (digenic interaction)")
    ax.set_ylabel("$-\\log_{10}$ $P$")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.24)
    n_pos = int((sig["interaction_score"] > 0).sum())
    ax.set_title(
        f"{len(sig)}/{len(di)} digenic interactions at FDR < 0.05\n"
        f"{n_pos} of them positive",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.3)
    return {"n_sig": len(sig), "n_total": len(di), "n_sig_pos": n_pos,
            "median": round(float(di["interaction_score"].median()), 3)}


def panel_digenic_matrix(ax):
    """Every pair's digenic score on total titer, as a 10 by 10 matrix.

    One cell per pair rather than one bar per pair: the pairs are the reference the
    trigenic claim is made against, and the matrix shows at a glance that no factor
    carries the positive digenic signal on its own.
    """
    di = load_scores(DIGENIC_CSV, TOTAL)
    n = len(TF_GENES)
    grid = np.full((n, n), np.nan)
    called = np.zeros((n, n), dtype=bool)
    idx = {g: i for i, g in enumerate(TF_GENES)}
    for row in di.itertuples():
        i, j = idx[row.genes[0]], idx[row.genes[1]]
        grid[i, j] = grid[j, i] = row.interaction_score
        called[i, j] = called[j, i] = bool(row.fdr_sig)
    lim = float(np.nanmax(np.abs(grid)))
    # Diverging around zero with the sign colors, so a cell reads as the same encoding as
    # every other panel; the midpoint is pinned at zero rather than at the data mean.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sign", [C_NEG, "#FFFFFF", C_POS])
    im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                   zorder=2)
    for i in range(n):
        for j in range(n):
            if called[i, j] and i < j:
                ax.plot([j], [i], marker="o", markersize=1.1, color="black", zorder=4)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(TF_GENES, rotation=90)
    ax.set_yticklabels(TF_GENES)
    ax.tick_params(length=1.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, zorder=3)
    ax.grid(which="major", visible=False)
    ax.set_title("digenic $\\varepsilon$ on total titer; dot marks FDR < 0.05\n"
                 "(upper triangle only, the matrix is symmetric)", fontsize=6, pad=3)
    cbar = ax.get_figure().colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=5, length=1.5)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("$\\varepsilon$", fontsize=6)
    # Per-factor call counts, printed so the Note's "GCN5 in 1 of its 9, TFC7 in 8 of its
    # 9" is read off the same table the matrix is drawn from.
    per_factor = {g: int(called[idx[g]].sum()) for g in TF_GENES}
    return {"n_pairs": int(np.isfinite(grid).sum() / 2), "max_abs": round(lim, 3),
            "calls_per_factor": dict(sorted(per_factor.items(), key=lambda kv: -kv[1]))}


def panel_digenic_contribution(ax):
    """What the pairs predict for a triple against what the triple does.

    The x axis is the pairwise part of the multiplicative expectation,
    f_i f_j f_k + eps_ij f_k + eps_ik f_j + eps_jk f_i, which is the triple's phenotype
    as predicted from its singles and its pairs. The y axis is the measured triple.
    The vertical distance to the diagonal is tau, so the panel shows the trigenic score
    as the residual it is defined to be.
    """
    paths = pd.read_csv(osp.join(RESULTS_DIR, "ffa_epistatic_paths.csv"))
    singles = {r.gene_1: r.f_single for r in paths.drop_duplicates("gene_1").itertuples()}
    doubles = {frozenset([r.gene_1, r.gene_2]): r.f_double for r in paths.itertuples()}
    triples = {frozenset(r.triple.split("-")): r.f_triple
               for r in paths.drop_duplicates("triple").itertuples()}
    eps = {k: v - float(np.prod([singles[g] for g in k])) for k, v in doubles.items()}

    tri = load_scores(TRIGENIC_CSV, TOTAL)
    called = {row.genes: bool(row.fdr_sig) for row in tri.itertuples()}

    pred, obs, sig = [], [], []
    for key, f_ijk in triples.items():
        g = sorted(key)
        f = [singles[x] for x in g]
        prod = float(np.prod(f))
        pair_term = (eps[frozenset(g[:2])] * f[2] + eps[frozenset([g[0], g[2]])] * f[1]
                     + eps[frozenset(g[1:])] * f[0])
        pred.append(prod + pair_term)
        obs.append(f_ijk)
        sig.append(called.get(tuple(g), False))
    pred, obs, sig = np.array(pred), np.array(obs), np.array(sig)

    lo = min(pred.min(), obs.min()) - 0.1
    hi = max(pred.max(), obs.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.5, linestyle="--", zorder=2)
    for mask, face, label in (
        (~sig, "white", f"not called (n={int((~sig).sum())})"),
        (sig, C_NEG, f"FDR < 0.05 (n={int(sig.sum())})"),
    ):
        ax.scatter(pred[mask], obs[mask], s=5, facecolor=face, edgecolor="black",
                   linewidth=0.25, zorder=3, label=label)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted from singles and pairs")
    ax.set_ylabel("measured triple titer $f_{ijk}$")
    ax.set_aspect("equal")
    below = int((obs < pred).sum())
    ax.set_title(
        f"{below} of {len(obs)} triples fall below what\ntheir pairs predict "
        "(distance to the line is $\\tau$)",
        fontsize=6, pad=3,
    )
    ax.legend(loc="upper left", frameon=False, handlelength=1.0, labelspacing=0.3,
              borderaxespad=0.3)
    return {"n_below": below, "n": len(obs)}


# ---------------------------------------------------------------------------------------
# The five species under the sum
# ---------------------------------------------------------------------------------------

def _per_readout_counts():
    """Called counts and medians per readout, for both orders, BH within readout."""
    rows = []
    for csv_name, order in ((DIGENIC_CSV, "digenic"), (TRIGENIC_CSV, "trigenic")):
        for readout in READOUTS:
            df = load_scores(csv_name, readout)
            sig = df[df["fdr_sig"]]
            rows.append({
                "order": order, "readout": readout, "n_testable": len(df),
                "n_called": len(sig),
                "n_called_pos": int((sig["interaction_score"] > 0).sum()),
                "n_called_neg": int((sig["interaction_score"] <= 0).sum()),
                "median": float(df["interaction_score"].median()),
            })
    return pd.DataFrame(rows)


def panel_species_counts(ax):
    """How many interactions each readout calls, split by sign, for both orders.

    The summed total is one column among six rather than the readout. C14:0 is the case
    the main text's consensus panel points at from the other side: all 15 of its digenic
    calls are negative and all 26 of its trigenic calls are positive.
    """
    counts = _per_readout_counts()
    x = np.arange(len(READOUTS))
    w = 0.38
    for k, (order, hatch) in enumerate((("digenic", ""), ("trigenic", "///"))):
        sub = counts[counts["order"] == order].set_index("readout").loc[READOUTS]
        off = x + (k - 0.5) * w
        ax.bar(off, sub["n_called_neg"], width=w, color=C_NEG, edgecolor="black",
               linewidth=0.4, hatch=hatch, zorder=3)
        ax.bar(off, sub["n_called_pos"], bottom=sub["n_called_neg"], width=w,
               color=C_POS, edgecolor="black", linewidth=0.4, hatch=hatch, zorder=3)
        for xi, total in zip(off, sub["n_called"]):
            ax.text(xi, total + 1.5, str(int(total)), ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUTS])
    ax.set_ylabel("interactions at FDR < 0.05")
    ax.set_ylim(0, counts["n_called"].max() * 1.30)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles = [
        matplotlib.patches.Patch(facecolor=C_NEG, edgecolor="black", linewidth=0.4,
                                 label="negative"),
        matplotlib.patches.Patch(facecolor=C_POS, edgecolor="black", linewidth=0.4,
                                 label="positive"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                                 label="digenic (45 pairs)"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", linewidth=0.4,
                                 hatch="///", label="trigenic (119 triples)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.2,
              labelspacing=0.25, borderaxespad=0.3, ncol=2, columnspacing=1.0)
    ax.set_title("every readout calls interactions, and the sign\n"
                 "they call depends on the readout", fontsize=6, pad=3)
    return counts.to_dict("records")


def panel_species_agreement(ax):
    """Trigenic tau of the same 119 triples, readout against readout.

    Pearson r below the diagonal and the count of triples called by both above it. If the
    summed total were a faithful summary of the five species, its column would correlate
    strongly with all five; it does not.
    """
    wide = {}
    called = {}
    for readout in READOUTS:
        df = load_scores(TRIGENIC_CSV, readout).set_index("genes")
        wide[readout] = df["interaction_score"]
        called[readout] = df["fdr_sig"]
    wide = pd.DataFrame(wide).dropna()
    called = pd.DataFrame(called).reindex(wide.index).fillna(False)
    n = len(READOUTS)
    corr = wide.corr(method="pearson").to_numpy()
    both = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            both[i, j] = int((called[READOUTS[i]] & called[READOUTS[j]]).sum())

    show = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i > j:
                show[i, j] = corr[i, j]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sign", [C_NEG, "#FFFFFF", C_POS])
    ax.imshow(show, cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1), zorder=2)
    for i in range(n):
        for j in range(n):
            if i > j:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        zorder=4)
            elif i < j:
                ax.text(j, i, str(both[i, j]), ha="center", va="center", fontsize=5,
                        color=C_GRAY, zorder=4)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUTS], rotation=90)
    ax.set_yticklabels([READOUT_LABELS[r] for r in READOUTS])
    ax.tick_params(length=1.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.4, zorder=3)
    ax.grid(which="major", visible=False)
    ax.set_title("trigenic $\\tau$ across readouts: Pearson $r$ below,\n"
                 "triples called by both above", fontsize=6, pad=3)
    off = [corr[i, j] for i in range(n) for j in range(n) if i > j]
    return {"min_r": round(float(np.min(off)), 3), "max_r": round(float(np.max(off)), 3)}


def panel_species_medians(ax):
    """Median interaction score per readout, digenic against trigenic.

    One line per order across the six readouts. The gap between the two lines is the
    quantity the main text reports on total titer alone, and its sign is not the same on
    every species.
    """
    counts = _per_readout_counts()
    x = np.arange(len(READOUTS))
    for order, color, marker in (("digenic", C_DIGENIC, "o"),
                                 ("trigenic", C_TRIGENIC, "s")):
        sub = counts[counts["order"] == order].set_index("readout").loc[READOUTS]
        ax.plot(x, sub["median"], color=color, marker=marker, markersize=2.6,
                linewidth=0.8, markeredgecolor="black", markeredgewidth=0.25, zorder=3,
                label=f"{order} median")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([READOUT_LABELS[r] for r in READOUTS], rotation=90)
    ax.set_ylabel("median interaction score")
    ax.set_xlim(-0.4, len(READOUTS) - 0.6)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.28)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, labelspacing=0.25,
              borderaxespad=0.3)
    med = counts.pivot(index="readout", columns="order", values="median")
    # Derived, never asserted: which way the third deletion moves the median is the
    # finding, so it is read off the same numbers the lines are drawn from.
    species = [r for r in READOUTS if r != TOTAL]
    down = [r for r in species if med.loc[r, "trigenic"] < med.loc[r, "digenic"]]
    up = [r for r in species if med.loc[r, "trigenic"] >= med.loc[r, "digenic"]]
    ax.set_title(
        f"the third deletion lowers the median on {len(down)} of the five\n"
        f"species and raises it on {', '.join(up)}",
        fontsize=6, pad=3,
    )
    return {r: {o: round(float(med.loc[r, o]), 3) for o in med.columns} for r in READOUTS}


# ---------------------------------------------------------------------------------------

def emit(name, width_key, height_mm, draw):
    """One panel to one pair of files, at a standard width and true size."""
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM[width_key]),
                                    mm_to_in(height_mm)))
    info = draw(ax)
    style_axes(ax)
    fig.tight_layout(pad=0.4)
    stem = osp.join(IMAGES_DIR, f"si_panel_{name}")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    plt.close(fig)
    print(f"wrote si_panel_{name}  ({PANEL_WIDTHS_MM[width_key]} x {height_mm} mm)  {info}")
    return info


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    emit("digenic_volcano", "third", 62.0, panel_digenic_volcano)
    emit("digenic_matrix", "third", 62.0, panel_digenic_matrix)
    emit("digenic_contribution", "third", 62.0, panel_digenic_contribution)
    emit("species_counts", "third", 62.0, panel_species_counts)
    emit("species_agreement", "third", 62.0, panel_species_agreement)
    emit("species_medians", "third", 62.0, panel_species_medians)
    # The per-readout table the Supplementary Notes quote, so no count in the text is
    # transcribed from a panel by hand.
    counts = _per_readout_counts()
    out = osp.join(RESULTS_DIR, "supplementary_readout_counts.csv")
    counts.to_csv(out, index=False)
    print(f"wrote {out}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
