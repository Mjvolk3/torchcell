# experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_comparison_palette.py
# [[experiments.002-dmi-tmi.scripts.traditional_ml-plot_representation_comparison_palette]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_comparison_palette
#
# "One last view": pooling over ALL embedding methods, is any of the four REPRESENTATION
# TYPES -- pert_sum, pert_mean, intact_sum, intact_mean (is_pert x aggregation) -- better
# than the others, irrespective of which embedding was chosen? One figure per task
# (interactions here; fitness in the smf copy); rows = model (Random Forest, SVR), columns
# = dataset scale (1e3, 1e4, 1e5).
#
# Each panel pools the 17 embeddings: every embedding contributes one test-Pearson value per
# representation type, drawn as a faint colored line across the four types (the comparison is
# PAIRED/repeated-measures, blocked by embedding). Overlaid black diamond = mean across
# embeddings +/- 1 SD. The Friedman omnibus test (repeated-measures across the four types)
# is reported as a bare p in each panel's top-right corner; "Friedman test" is named in the
# title. Friedman is an omnibus test -- it says whether ANY type differs, not which pair;
# pairwise letters are deliberately left for a later pass with the written explanation.
#
# Elastic Net is intentionally absent: it only ever ran the `sum` aggregation, so it has no
# pert_mean / intact_mean and cannot enter a 4-way comparison.
#
# Selection: for each (embedding, representation type) take the best-by-val_r2 row among the
# test-evaluated ones in combined_df_spearman_{size}.csv, then read its test_pearson.
#
# The suptitle is a REVIEW-ONLY caption meant to be cropped for the final figure; the script
# prints the full figure size and the title-band height (what you lose by cropping) in mm.
#
# Run from repo root:
#   python experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_comparison_palette.py

import os
import os.path as osp
import ast
import itertools

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.5,
    "savefig.bbox": "standard", "savefig.pad_inches": 0.01,
})

import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.ticker import MultipleLocator
from dotenv import load_dotenv

from torchcell.utils import savefig_true_size_svg, mm_to_in, PANEL_WIDTHS_MM, PLOT_PALETTE

load_dotenv()

# ----- experiment config (the only block that differs between 002 and smf copies) -------
RESULTS_ROOT = "experiments/002-dmi-tmi/results"
OUT_PREFIX = "002-dmi-tmi"
TASK_LABEL = "Gene interaction"
Y_LIM = (0.0, 0.5)
Y_MAJOR = 0.1
FIG_W_MM = PANEL_WIDTHS_MM["full"]  # 179 mm, full Nature width
FIG_H_MM = 90.0
# ----------------------------------------------------------------------------------------

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
MODELS = [("random_forest", "Random Forest"), ("svr", "SVR")]
SIZES = [("1e3", "$10^3$"), ("1e4", "$10^4$"), ("1e5", "$10^5$")]
REP = ["pert_sum", "pert_mean", "intact_sum", "intact_mean"]
REP_LBL = ["pert\nsum", "pert\nmean", "intact\nsum", "intact\nmean"]

node_embedding_order = [
    "random_1", "random_10", "codon_frequency", "random_100",
    "normalized_chrom_pathways", "calm", "fudt_upstream", "fudt_downstream",
    "random_1000", "prot_T5_all", "prot_T5_no_dubious", "esm2_t33_650M_UR50D_all",
    "esm2_t33_650M_UR50D_no_dubious", "nt_window_5979", "nt_window_three_prime_300",
    "nt_window_five_prime_1003", "one_hot_gene",
]
COLOR = {e: c for e, c in zip(node_embedding_order, PLOT_PALETTE)}


def _parse_emb(x):
    if isinstance(x, str) and x.strip().startswith("["):
        v = ast.literal_eval(x)
        return v[0] if len(v) == 1 else "|".join(v)
    return x


def load(model, sci):
    """17-embedding x 4-rep-type frame of test Pearson (best-by-val_r2 among tested rows)."""
    df = pd.read_csv(osp.join(RESULTS_ROOT, model, f"combined_df_spearman_{sci}.csv"))
    df["emb"] = df["cell_dataset.node_embeddings"].apply(_parse_emb)
    df = df[df["emb"].isin(node_embedding_order) & df["test_pearson"].notna()].copy()
    if df.empty:
        return None
    df["rep"] = df.apply(
        lambda r: ("pert" if r["cell_dataset.is_pert"] else "intact")
        + "_" + r["cell_dataset.aggregation"], axis=1)
    best = df.loc[df.groupby(["emb", "rep"])["val_r2"].idxmax()]
    return best.pivot_table(index="emb", columns="rep", values="test_pearson").reindex(
        columns=REP)


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    return out


def bootstrap_ci(vals, n_boot=2000, seed=42):
    """95% percentile bootstrap CI of the mean over embeddings (commensurate with the axis)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cld_letters(test, present, alpha=0.05, min_n=6):
    """Compact letter display: all pairwise Wilcoxon (paired, BH-FDR); letter the types so
    that types sharing a letter are NOT significantly different. Letters ordered 'a' = top
    group by mean. Returns {rep: 'a'/'ab'/...}."""
    if len(present) < 2:
        return {r: "" for r in present}
    pairs = list(itertools.combinations(present, 2))
    pvals = []
    for a, b in pairs:
        sub = test[[a, b]].dropna()
        if len(sub) < min_n or (sub[a] - sub[b]).abs().sum() == 0:
            pvals.append(1.0)
        else:
            try:
                _, p = stats.wilcoxon(sub[a], sub[b])
            except ValueError:
                p = 1.0
            pvals.append(p)
    padj = dict(zip(pairs, bh_fdr(pvals)))

    def not_diff(a, b):
        if a == b:
            return True
        key = (a, b) if (a, b) in padj else (b, a)
        return padj[key] >= alpha

    # all cliques of the not-different graph, then keep the maximal ones
    cliques = []
    for size in range(len(present), 0, -1):
        for combo in itertools.combinations(present, size):
            if all(not_diff(a, b) for a, b in itertools.combinations(combo, 2)):
                cliques.append(frozenset(combo))
    maximal = [c for c in cliques if not any(c < d for d in cliques)]
    means = test[present].mean()
    maximal.sort(key=lambda c: -np.mean([means[m] for m in c]))
    letters = {r: "" for r in present}
    for i, clique in enumerate(maximal):
        L = chr(ord("a") + i)
        for m in clique:
            letters[m] += L
    return {r: "".join(sorted(letters[r])) for r in present}


def main():
    fig, axes = plt.subplots(
        len(MODELS), 3, figsize=(mm_to_in(FIG_W_MM), mm_to_in(FIG_H_MM)),
        sharey=True, squeeze=False,
    )
    x = np.arange(4)
    for r, (mkey, mlbl) in enumerate(MODELS):
        for c, (sci, sci_tex) in enumerate(SIZES):
            ax = axes[r][c]
            test = load(mkey, sci)
            yr = Y_LIM[1] - Y_LIM[0]
            if test is not None:
                for emb in test.index:
                    y = test.loc[emb, REP].to_numpy(dtype=float)
                    ax.plot(x, y, "-", color=COLOR.get(emb, "#999"), lw=0.4, alpha=0.4,
                            zorder=1)
                    ax.scatter(x, y, s=5, color=COLOR.get(emb, "#999"), alpha=0.7,
                               zorder=2, edgecolors="none")
                # mean +/- 95% bootstrap CI (uncertainty ON the mean, in test-r units)
                present = [rep for rep in REP if test[rep].notna().sum() >= 3]
                comp = test.dropna()
                fried_p = (stats.friedmanchisquare(*[comp[k] for k in REP]).pvalue
                           if len(comp) >= 3 else float("nan"))
                # Protected post-hoc: only letter pairwise differences when the Friedman
                # omnibus gate is significant; otherwise all types share one letter ("a").
                cld_input = [rep for rep in REP if test[rep].notna().sum() >= 6]
                if np.isfinite(fried_p) and fried_p < 0.05:
                    letters = cld_letters(test, cld_input)
                else:
                    letters = {rep: "a" for rep in cld_input}
                for rep in present:
                    i = REP.index(rep)
                    vals = test[rep].dropna().to_numpy()
                    mean = vals.mean()
                    lo, hi = bootstrap_ci(vals)
                    yerr = ([[mean - lo], [hi - mean]] if np.isfinite(lo) else None)
                    ax.errorbar(i, mean, yerr=yerr, fmt="D", ms=2.6, color="black",
                                mfc="black", ecolor="black", elinewidth=0.7, capsize=1.8,
                                capthick=0.7, zorder=4)
                    if letters.get(rep):
                        # place above the column's tallest point, but keep a clear margin
                        # below the top border so letters never touch/exit the frame
                        yl = min(vals.max() + 0.03 * yr, Y_LIM[1] - 0.08 * yr)
                        ax.text(i, yl, letters[rep], ha="center", va="bottom", fontsize=6,
                                fontweight="bold", color="black", zorder=5)
                if np.isfinite(fried_p):
                    ptxt = (f"Friedman p = {fried_p:.1e}" if fried_p < 0.05
                            else f"Friedman p = {fried_p:.2f} n.s.")
                    # in the margin just above each panel, right-aligned -- the only band
                    # that is data- and letter-free in every panel (letters sit along the
                    # top of each column, so no in-panel corner is reliably empty)
                    ax.text(1.0, 1.015, ptxt, transform=ax.transAxes, fontsize=5,
                            ha="right", va="bottom",
                            color="#333333" if fried_p < 0.05 else "#B85450")
            if r == 0:
                ax.set_title(sci_tex, fontsize=6, pad=6)
            if c == 0:
                ax.set_ylabel(f"{mlbl}\nTest Pearson $r$", fontsize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(REP_LBL if r == len(MODELS) - 1 else [], fontsize=5.5)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(*Y_LIM)
            ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR))
            ax.grid(True, axis="y", ls="-", lw=0.5, alpha=0.15)
            ax.tick_params(axis="both", which="major", labelsize=5.5, width=0.5, length=2)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_edgecolor("black")

    # Review-only caption (crop for the final figure). All legend content lives here so no
    # in-panel legend obstructs data. y is measured in figure fraction; the band it occupies
    # (top TITLE_BAND_MM) is what a crop removes -- reported below.
    fig.suptitle(
        f"{TASK_LABEL}: is any representation type better, irrespective of embedding?\n"
        f"faint line = one embedding (17 total);  ◆ = mean (95% bootstrap CI);  "
        f"Friedman p = omnibus test\n"
        f"letters = compact letter display — types sharing a letter are not significantly "
        f"different (pairwise Wilcoxon, BH-FDR; shown only when Friedman is significant)",
        fontsize=6, y=0.99)
    top = 0.80
    fig.subplots_adjust(left=0.10, right=0.99, top=top, bottom=0.10, hspace=0.13,
                        wspace=0.07)

    out = osp.join(ASSET_IMAGES_DIR, f"{OUT_PREFIX}_representation_type_comparison_palette.svg")
    savefig_true_size_svg(fig, out)

    w_in, h_in = fig.get_size_inches()
    w_mm, h_mm = w_in * 25.4, h_in * 25.4
    title_band_mm = (1.0 - top) * h_mm  # figure area above the panels = croppable caption
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  FIGURE SIZE: {w_mm:.1f} mm wide x {h_mm:.1f} mm tall "
          f"(full Nature width = {PANEL_WIDTHS_MM['full']} mm)")
    print(f"  croppable review caption occupies the top ~{title_band_mm:.1f} mm; "
          f"plot body below it is ~{h_mm - title_band_mm:.1f} mm tall")

    for mkey, mlbl in MODELS:
        for sci, _ in SIZES:
            test = load(mkey, sci)
            if test is None:
                print(f"  [{mlbl}] {sci}: no test-evaluated rows")
                continue
            comp = test.dropna()
            means = test.mean()
            p = (stats.friedmanchisquare(*[comp[k] for k in REP]).pvalue
                 if len(comp) >= 3 else float("nan"))
            print(f"  [{mlbl}] {sci}: n={len(comp)} best={means.idxmax()} "
                  f"({means.max():.3f}); Friedman p={p:.2g}")


if __name__ == "__main__":
    main()
