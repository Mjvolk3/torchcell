# experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_rank_summary_palette.py
# [[experiments.002-dmi-tmi.scripts.traditional_ml-plot_representation_rank_summary_palette]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_rank_summary_palette
#
# The SMALL summary figure for Fig 7: across every (task x model x scale) setting, which
# representation type (pert_sum / pert_mean / intact_sum / intact_mean) is best overall?
# This is a rank-aggregation (Demsar 2006) view -- the single headline the six-panel
# per-setting figure lacks.
#
# Method: within each of the 11 complete settings (2 tasks x 2 models x 3 scales = 12, minus
# SVR-interactions-1e3 which has no complete 4-type row) we rank the four types by their mean
# test Pearson over the 17 embeddings (1 = best). We then AGGREGATE the ranks (raw r is not
# comparable across fitness ~0.9 vs interactions ~0.4; ranks are). Reported:
#   - each type's AVERAGE RANK across settings (the ordering) + every setting's rank as a dot
#     colored by task, so the spread and the interactions<->fitness shift are both visible;
#   - the Friedman omnibus p across settings (are the average ranks different at all?);
#   - the Nemenyi critical difference (CD) bar -- the conservative pairwise gate. With only
#     N=11 settings CD is large, so it formally separates only the extremes; the average-rank
#     ordering and the #1-finish tally are the descriptive evidence.
#
# Reads BOTH experiment result dirs (interactions = 002-dmi-tmi, fitness = smf-dmf-tmf-001).
#
# Run from repo root:
#   python experiments/002-dmi-tmi/scripts/traditional_ml-plot_representation_rank_summary_palette.py

import os
import os.path as osp
import ast

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

ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
REP = ["pert_sum", "pert_mean", "intact_sum", "intact_mean"]
REP_LBL = {"pert_sum": "pert_sum", "pert_mean": "pert_mean",
           "intact_sum": "intact_sum", "intact_mean": "intact_mean"}
TASKS = [("interactions", "experiments/002-dmi-tmi/results", PLOT_PALETTE[0]),   # orange (slot 1)
         ("fitness", "experiments/smf-dmf-tmf-001/results", PLOT_PALETTE[1])]    # red (slot 2)
MODELS = ["random_forest", "svr"]
SIZES = ["1e3", "1e4", "1e5"]
ORDER = [
    "random_1", "random_10", "codon_frequency", "random_100", "normalized_chrom_pathways",
    "calm", "fudt_upstream", "fudt_downstream", "random_1000", "prot_T5_all",
    "prot_T5_no_dubious", "esm2_t33_650M_UR50D_all", "esm2_t33_650M_UR50D_no_dubious",
    "nt_window_5979", "nt_window_three_prime_300", "nt_window_five_prime_1003", "one_hot_gene",
]


def _parse(x):
    if isinstance(x, str) and x.strip().startswith("["):
        v = ast.literal_eval(x)
        return v[0] if len(v) == 1 else "|".join(v)
    return x


def setting_ranks(rroot, model, sci):
    """Rank the 4 rep types (1=best) by mean test-Pearson over embeddings, or None."""
    df = pd.read_csv(osp.join(rroot, model, f"combined_df_spearman_{sci}.csv"))
    df["emb"] = df["cell_dataset.node_embeddings"].apply(_parse)
    df = df[df["emb"].isin(ORDER) & df["test_pearson"].notna()].copy()
    if df.empty:
        return None
    df["rep"] = df.apply(lambda r: ("pert" if r["cell_dataset.is_pert"] else "intact")
                         + "_" + r["cell_dataset.aggregation"], axis=1)
    best = df.loc[df.groupby(["emb", "rep"])["val_r2"].idxmax()]
    means = best.pivot_table(index="emb", columns="rep",
                             values="test_pearson").reindex(columns=REP).mean()
    if means.isna().any():
        return None
    return means.rank(ascending=False)  # 1 = best


def main():
    rows, dot_task = [], []
    for task, rroot, _ in TASKS:
        for model in MODELS:
            for sci in SIZES:
                rk = setting_ranks(rroot, model, sci)
                if rk is not None:
                    rows.append(rk)
                    dot_task.append(task)
    R = pd.DataFrame(rows)
    N = len(R)
    avg = R.mean().sort_values()           # ascending -> best (lowest rank) first
    order = list(avg.index)                # best..worst
    friedman_p = stats.friedmanchisquare(*[R[c] for c in REP]).pvalue
    q05 = 2.569                            # Nemenyi studentized-range/sqrt2, k=4, alpha=0.05
    CD = q05 * np.sqrt(4 * 5 / (6 * N))

    task_color = {t: c for t, _, c in TASKS}
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["half_plus"]), mm_to_in(56)))
    # Fixed row order matching the per-setting bar / comparison figures
    # (pert_sum top ... intact_mean bottom), NOT sorted by rank -- keeps each type in the
    # same position across all of Fig 7. The rank is still read off the diamond's x + label.
    y_of = {rep: i for i, rep in enumerate(reversed(REP))}

    rng = np.random.default_rng(42)
    for rep in REP:
        y = y_of[rep]
        ranks = R[rep].to_numpy()
        jit = (rng.random(len(ranks)) - 0.5) * 0.34
        for rk, tk, j in zip(ranks, dot_task, jit):
            ax.scatter(rk, y + j, s=7, color=task_color[tk], alpha=0.75,
                       edgecolors="none", zorder=2)
        ax.scatter(avg[rep], y, s=34, marker="D", color="black", zorder=4,
                   edgecolors="white", linewidths=0.4)
        ax.text(avg[rep], y + 0.30, f"{avg[rep]:.2f}", ha="center", va="bottom",
                fontsize=6, color="black", zorder=5)

    # CD reference ruler at the BOTTOM headroom band; the horizontal legend sits inside the
    # top band (upper-center), so neither can occlude data.
    y_cd = -0.58
    x0 = avg.min()  # best (lowest) average rank, wherever that type sits in the fixed order
    ax.plot([x0, x0 + CD], [y_cd, y_cd], color="#666666", lw=1.0, zorder=3)
    for xx in (x0, x0 + CD):
        ax.plot([xx, xx], [y_cd - 0.08, y_cd + 0.08], color="#666666", lw=1.0, zorder=3)
    # label ABOVE the bar (between the ruler and the data) so it never touches the bottom
    # border
    ax.text((x0 + x0 + CD) / 2, y_cd + 0.07, f"CD = {CD:.2f}", ha="center", va="bottom",
            fontsize=6, color="#666666")

    ax.set_yticks(range(len(REP)))
    ax.set_yticklabels([REP_LBL[r] for r in reversed(REP)], fontsize=6)
    ax.set_ylim(-0.8, len(REP) + 0.55)
    ax.set_xlim(0.7, 4.3)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Average rank  (1 = best)", fontsize=6)
    # rank 1 (best) on the left = most intuitive reading; Demsar often flips it, but
    # left-to-right best-to-worst matches the y-order (best at top).
    ax.grid(True, axis="x", ls="-", lw=0.5, alpha=0.15)
    ax.tick_params(axis="both", which="major", labelsize=6, width=0.5, length=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("black")

    handles = [
        plt.Line2D([0], [0], marker="o", ms=3, ls="none", mfc=task_color["interactions"],
                   mec="none", label="interactions"),
        plt.Line2D([0], [0], marker="o", ms=3, ls="none", mfc=task_color["fitness"],
                   mec="none", label="fitness"),
        plt.Line2D([0], [0], marker="D", ms=4, ls="none", mfc="black", mec="white",
                   mew=0.4, label="mean rank"),
    ]
    # one horizontal bar, centered, INSIDE the top band -- headroom keeps it above all data
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=6, handletextpad=0.3,
              columnspacing=1.4, borderpad=0.3, framealpha=0.9)

    fried = f"Friedman p = {friedman_p:.3f}" if friedman_p >= 1e-3 else f"Friedman p = {friedman_p:.1e}"
    fig.suptitle(
        f"Which representation type is best overall?  "
        f"({N} task×model×scale configurations)\n"
        f"dot = one configuration;  {fried};  CD = Nemenyi (α=0.05)",
        fontsize=6, y=0.995)
    top = 0.85
    fig.subplots_adjust(left=0.24, right=0.98, top=top, bottom=0.17)

    out = osp.join(ASSET_IMAGES_DIR, "traditional-ml_representation_rank_summary_palette.svg")
    savefig_true_size_svg(fig, out)
    w_in, h_in = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  FIGURE SIZE: {w_in*25.4:.1f} mm wide x {h_in*25.4:.1f} mm tall "
          f"(caption top ~{(1-top)*h_in*25.4:.1f} mm croppable)")
    print(f"  N={N} settings; Friedman p={friedman_p:.3g}; Nemenyi CD={CD:.2f}")
    print("  average rank (1=best):")
    for rep in order:
        print(f"    {rep:<12} {avg[rep]:.2f}   #1 finishes: {int((R[rep]==1).sum())}/{N}")


if __name__ == "__main__":
    main()
