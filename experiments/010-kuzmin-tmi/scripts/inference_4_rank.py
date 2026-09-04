# experiments/010-kuzmin-tmi/scripts/inference_4_rank.py
# [[experiments.010-kuzmin-tmi.scripts.inference_4_rank]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_4_rank
#
# Rank the inference_4 space, with the checks that the previous panel lacked.
#
# The three checkpoints disagree by more than a factor of two on how many triples clear
# the positive call, so no single model's tail is reportable. The ranking statistic is
# the ensemble MEAN across the three, and the across-checkpoint SPREAD is carried beside
# it, because a leader that only one model likes is the failure mode that produced the
# collapsed panel.
#
# Three gates are reported rather than applied silently, so the cost of each is visible:
#
#   support   n_supported counts how many of a triple's genes were measured under at
#             least 50 distinct Kuzmin query screens. Every triple here has at least
#             one by construction; 2 and 3 are strictly stronger and are subsets.
#   range     Predictions run past the measured label range. The training labels span
#             [-1.08, +1.13] with sd 0.063; predicted sd is roughly half that, the
#             usual regression-to-the-mean signature. A prediction outside the label
#             range is extrapolation and is flagged, never silently ranked.
#   agreement all three checkpoints positive, and the spread relative to the mean.
#
# Nothing here is a measurement of an interaction. These are model outputs on a space
# with no labels, and the ranking is only as good as the query-pair-disjoint question
# that has not yet been answered for this model.
#
# Run from repo root:
#   ~/miniconda3/envs/torchcell/bin/python \
#     experiments/010-kuzmin-tmi/scripts/inference_4_rank.py
#
# Outputs:
#   results/inference_4/top_triples.csv          the head of the ensemble ranking
#   results/inference_4/rank_summary.json        counts quoted in prose
#   results/inference_4/checkpoint_agreement.csv per-checkpoint tail sizes
#   $ASSET_IMAGES_DIR/010-kuzmin-tmi/inference_4_rank.{png,svg}

import glob
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]

BASE = osp.join(DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/inference_4")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results", "inference_4")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")

# tag -> (short name, validation Pearson) for the three trained checkpoints.
CHECKPOINTS = {
    "lzs9pcj3": ("M01", 0.4520),
    "yv4r30bi": ("M02", 0.4472),
    "c7671wgj": ("M03", 0.4619),
}
CUTS = [0.08, 0.12, 0.16, 0.20, 0.30]
TOP_N = 500
N_TABLE = 25

# The measured training-label range; anything outside it is extrapolation.
LABEL_MIN, LABEL_MAX, LABEL_SD = -1.0816, 1.1280, 0.06326


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial", "font.size": 6, "axes.labelsize": 6,
            "axes.titlesize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "legend.fontsize": 5, "legend.title_fontsize": 5, "figure.titlesize": 6,
            "svg.fonttype": "none", "axes.linewidth": 0.5,
            "savefig.bbox": None, "savefig.pad_inches": 0.0,
        }
    )


def load_checkpoint(tag: str) -> np.ndarray:
    """One checkpoint's predictions in dataset order, verified contiguous."""
    files = sorted(glob.glob(osp.join(BASE, "inferred", f"*{tag}*shard*.parquet")))
    if len(files) != 4:
        raise SystemExit(f"{tag}: expected 4 shard files, found {len(files)}")
    parts = []
    cursor = 0
    for f in files:
        t = pq.read_table(f, columns=["index", "prediction"])
        i = t["index"].to_numpy()
        if i[0] != cursor or not np.all(np.diff(i) == 1):
            raise SystemExit(f"{tag}: shard {f} is not contiguous from {cursor}")
        cursor = int(i[-1]) + 1
        parts.append(t["prediction"].to_numpy())
    p = np.concatenate(parts)
    print(f"  {tag} ({CHECKPOINTS[tag][0]}): {len(p):,} rows, "
          f"mean {p.mean():+.5f}, sd {p.std():.5f}, "
          f"min {p.min():+.4f}, max {p.max():+.4f}")
    return p


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print("loading checkpoints ...")
    preds = {tag: load_checkpoint(tag) for tag in CHECKPOINTS}
    n = len(next(iter(preds.values())))
    stack = np.stack([preds[t] for t in CHECKPOINTS], axis=1)
    mean = stack.mean(axis=1)
    sd = stack.std(axis=1)
    all_pos = (stack > 0).all(axis=1)

    idx = pq.read_table(
        osp.join(BASE, "triple_index.parquet"),
        columns=["index", "gene1", "gene2", "gene3",
                 "n_metabolic", "n_regulator", "n_supported"],
    )
    if int(idx["index"][0].as_py()) != 0 or idx.num_rows != n:
        raise SystemExit("triple_index.parquet does not align with the predictions")
    n_supported = idx["n_supported"].to_numpy()

    # Per-checkpoint tail sizes: this is what says a single model's tail is not
    # reportable on its own.
    rows = []
    for tag, (short, pear) in CHECKPOINTS.items():
        for c in CUTS:
            rows.append({
                "checkpoint": tag, "name": short, "val_pearson": pear,
                "cut": c, "n_above": int((preds[tag] > c).sum()),
            })
    for c in CUTS:
        rows.append({
            "checkpoint": "ensemble_mean", "name": "mean", "val_pearson": np.nan,
            "cut": c, "n_above": int((mean > c).sum()),
        })
        rows.append({
            "checkpoint": "all_three_above", "name": "all3", "val_pearson": np.nan,
            "cut": c, "n_above": int((stack > c).all(axis=1).sum()),
        })
    agree = pd.DataFrame(rows)
    agree.to_csv(osp.join(RESULTS_DIR, "checkpoint_agreement.csv"), index=False)
    print("\n=== how many triples clear each cut ===")
    print(agree.pivot_table(index="cut", columns="name", values="n_above").to_string())

    # Head of the ranking, with everything a reader needs to distrust it.
    order = np.argsort(mean)[::-1][:TOP_N]
    g1 = idx["gene1"].to_numpy(zero_copy_only=False)[order]
    g2 = idx["gene2"].to_numpy(zero_copy_only=False)[order]
    g3 = idx["gene3"].to_numpy(zero_copy_only=False)[order]
    top = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "index": order,
        "gene1": g1, "gene2": g2, "gene3": g3,
        "ensemble_mean": mean[order],
        "checkpoint_sd": sd[order],
        **{CHECKPOINTS[t][0]: preds[t][order] for t in CHECKPOINTS},
        "n_metabolic": idx["n_metabolic"].to_numpy()[order],
        "n_regulator": idx["n_regulator"].to_numpy()[order],
        "n_supported": n_supported[order],
        "all_three_positive": all_pos[order],
        "outside_label_range": (mean[order] > LABEL_MAX) | (mean[order] < LABEL_MIN),
    })
    top["spread_over_mean"] = top["checkpoint_sd"] / top["ensemble_mean"].abs()
    top.to_csv(osp.join(RESULTS_DIR, "top_triples.csv"), index=False)

    print(f"\n=== top {N_TABLE} by ensemble mean ===")
    show = top.head(N_TABLE)[
        ["rank", "gene1", "gene2", "gene3", "ensemble_mean", "checkpoint_sd",
         "M01", "M02", "M03", "n_supported", "all_three_positive"]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The same ranking restricted to triples every gene of which is screen-supported.
    gated = top[top["n_supported"] == 3]
    print(f"\ntop {TOP_N} that are fully screen-supported: {len(gated)}")
    if len(gated):
        print(gated.head(10)[["rank", "gene1", "gene2", "gene3", "ensemble_mean",
                              "checkpoint_sd", "all_three_positive"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Which genes actually carry the head?
    head_genes = pd.Series(
        np.concatenate([top["gene1"], top["gene2"], top["gene3"]])
    )
    gene_counts = head_genes.value_counts()
    print(f"\n=== gene concentration in the top {TOP_N} ===")
    print(f"{len(gene_counts)} distinct genes across {TOP_N} triples "
          f"({3 * TOP_N} slots) drawn from a 934-gene roster")
    print(gene_counts.head(12).to_string())

    summary = {
        "n_triples": int(n),
        "checkpoints": {t: {"name": v[0], "val_pearson": v[1],
                            "mean": float(preds[t].mean()),
                            "sd": float(preds[t].std()),
                            "max": float(preds[t].max())}
                        for t, v in CHECKPOINTS.items()},
        "ensemble_mean_max": float(mean.max()),
        "ensemble_mean_sd": float(mean.std()),
        "label_sd": LABEL_SD,
        "n_above_0.08_ensemble": int((mean > 0.08).sum()),
        "n_above_0.16_ensemble": int((mean > 0.16).sum()),
        "n_above_0.08_all_three": int((stack > 0.08).all(axis=1).sum()),
        "n_above_0.16_all_three": int((stack > 0.16).all(axis=1).sum()),
        "n_outside_label_range": int(((mean > LABEL_MAX) | (mean < LABEL_MIN)).sum()),
        "top_ensemble_mean": float(top["ensemble_mean"].iloc[0]),
        "top_triple": [str(top["gene1"].iloc[0]), str(top["gene2"].iloc[0]),
                       str(top["gene3"].iloc[0])],
        "median_spread_of_top500": float(top["checkpoint_sd"].median()),
        "n_top500_all_three_positive": int(top["all_three_positive"].sum()),
        "n_top500_fully_supported": int((top["n_supported"] == 3).sum()),
        "support_distribution_of_top500": {
            int(k): int(v) for k, v in top["n_supported"].value_counts().items()
        },
        "support_distribution_of_space": {
            int(k): int(v) for k, v in pd.Series(n_supported).value_counts().items()
        },
        # Concentration is the diagnostic that matters. The previous panel's tail was
        # one clique, and a ranked list that is really a list about six genes is a
        # claim about those genes, not about 41.9 million triples.
        "n_distinct_genes_in_top500": int(len(gene_counts)),
        "top_gene": str(gene_counts.index[0]),
        "top_gene_share_of_head": float(gene_counts.iloc[0] / TOP_N),
        "share_of_head_covered_by_top6_genes": float(
            (head_genes.isin(gene_counts.head(6).index)
             .to_numpy().reshape(-1, 3).any(axis=1)).mean()
        ),
        "n_roster_genes": 934,
    }
    with open(osp.join(RESULTS_DIR, "rank_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    plot(preds, mean, sd, stack, n_supported, top, gene_counts,
         osp.join(IMAGES_DIR, "inference_4_rank"))
    print(f"\nwrote {RESULTS_DIR} and figures to {IMAGES_DIR}")


def plot(preds, mean, sd, stack, n_supported, top, gene_counts, out_stem):
    set_plot_style()
    fig, axes2 = plt.subplots(
        2, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(112.0))
    )
    # a b on the top row, c d on the bottom, so the letters read in Z order.
    axes = [axes2[0][0], axes2[0][1], axes2[1][0], axes2[1][1]]

    # a: the three checkpoints disagree, which is why the ensemble is the statistic.
    ax = axes[0]
    grid = np.linspace(0.0, 0.6, 300)
    for i, (tag, (short, pear)) in enumerate(CHECKPOINTS.items()):
        p = preds[tag]
        surv = np.array([(p > t).mean() for t in grid])
        ax.plot(grid, np.where(surv > 0, surv, np.nan), color=PLOT_PALETTE[i],
                linewidth=0.9, label=f"{short} (val r = {pear:.3f})", zorder=3)
    surv = np.array([(mean > t).mean() for t in grid])
    ax.plot(grid, np.where(surv > 0, surv, np.nan), color="black", linewidth=1.1,
            linestyle="--", label="ensemble mean", zorder=4)
    for c in CUTS:
        ax.axvline(c, color="0.7", linewidth=0.4, linestyle=":", zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted $\\tau$")
    ax.set_ylabel("Fraction of the space above")
    ax.set_title("a  The checkpoints disagree on the tail", fontsize=6, loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.4,
              labelspacing=0.25, borderpad=0.3)

    # b: agreement against magnitude for the ranked head.
    ax = axes[1]
    ax.scatter(top["ensemble_mean"], top["checkpoint_sd"], s=3.0,
               color=[PLOT_PALETTE[int(k) - 1] for k in top["n_supported"]],
               linewidths=0, zorder=3)
    lim = np.linspace(0, float(top["ensemble_mean"].max()), 50)
    ax.plot(lim, lim, color="black", linewidth=0.5, linestyle="--", zorder=4)
    ax.set_xlabel("Ensemble mean predicted $\\tau$")
    ax.set_ylabel("Across-checkpoint SD")
    ax.set_title(
        f"b  Top {TOP_N} by ensemble mean\n"
        f"dashed line is SD equal to the mean; "
        f"{int(top['all_three_positive'].sum())} of {TOP_N} have all three positive",
        fontsize=6, loc="left", pad=3,
    )
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", linestyle="none", markersize=2.4,
                   color=PLOT_PALETTE[k - 1], label=f"{k} screen-supported gene(s)")
        for k in sorted(top["n_supported"].unique())
    ], loc="upper left", frameon=True, fontsize=5, handlelength=1.0,
        labelspacing=0.25, borderpad=0.3)

    # c: support composition, ranked head against the whole space.
    ax = axes[2]
    space = pd.Series(n_supported).value_counts(normalize=True).sort_index()
    head = top["n_supported"].value_counts(normalize=True).sort_index()
    ks = sorted(set(space.index) | set(head.index))
    xs = np.arange(len(ks))
    ax.bar(xs - 0.2, [space.get(k, 0) for k in ks], 0.4, color="0.72",
           edgecolor="black", linewidth=0.4, zorder=3, label="whole space")
    ax.bar(xs + 0.2, [head.get(k, 0) for k in ks], 0.4, color=PLOT_PALETTE[0],
           edgecolor="black", linewidth=0.4, zorder=3, label=f"top {TOP_N}")
    for x, k in zip(xs, ks):
        ax.text(x - 0.2, space.get(k, 0) + 0.01, f"{space.get(k, 0):.1%}",
                ha="center", va="bottom", fontsize=4.5)
        ax.text(x + 0.2, head.get(k, 0) + 0.01, f"{head.get(k, 0):.1%}",
                ha="center", va="bottom", fontsize=4.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Genes measured under $\\geq$ 50 screens")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.12)
    ax.set_title("c  Does the head rest on supported genes?", fontsize=6,
                 loc="left", pad=3)
    ax.legend(loc="upper right", frameon=True, fontsize=5, handlelength=1.0,
              labelspacing=0.25, borderpad=0.3)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    # d: the head is a clique, and saying so is the point of the panel.
    ax = axes[3]
    k = 12
    gc = gene_counts.head(k)
    ys = np.arange(len(gc))[::-1]
    ax.barh(ys, gc.to_numpy() / TOP_N, 0.62, color=PLOT_PALETTE[1],
            edgecolor="black", linewidth=0.4, zorder=3)
    for y, v in zip(ys, gc.to_numpy() / TOP_N):
        ax.text(v + 0.01, y, f"{v:.0%}", va="center", fontsize=5)
    ax.set_yticks(ys)
    ax.set_yticklabels(gc.index, fontsize=5)
    ax.set_xlim(0, float(gc.max() / TOP_N) * 1.25)
    ax.set_xlabel(f"Share of the top {TOP_N} triples containing the gene")
    ax.set_title(
        f"d  The head is a clique\n"
        f"{len(gene_counts)} distinct genes carry the top {TOP_N}, "
        f"out of a 934-gene roster",
        fontsize=6, loc="left", pad=3,
    )
    ax.xaxis.set_major_locator(MultipleLocator(0.05))

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.3, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "inference_4 ranked over 41,877,232 metabolism-by-regulation triples. "
        "Predicted values, not measurements, and the model has not been refit on a "
        "query-pair-disjoint split.",
        fontsize=6, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(f"{out_stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
