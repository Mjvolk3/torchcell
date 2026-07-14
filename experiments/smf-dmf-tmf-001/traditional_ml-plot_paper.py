# experiments/smf-dmf-tmf-001/traditional_ml-plot_paper.py
# [[experiments.smf-dmf-tmf-001.traditional_ml-plot_paper]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/smf-dmf-tmf-001/traditional_ml-plot_paper
"""Paper figures for the classical-ML case study (Supplementary Note 5), in the TorchCell
figure palette (matches Fig 1). Reads the reconstructable summary tables
(experiments/{DS}/results/traditional_ml_summary_with_std.csv, produced by
traditional_ml-summary_table.py) -- so figures and Table S2 share one data source.

Writes NEW files (prefix ``paper_tradml_``) to ASSET_IMAGES_DIR; it does NOT overwrite the
old matplotlib-style bar/line charts.

Outputs (14):
  paper_tradml_bar_{fitness,interaction}_{rf,svr}_{1e3,1e4,1e5}.png   (12 bar charts)
  paper_tradml_progression_{fitness,interaction}.png                 (2 max-perf line plots)

Bars: one validation-selected best config per encoding; val (solid) + test (light) bars,
black error bar = 5-fold CV s.d. on the validation score (only n=1e3/1e4 have CV).
Progression: max-over-encodings test Spearman vs dataset size, one line per model (RF, SVR),
error = CV s.d. of the arg-max config (none at 1e5, no CV run there).
"""
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from dotenv import load_dotenv

load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
SUMMARY = {
    "fitness": "experiments/smf-dmf-tmf-001/results/traditional_ml_summary_with_std.csv",
    "interaction": "experiments/002-dmi-tmi/results/traditional_ml_summary_with_std.csv",
}
SIZES = ["1e3", "1e4", "1e5"]
SIZE_N = {"1e3": 1000, "1e4": 10000, "1e5": 100000}
MODEL_NAME = {"random_forest": "RF", "svr": "SVR"}

# --- TorchCell figure palette (notes/assets/images/color-palette.svg) ---
PAL = {
    "orange": ("#FFE6CC", "#BD8800"), "red": ("#F8CECC", "#A24A46"),
    "purple": ("#E1D5E7", "#846592"), "yellow": ("#FFF2CC", "#BCA04C"),
    "blue": ("#DAE8FC", "#5F7DA8"), "green": ("#D5E8D4", "#729E5A"),
    "gray": ("#F5F5F5", "#5A5A5A"),
}
# encoding family -> palette key (fill, edge)
FAMILY = {
    "random_1": "blue", "random_10": "blue", "random_100": "blue", "random_1000": "blue",
    "codon_frequency": "yellow", "normalized_chrom_pathways": "yellow",
    "calm": "purple", "fudt_upstream": "purple", "fudt_downstream": "purple",
    "prot_T5_all": "purple", "prot_T5_no_dubious": "purple",
    "esm2_t33_650M_UR50D_all": "purple", "esm2_t33_650M_UR50D_no_dubious": "purple",
    "nt_window_5979": "purple", "nt_window_three_prime_300": "purple",
    "nt_window_five_prime_1003": "purple", "one_hot_gene": "green",
}
ORDER = [  # top -> bottom on the y-axis (reversed at plot time)
    "one_hot_gene", "nt_window_five_prime_1003", "nt_window_three_prime_300",
    "nt_window_5979", "esm2_t33_650M_UR50D_no_dubious", "esm2_t33_650M_UR50D_all",
    "prot_T5_no_dubious", "prot_T5_all", "fudt_downstream", "fudt_upstream", "calm",
    "normalized_chrom_pathways", "codon_frequency",
    "random_1000", "random_100", "random_10", "random_1",
]
LBL = {
    "one_hot_gene": "one-hot (6607)", "nt_window_five_prime_1003": "NT 5' 1003",
    "nt_window_three_prime_300": "NT 3' 300", "nt_window_5979": "NT 5979",
    "esm2_t33_650M_UR50D_no_dubious": "ESM2 (no dub)", "esm2_t33_650M_UR50D_all": "ESM2",
    "prot_T5_no_dubious": "ProtT5 (no dub)", "prot_T5_all": "ProtT5",
    "fudt_downstream": "FUDT down", "fudt_upstream": "FUDT up", "calm": "CaLM",
    "normalized_chrom_pathways": "chrom pathways", "codon_frequency": "codon freq",
    "random_1000": "random 1000", "random_100": "random 100", "random_10": "random 10",
    "random_1": "random 1",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": "#5A5A5A", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#5A5A5A", "ytick.color": "#5A5A5A", "text.color": "#222",
    "axes.labelcolor": "#222", "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(dataset):
    df = pd.read_csv(SUMMARY[dataset], dtype={"size": str})
    df["size"] = df["size"].str.replace(".0", "", regex=False).map(
        {"1000": "1e3", "10000": "1e4", "100000": "1e5",
         "1e3": "1e3", "1e4": "1e4", "1e5": "1e5"}).fillna(df["size"])
    return df


def bar_chart(dataset, model, size, df):
    sub = df[(df.model == model) & (df["size"] == size)].set_index("embedding")
    embs = [e for e in ORDER if e in sub.index]
    y = np.arange(len(embs))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    for yi, e in zip(y, embs):
        fill, edge = PAL[FAMILY[e]]
        val = sub.loc[e, "val_spearman"]
        test = sub.loc[e, "test_spearman"]
        std = sub.loc[e, "cv_spearman_std"]
        ax.barh(yi + h / 2, val if pd.notna(val) else 0, height=h, color=fill,
                edgecolor=edge, linewidth=1.0)
        ax.barh(yi - h / 2, test if pd.notna(test) else 0, height=h, color=fill,
                edgecolor=edge, linewidth=1.0, alpha=0.45)
        if pd.notna(std) and pd.notna(val):
            ax.errorbar(val, yi + h / 2, xerr=std, fmt="none", ecolor="#333",
                        elinewidth=1.1, capsize=2.2, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([LBL[e] for e in embs], fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Spearman $\\rho$", fontsize=11)
    cv_note = "5-fold CV s.d." if size in ("1e3", "1e4") else "single split (no CV)"
    ax.set_title(f"{dataset.capitalize()} · {MODEL_NAME[model]} · "
                 f"$n{{=}}10^{{{size[-1]}}}$", fontsize=12, color="#222")
    ax.grid(axis="x", color="#CACACA", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    legend = [
        Patch(facecolor="#B0B0B0", edgecolor="#5A5A5A", label="validation"),
        Patch(facecolor="#B0B0B0", edgecolor="#5A5A5A", alpha=0.45, label="test"),
        Patch(facecolor=PAL["green"][0], edgecolor=PAL["green"][1], label="identity"),
        Patch(facecolor=PAL["purple"][0], edgecolor=PAL["purple"][1], label="biological"),
        Patch(facecolor=PAL["yellow"][0], edgecolor=PAL["yellow"][1], label="hand-crafted"),
        Patch(facecolor=PAL["blue"][0], edgecolor=PAL["blue"][1], label="random"),
    ]
    ax.legend(handles=legend, fontsize=7.5, loc="lower right", frameon=True,
              framealpha=0.9, title=cv_note, title_fontsize=7.5)
    fig.tight_layout()
    out = osp.join(ASSET_IMAGES_DIR, f"paper_tradml_bar_{dataset}_{MODEL_NAME[model].lower()}_{size}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def progression(dataset, df):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    model_pal = {"random_forest": PAL["orange"][1], "svr": PAL["blue"][1]}
    for model in ["random_forest", "svr"]:
        xs, ys, es = [], [], []
        for size in SIZES:
            sub = df[(df.model == model) & (df["size"] == size)].dropna(subset=["test_spearman"])
            if sub.empty:
                continue
            r = sub.loc[sub["test_spearman"].idxmax()]
            xs.append(SIZE_N[size]); ys.append(r["test_spearman"])
            es.append(r["cv_spearman_std"] if pd.notna(r["cv_spearman_std"]) else np.nan)
        c = model_pal[model]
        ax.errorbar(xs, ys, yerr=[0 if np.isnan(e) else e for e in es], fmt="-o", color=c,
                    markerfacecolor=c, markeredgecolor="white", markersize=7, linewidth=2.2,
                    capsize=3, elinewidth=1.3, label=MODEL_NAME[model], zorder=4)
    ax.set_xscale("log")
    ax.set_xticks([SIZE_N[s] for s in SIZES])
    ax.set_xticklabels(["$10^3$", "$10^4$", "$10^5$"], fontsize=11)
    ax.set_xlabel("Training instances", fontsize=11)
    ax.set_ylabel("Best Spearman $\\rho$ (max over encodings)", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{dataset.capitalize()}: best classical performance vs. data",
                 fontsize=12, color="#222")
    ax.grid(color="#CACACA", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left", frameon=True, framealpha=0.9)
    fig.text(0.5, -0.02, "Error bars = 5-fold CV s.d.; no CV at $10^5$.", ha="center",
             fontsize=7.5, color="#5A5A5A")
    fig.tight_layout()
    out = osp.join(ASSET_IMAGES_DIR, f"paper_tradml_progression_{dataset}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    made = []
    for dataset in ("fitness", "interaction"):
        df = load(dataset)
        for model in ("random_forest", "svr"):
            for size in SIZES:
                if not df[(df.model == model) & (df["size"] == size)].empty:
                    made.append(bar_chart(dataset, model, size, df))
        made.append(progression(dataset, df))
    print(f"wrote {len(made)} figures to {ASSET_IMAGES_DIR}:")
    for m in made:
        print("  " + osp.basename(m))


if __name__ == "__main__":
    main()


# ---- Fitness dataset-construction figures (regenerated in palette from the saved splits) ----
CONSTR_BASE = osp.join(os.getenv("DATA_ROOT") or ".",
    "data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene")
CONSTR_SIZES = ["1e03", "1e04", "1e05"]
SPLIT_COLOR = {"train": PAL["blue"][1], "val": PAL["orange"][1], "test": PAL["green"][1]}


def data_split_dist():
    """Fitness label distribution per split (train/val/test), one panel per dataset size."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, size in zip(axes, CONSTR_SIZES):
        for split in ("train", "val", "test"):
            y = np.load(osp.join(CONSTR_BASE, f"sum_{size}", split, "y.npy"))
            ax.hist(y, bins=40, range=(-0.2, 1.6), density=True, histtype="step",
                    linewidth=2, color=SPLIT_COLOR[split], label=f"{split} (n={len(y)})")
        ax.set_title(f"$n{{=}}10^{{{int(size[-1])}}}$", fontsize=11)
        ax.set_xlabel("Fitness", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(axis="y", color="#CACACA", linewidth=0.5, alpha=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Fitness dataset: label distribution across splits", fontsize=12)
    fig.tight_layout()
    out = osp.join(ASSET_IMAGES_DIR, "paper_tradml_datadist_fitness.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def gene_coverage():
    """Per-gene deletion coverage: how many strains delete each gene, by dataset size.
    From the intact one-hot sum X (X[i,g]=1 if gene g present) -> deletions = n - column sum."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    colors = [PAL["blue"][1], PAL["orange"][1], PAL["green"][1]]
    for size, c in zip(CONSTR_SIZES, colors):
        X = np.load(osp.join(CONSTR_BASE, f"sum_{size}", "all", "X.npy"))
        n = X.shape[0]
        deletions = n - X.sum(axis=0)              # per-gene deletion count
        covered = int((deletions > 0).sum())
        vals = deletions[deletions > 0]
        ax.hist(vals, bins=np.logspace(0, np.log10(max(vals.max(), 2)), 30), histtype="step",
                linewidth=2, color=c, label=f"$10^{{{int(size[-1])}}}$: {covered} genes covered")
    ax.set_xscale("log")
    ax.set_xlabel("Deletions per gene (strains)", fontsize=10)
    ax.set_ylabel("Number of genes", fontsize=10)
    ax.set_title("Fitness dataset: gene deletion coverage", fontsize=12)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#CACACA", linewidth=0.5, alpha=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    out = osp.join(ASSET_IMAGES_DIR, "paper_tradml_genecoverage_fitness.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    print("construction figures:")
    print("  " + osp.basename(data_split_dist()))
    print("  " + osp.basename(gene_coverage()))
