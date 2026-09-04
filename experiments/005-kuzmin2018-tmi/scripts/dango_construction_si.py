# experiments/005-kuzmin2018-tmi/scripts/dango_construction_si.py
# [[experiments.005-kuzmin2018-tmi.scripts.dango_construction_si]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dango_construction_si
"""DANGO construction facts for the Supplementary Note: per-channel lambda and the dataset.

DANGO weights the zero entries of each STRING channel's adjacency in its reconstruction loss
by a per-channel ``lambda_k``, set from the "percentage of decreased zeroes" between two
STRING releases: a channel whose zeros shrank by more than 1% gets ``lambda_k = 0.1``,
otherwise ``lambda_k = 1.0``. The DANGO paper reports the rule and two of its values
(0.02% co-occurrence, 2.42% co-expression, v9.1 to v11.0) but not the computation. TorchCell
recomputes it from the cached ``SCerevisiaeGraph`` STRING channels with this definition,
stated once here and used for both transitions:

    common nodes   V  = genes present in BOTH releases of the channel
    pairs          P  = {unordered {u, v} : u, v in V, u != v}     (|P| = |V|(|V|-1)/2)
    edges          E_old, E_new = channel edges restricted to pairs in P
    decreased zeros    = |E_new \\ E_old| / (|P| - |E_old|) x 100

so the denominator is the number of zero entries of the older release over the shared
genes and the numerator the pairs that were zero and became edges. Undirected pairs are
canonicalized (sorted tuples) before the set difference, so an edge stored as (u, v) in one
release and (v, u) in the other is the same pair. The two exploratory scripts
(``dango_lambda_determination.py``, ``..._string11_0_to_string12_0.py``) compare raw
``G.edges()`` tuples without canonicalizing; this script is the committed source for the
paper and its numbers are checked against theirs in the Dendron note.

The dataset facts (records, split sizes, perturbed genes, label sign) are read from the
data-module cache of the Kuzmin 2018 trigenic build the replication trained on
(``$DATA_ROOT/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build``, seed 42,
80/10/10 by ``CellDataModule``).

Outputs
-------
results : experiments/005-kuzmin2018-tmi/results/dango_decreased_zeros.csv
          experiments/005-kuzmin2018-tmi/results/dango_dataset_split.csv
panel   : $ASSET_IMAGES_DIR/005-kuzmin2018-tmi/dango_decreased_zeros.{svg,png}

Run from the repo root (``--from-csv`` re-renders the panel from the frozen CSV without
loading the graphs):
    python experiments/005-kuzmin2018-tmi/scripts/dango_construction_si.py [--from-csv]
"""

import argparse
import json
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

# Set AFTER the torchcell imports: torchcell.graph (imported lazily below) applies the repo
# mplstyle on import; these rcParams are re-applied in main() after that import too.
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.5,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.01,
}
plt.rcParams.update(RC)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

RESULTS_DIR = "experiments/005-kuzmin2018-tmi/results"
ZEROS_CSV = osp.join(RESULTS_DIR, "dango_decreased_zeros.csv")
SPLIT_CSV = osp.join(RESULTS_DIR, "dango_dataset_split.csv")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "005-kuzmin2018-tmi")
BUILD_DIR = osp.join(DATA_ROOT, "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build")

CHANNELS = ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database"]
CHANNEL_LABEL = {
    "neighborhood": "neighborhood",
    "fusion": "fusion",
    "cooccurence": "co-occurrence",
    "coexpression": "co-expression",
    "experimental": "experimental",
    "database": "database",
}
# (older release, newer release): the older release's channels take the resulting lambda.
TRANSITIONS = [("9_1", "11_0"), ("11_0", "12_0")]
VERSION_LABEL = {"9_1": "v9.1", "11_0": "v11.0", "12_0": "v12.0"}
# Same release colors as dango_string_version_sweep.py and graph_statistics.py.
VERSION_COLOR = {"9_1": PLOT_PALETTE[0], "11_0": PLOT_PALETTE[1], "12_0": PLOT_PALETTE[2]}
THRESHOLD_PCT = 1.0
# The two values the DANGO paper prints for its own v9.1 -> v11.0 computation.
DANGO_PAPER_PCT = {"cooccurence": 0.02, "coexpression": 2.42}


def dango_lambda(pct: float) -> float:
    """DANGO rule: more than 1% decreased zeros -> 0.1, otherwise 1.0."""
    return 0.1 if pct > THRESHOLD_PCT else 1.0


def undirected_pairs(G) -> set[tuple[str, str]]:
    return {tuple(sorted((u, v))) for u, v in G.edges() if u != v}


def decreased_zeros(builder) -> pd.DataFrame:
    rows = []
    for old, new in TRANSITIONS:
        for ch in CHANNELS:
            G_old = getattr(builder, f"G_string{old}_{ch}").graph
            G_new = getattr(builder, f"G_string{new}_{ch}").graph
            common = set(G_old.nodes()) & set(G_new.nodes())
            E_old = {e for e in undirected_pairs(G_old) if e[0] in common and e[1] in common}
            E_new = {e for e in undirected_pairs(G_new) if e[0] in common and e[1] in common}
            n = len(common)
            possible = n * (n - 1) // 2
            zeros_old = possible - len(E_old)
            gained = len(E_new - E_old)
            pct = 100.0 * gained / zeros_old
            rows.append(
                {
                    "transition": f"{old}->{new}",
                    "old_release": old,
                    "new_release": new,
                    "channel": ch,
                    "common_nodes": n,
                    "possible_pairs": possible,
                    "edges_old": len(E_old),
                    "edges_new": len(E_new),
                    "pairs_gained": gained,
                    "pairs_lost": len(E_old - E_new),
                    "zeros_old": zeros_old,
                    "pct_decreased_zeros": pct,
                    "lambda": dango_lambda(pct),
                }
            )
            print(f"  {old:>4s}->{new:<4s} {ch:13s} nodes={n:5d} E_old={len(E_old):7d} E_new={len(E_new):7d} "
                  f"gained={gained:7d} zeros_old={zeros_old:9d} pct={pct:.4f} lambda={dango_lambda(pct)}")
    return pd.DataFrame(rows)


def dataset_split() -> pd.DataFrame:
    idx = json.load(open(osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")))
    labels = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    perturbed = json.load(open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")))
    counts = json.load(open(osp.join(BUILD_DIR, "processed", "perturbation_count_index.json")))
    gene_set = json.load(open(osp.join(BUILD_DIR, "processed", "gene_set.json")))
    y = labels["gene_interaction"].dropna()
    row = {
        "records": len(labels),
        "train": len(idx["train"]),
        "val": len(idx["val"]),
        "test": len(idx["test"]),
        "perturbed_genes": len(perturbed),
        "vocabulary_genes": len(gene_set),
        "perturbation_sizes": ";".join(f"{k}:{len(v)}" for k, v in sorted(counts.items())),
        "labels_negative": int((y < 0).sum()),
        "labels_zero": int((y == 0).sum()),
        "labels_positive": int((y > 0).sum()),
        "label_min": float(y.min()),
        "label_max": float(y.max()),
        "label_mean": float(y.mean()),
        "label_sd": float(y.std(ddof=1)),
        "random_seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
    }
    print(pd.Series(row).to_string())
    return pd.DataFrame([row])


def panel(df: pd.DataFrame):
    """Half-width panel: percent decreased zeros per channel for the two release transitions,
    the 1% rule as a dashed line, the resulting lambda printed over each bar, and the two
    values the DANGO paper reports as open markers."""
    w, h = PANEL_WIDTHS_MM["half"], 52.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.28)
    x = np.arange(len(CHANNELS))
    bw = 0.36
    for i, (old, new) in enumerate(TRANSITIONS):
        sub = df[df["transition"] == f"{old}->{new}"].set_index("channel").loc[CHANNELS]
        xs = x + (i - 0.5) * bw
        ax.bar(xs, sub["pct_decreased_zeros"], bw, color=VERSION_COLOR[old], edgecolor="black",
               linewidth=0.4, label=f"{VERSION_LABEL[old]} to {VERSION_LABEL[new]}")
        for xi, pct, lam in zip(xs, sub["pct_decreased_zeros"], sub["lambda"]):
            ax.text(xi, pct + 0.12, f"{lam:g}", ha="center", va="bottom", fontsize=5)
    for ch, pct in DANGO_PAPER_PCT.items():
        ax.scatter(CHANNELS.index(ch) - 0.5 * bw, pct, s=9, marker="o", facecolor="white",
                   edgecolor="black", linewidth=0.5, zorder=5)
    ax.axhline(THRESHOLD_PCT, color="black", linestyle=(0, (3, 2)), linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABEL[c] for c in CHANNELS], rotation=25, ha="right", rotation_mode="anchor")
    ax.set_xlabel("STRING channel")
    ax.set_ylabel("Decreased zeros (%)")
    ax.set_ylim(0, 6.3)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=2, width=0.5)
    ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=VERSION_COLOR[old], edgecolor="black", linewidth=0.4,
              label=f"{VERSION_LABEL[old]} to {VERSION_LABEL[new]}")
        for old, new in TRANSITIONS
    ]
    handles.append(Line2D([], [], color="black", linestyle=(0, (3, 2)), linewidth=0.6,
                          label="1% rule: above, λ = 0.1; below, λ = 1"))
    handles.append(Line2D([], [], marker="o", markersize=3, markerfacecolor="white",
                          markeredgecolor="black", linestyle="none", label="DANGO paper (v9.1 to v11.0)"))
    ax.legend(handles=handles, frameon=False, loc="upper right", handlelength=1.4, handletextpad=0.5,
              labelspacing=0.3)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, "dango_decreased_zeros.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, "dango_decreased_zeros.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true", help="re-render from the frozen CSVs")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.from_csv:
        df = pd.read_csv(ZEROS_CSV)
    else:
        from torchcell.graph import SCerevisiaeGraph
        from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

        plt.rcParams.update(RC)  # torchcell.graph applies the repo mplstyle on import
        genome = SCerevisiaeGenome(
            genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
            go_root=osp.join(DATA_ROOT, "data/go"),
            overwrite=False,
        )
        builder = SCerevisiaeGraph(
            sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
            string_root=osp.join(DATA_ROOT, "data/string"),
            tflink_root=osp.join(DATA_ROOT, "data/tflink"),
            genome=genome,
        )
        df = decreased_zeros(builder)
        df.to_csv(ZEROS_CSV, index=False)
        print(f"wrote {ZEROS_CSV}")
        split = dataset_split()
        split.to_csv(SPLIT_CSV, index=False)
        print(f"wrote {SPLIT_CSV}")
    panel(df)


if __name__ == "__main__":
    main()
