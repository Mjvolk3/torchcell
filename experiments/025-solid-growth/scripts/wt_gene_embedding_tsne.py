# experiments/025-solid-growth/scripts/wt_gene_embedding_tsne.py
# [[experiments.025-solid-growth.scripts.wt_gene_embedding_tsne]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/wt_gene_embedding_tsne
"""t-SNE views of the wild-type encoder gene embedding, beside the UMAP in Figure 2.

Reads the encoder embedding the clustering script saved (encoder_H_genes.npy, one row per
gene in gene_order.json) and the cluster assignments and training coverage from
cluster_assignments.csv, then draws a grid of t-SNE embeddings over perplexity and early
exaggeration, each colored by training coverage (the arc of the UMAP) and by encoder
cluster. The point is to see whether the genes in no training triple still form a tail at
other settings, and to find a setting that gives the trained cluster the room of the panel.
PCA to 50 dimensions first, as the UMAP had. Writes a timestamped PNG and SVG to
ASSET_IMAGES_DIR/025-solid-growth and the embeddings to results/wt_gene_embeddings/tsne.npz.
"""
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torchcell.timestamp import timestamp  # noqa: E402
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in  # noqa: E402

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
WT = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results", "wt_gene_embeddings")
IMG_DIR = osp.join(os.environ["ASSET_IMAGES_DIR"], "025-solid-growth")
ORANGE, RED, PURPLE, YELLOW, BLUE, GRAY = PLOT_PALETTE[:6]
CLUSTER_COLORS = [ORANGE, RED, PURPLE, YELLOW, BLUE]
plt.rcParams.update({"font.family": "Arial", "font.size": 6, "axes.titlesize": 6, "svg.fonttype": "none"})

CHOSEN: dict[str, Any] = {"perplexity": 30, "early_exaggeration": 4.0}
SETTINGS: list[dict[str, Any]] = [
    {"perplexity": 30, "early_exaggeration": 12.0},
    {"perplexity": 50, "early_exaggeration": 12.0},
    {"perplexity": 100, "early_exaggeration": 12.0},
    {"perplexity": 30, "early_exaggeration": 4.0},
    {"perplexity": 100, "early_exaggeration": 4.0},
    {"perplexity": 200, "early_exaggeration": 4.0},
]


def main() -> None:
    H = np.load(osp.join(WT, "encoder_H_genes.npy"))
    genes = json.load(open(osp.join(WT, "gene_order.json")))
    assign = pd.read_csv(osp.join(WT, "cluster_assignments.csv")).set_index("gene").loc[genes]
    enr = pd.read_csv(osp.join(WT, "go_enrichment.csv"))
    best = enr[enr.representation == "encoder"].sort_values("q").groupby("cluster", as_index=False).first().sort_values("q")
    top = [int(c) for c in best[best.q < 0.05].cluster.head(len(CLUSTER_COLORS))]
    color_of = {c: CLUSTER_COLORS[i] for i, c in enumerate(top)}
    it = assign.in_train_triple.to_numpy(dtype=bool)
    lab = assign.encoder_cluster.to_numpy()
    X = PCA(n_components=50, random_state=0).fit_transform(H)
    print(f"genes {H.shape[0]} dims {H.shape[1]} -> PCA 50, explained {PCA(n_components=50, random_state=0).fit(H).explained_variance_ratio_.sum():.3f}", flush=True)

    fig, axes = plt.subplots(2, len(SETTINGS), figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62)))
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.9, wspace=0.08, hspace=0.18)
    out: dict[str, np.ndarray] = {}
    for j, st in enumerate(SETTINGS):
        t0 = time.time()
        Y = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0, **st).fit_transform(X)
        key = f"p{st['perplexity']}_e{st['early_exaggeration']:g}"
        out[key] = Y
        # How much of the panel the trained genes get: the span of their points over the span of all.
        span = np.ptp(Y[it], axis=0) / np.ptp(Y, axis=0)
        print(f"{key}: {time.time() - t0:.0f} s, trained-gene span of panel x {span[0]:.2f} y {span[1]:.2f}", flush=True)
        for i, ax in enumerate(axes[:, j]):
            if i == 0:
                ax.scatter(Y[~it, 0], Y[~it, 1], s=1.0, color="#C8C8C8", linewidths=0, rasterized=True, zorder=2)
                ax.scatter(Y[it, 0], Y[it, 1], s=1.0, color=RED, linewidths=0, rasterized=True, zorder=3)
                ax.set_title(f"perplexity {st['perplexity']}\nexaggeration {st['early_exaggeration']:g}\ntrained span {span[0]:.2f} x {span[1]:.2f}", fontsize=5)
            else:
                other = ~np.isin(lab, top)
                ax.scatter(Y[other, 0], Y[other, 1], s=1.0, color="#C8C8C8", linewidths=0, rasterized=True, zorder=2)
                for c in top:
                    m = lab == c
                    ax.scatter(Y[m, 0], Y[m, 1], s=1.0, color=color_of[c], linewidths=0, rasterized=True, zorder=3)
                    ax.text(np.median(Y[m, 0]), np.median(Y[m, 1]), str(c), fontsize=6, fontweight="bold", ha="center", va="center", zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].text(-0.02, 0.5, "training coverage", transform=axes[0, 0].transAxes, rotation=90, ha="right", va="center")
    axes[1, 0].text(-0.02, 0.5, "encoder clusters", transform=axes[1, 0].transAxes, rotation=90, ha="right", va="center")
    fig.text(0.5, 0.01, "t-SNE of the wild-type encoder embedding, 6,607 genes, PCA 50 first; top: red = in a training triple, gray = in none; bottom: the five best-enriched encoder clusters", ha="center", va="bottom", fontsize=6)
    np.savez(osp.join(WT, "tsne.npz"), **out)
    # The setting the sheet draws (review 2026-10-13: perplexity 30, exaggeration 4), for the
    # encoder and for the input gene table under the same PCA-50 reduction, one row per gene.
    chosen = CHOSEN
    T = np.load(osp.join(WT, "gene_table.npy"))
    Yt = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=0, **chosen).fit_transform(PCA(n_components=50, random_state=0).fit_transform(T))
    Ye = out[f"p{chosen['perplexity']}_e{chosen['early_exaggeration']:g}"]
    coords = pd.DataFrame({"gene": genes, "encoder_tsne_1": Ye[:, 0], "encoder_tsne_2": Ye[:, 1], "table_tsne_1": Yt[:, 0], "table_tsne_2": Yt[:, 1]})
    coords.to_csv(osp.join(WT, "tsne_coords.csv"), index=False)
    print(f"wrote tsne_coords.csv at perplexity {chosen['perplexity']}, exaggeration {chosen['early_exaggeration']:g}")
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = osp.join(IMG_DIR, f"wt_gene_embedding_tsne_{timestamp()}")
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".svg")
    print("wrote", stem + ".png")


if __name__ == "__main__":
    sys.exit(main())
