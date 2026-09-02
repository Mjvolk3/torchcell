# experiments/010-kuzmin-tmi/scripts/additive_vs_transformer_inference_space.py
# [[experiments.010-kuzmin-tmi.scripts.additive_vs_transformer_inference_space]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/additive_vs_transformer_inference_space

"""Does the additive null rank the 010 design space the way the transformer does?

This is the experiment 010 analogue of Figure 1 of Visani, Verma & DeWitt
(bioRxiv 2026.04.23.719915), which shows that MULTI-evolve's neural network and
a plain additive model produce near-identical predictions across the
combinatorial variant space, so that the engineering selection made from the
network was reproducible from the additive model.

Here the combinatorial space is the 465,735,532 unmeasured yeast triples scored
by checkpoint ``c7671wgj`` for the 010 panel selection. The transformer's
prediction for each triple is on disk; this script scores the same triples with
the additive per-gene ridge fit by
``additive_baseline_gene_interaction.py`` and reports:

  * the Pearson and Spearman correlation between the two predictors over the
    whole space, streamed row group by row group,
  * the overlap of the top-K and bottom-K sets the two predictors nominate,
    which is what the panel selection actually consumed,
  * a subsampled joint scatter matching the paper's figure.

A high correlation plus a high top-K overlap would mean the triples 010 chose to
construct were recoverable without the transformer. A low overlap would mean the
transformer's ranking carries information the additive model does not have.
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")
INFERENCE_PARQUET = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/010-kuzmin-tmi/inference_3/inferred",
    "models-checkpoints-compute-3-3-2036902_bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5"
    "f60f100d6dc0e0288cd97e366fc15e-c7671wgj-best-pearson-epoch=24-val-"
    "gene_interaction-Pearson=0.4619.parquet",
)
COEF_NPZ = osp.join(RESULTS_DIR, "additive_baseline_B1_coefficients.npz")
BUILD_DIR = osp.join(
    DATA_ROOT, "data/torchcell/experiments/010-kuzmin-tmi/001-small-build"
)

TOP_K = [100, 1_000, 10_000, 100_000]
SUBSAMPLE_PER_GROUP = 4_000
RNG = np.random.default_rng(0)

# Triples are bucketed by the training support of their least-observed gene, to
# separate "the two models disagree" from "neither model was given any data".
SUPPORT_BINS = [0, 25, 50, 100, 200, 400, 800, 1 << 30]


def train_gene_counts(gene_names: list[str]) -> np.ndarray:
    """Training-split observation count for each gene in the additive fit."""
    with open(osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")) as f:
        gene_index = json.load(f)
    with open(osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")) as f:
        train_ids = set(json.load(f)["train"])
    return np.array(
        [sum(1 for r in gene_index[g] if r in train_ids) for g in gene_names],
        dtype=np.int64,
    )


class Moments:
    """Streaming co-moments for a Pearson correlation."""

    def __init__(self) -> None:
        """Start with an empty accumulator."""
        self.n = 0
        self.sa = self.sb = self.saa = self.sbb = self.sab = 0.0

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        """Fold one aligned block of paired observations into the accumulator."""
        self.n += a.size
        self.sa += a.sum()
        self.sb += b.sum()
        self.saa += float(a @ a)
        self.sbb += float(b @ b)
        self.sab += float(a @ b)

    def pearson(self) -> float:
        """Correlation of everything added so far, NaN if it is undefined."""
        if self.n < 2:
            return float("nan")
        cov = self.sab / self.n - (self.sa / self.n) * (self.sb / self.n)
        va = self.saa / self.n - (self.sa / self.n) ** 2
        vb = self.sbb / self.n - (self.sb / self.n) ** 2
        if va <= 0 or vb <= 0:
            return float("nan")
        return cov / np.sqrt(va * vb)


class TopK:
    """Streaming top-k over (value, row id), merged with numpy rather than a heap."""

    def __init__(self, k: int) -> None:
        """Track the k largest values seen, with their row ids."""
        self.k = k
        self.vals = np.empty(0)
        self.rows = np.empty(0, dtype=np.int64)

    def push(self, values: np.ndarray, rows: np.ndarray) -> None:
        """Merge one block of candidates into the running top-k."""
        if values.size > self.k:
            sel = np.argpartition(values, -self.k)[-self.k :]
            values, rows = values[sel], rows[sel]
        vals = np.concatenate([self.vals, values])
        ids = np.concatenate([self.rows, rows])
        if vals.size > self.k:
            sel = np.argpartition(vals, -self.k)[-self.k :]
            vals, ids = vals[sel], ids[sel]
        self.vals, self.rows = vals, ids

    def ranked(self) -> np.ndarray:
        """Row ids, best first."""
        return self.rows[np.argsort(-self.vals)]


def main() -> None:
    coef = np.load(COEF_NPZ, allow_pickle=True)
    gene_names = list(coef["gene_names"])
    beta = coef["beta"]
    intercept = float(coef["intercept"])
    gene_to_col = {g: j for j, g in enumerate(gene_names)}
    print(f"additive model: {len(gene_names)} genes, alpha={float(coef['alpha'])}")

    counts = train_gene_counts(gene_names)
    print(
        f"training support per gene: median {int(np.median(counts))}  "
        f"min {int(counts.min())}  max {int(counts.max())}"
    )
    support_moments = {i: Moments() for i in range(len(SUPPORT_BINS) - 1)}

    gene_cols = ["gene1", "gene2", "gene3"]
    pf = pq.ParquetFile(INFERENCE_PARQUET, read_dictionary=gene_cols)
    print(f"inference space: {pf.metadata.num_rows} rows, {pf.num_row_groups} groups")

    # Streaming accumulators for Pearson over the full space.
    n = 0
    s_a = s_t = s_aa = s_tt = s_at = 0.0
    kmax = max(TOP_K)
    top_add, top_cgt = TopK(kmax), TopK(kmax)
    bot_add, bot_cgt = TopK(kmax), TopK(kmax)
    sub_add: list[np.ndarray] = []
    sub_cgt: list[np.ndarray] = []
    missing = 0
    offset = 0

    for gi in range(pf.num_row_groups):
        table = pf.read_row_group(gi, columns=gene_cols + ["prediction"])
        cgt = (
            table.column("prediction").to_numpy(zero_copy_only=False).astype(np.float64)
        )
        add = np.full(cgt.size, intercept)
        ok = np.ones(cgt.size, dtype=bool)
        support = np.full(cgt.size, np.inf)
        for col in gene_cols:
            chunk = table.column(col).combine_chunks()
            # Map the small dictionary once, then gather by code.
            vocab = chunk.dictionary.to_pylist()
            lookup = np.array(
                [beta[gene_to_col[g]] if g in gene_to_col else np.nan for g in vocab]
            )
            sup = np.array(
                [counts[gene_to_col[g]] if g in gene_to_col else np.nan for g in vocab]
            )
            codes = chunk.indices.to_numpy(zero_copy_only=False).astype(np.int64)
            contrib = lookup[codes]
            ok &= ~np.isnan(contrib)
            add += np.nan_to_num(contrib)
            support = np.minimum(support, np.nan_to_num(sup[codes], nan=np.inf))

        missing += int((~ok).sum())

        # Only triples whose three genes all have an additive coefficient are
        # comparable; the rest carry no additive prediction at all.
        cgt, add, support = cgt[ok], add[ok], support[ok]
        rows = offset + np.nonzero(ok)[0]

        bucket = np.digitize(support, SUPPORT_BINS[1:-1])
        for b in np.unique(bucket):
            sel = bucket == b
            support_moments[int(b)].add(add[sel], cgt[sel])

        n += cgt.size
        s_a += add.sum()
        s_t += cgt.sum()
        s_aa += float(add @ add)
        s_tt += float(cgt @ cgt)
        s_at += float(add @ cgt)

        top_add.push(add, rows)
        top_cgt.push(cgt, rows)
        bot_add.push(-add, rows)
        bot_cgt.push(-cgt, rows)

        pick = RNG.choice(
            cgt.size, size=min(SUBSAMPLE_PER_GROUP, cgt.size), replace=False
        )
        sub_add.append(add[pick])
        sub_cgt.append(cgt[pick])

        offset += ok.size
        if gi % 25 == 0:
            print(f"  row group {gi}/{pf.num_row_groups}  comparable rows so far {n}")

    print(f"triples dropped for an out-of-vocabulary gene: {missing}")

    cov = s_at / n - (s_a / n) * (s_t / n)
    r = cov / np.sqrt((s_aa / n - (s_a / n) ** 2) * (s_tt / n - (s_t / n) ** 2))
    print(f"\nfull-space Pearson(additive, transformer) = {r:.4f} over {n} triples")

    sub_add_arr = np.concatenate(sub_add)
    sub_cgt_arr = np.concatenate(sub_cgt)
    sub_r = float(pearsonr(sub_add_arr, sub_cgt_arr)[0])
    sub_rho = float(spearmanr(sub_add_arr, sub_cgt_arr)[0])
    print(
        f"subsample (n={sub_add_arr.size}): Pearson {sub_r:.4f}  Spearman {sub_rho:.4f}"
    )

    rows: list[dict[str, object]] = [
        {"quantity": "n_triples_comparable", "value": n},
        {"quantity": "n_triples_dropped_oov_gene", "value": missing},
        {"quantity": "pearson_full_space", "value": r},
        {"quantity": "subsample_n", "value": sub_add_arr.size},
        {"quantity": "pearson_subsample", "value": sub_r},
        {"quantity": "spearman_subsample", "value": sub_rho},
    ]

    print("\nagreement stratified by training support of the least-observed gene")
    for b, mom in support_moments.items():
        lo, hi = SUPPORT_BINS[b], SUPPORT_BINS[b + 1]
        hi_label = "inf" if hi == 1 << 30 else str(hi)
        print(
            f"  support [{lo:>4}, {hi_label:>4})  n={mom.n:>12}  "
            f"Pearson {mom.pearson(): .4f}"
        )
        rows.append(
            {
                "quantity": f"pearson_support_{lo}_{hi_label}",
                "value": mom.pearson(),
            }
        )
        rows.append({"quantity": f"n_support_{lo}_{hi_label}", "value": mom.n})

    add_top_sorted = top_add.ranked()
    cgt_top_sorted = top_cgt.ranked()
    add_bot_sorted = bot_add.ranked()
    cgt_bot_sorted = bot_cgt.ranked()
    for k in TOP_K:
        top_overlap = np.intersect1d(add_top_sorted[:k], cgt_top_sorted[:k]).size / k
        bot_overlap = np.intersect1d(add_bot_sorted[:k], cgt_bot_sorted[:k]).size / k
        print(
            f"K={k:>9}  top overlap {top_overlap:.4f}   bottom overlap {bot_overlap:.4f}"
        )
        rows.append({"quantity": f"top_{k}_overlap", "value": top_overlap})
        rows.append({"quantity": f"bottom_{k}_overlap", "value": bot_overlap})

    out = osp.join(RESULTS_DIR, "additive_vs_transformer_inference_space.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")

    np.savez_compressed(
        osp.join(RESULTS_DIR, "additive_vs_transformer_subsample.npz"),
        additive=sub_add_arr,
        transformer=sub_cgt_arr,
    )
    plot(sub_add_arr, sub_cgt_arr, r)


def plot(add: np.ndarray, cgt: np.ndarray, r: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(80.0))
    )
    ax.hexbin(add, cgt, gridsize=90, bins="log", cmap="magma_r", linewidths=0)
    lo = float(min(add.min(), cgt.min()))
    hi = float(max(add.max(), cgt.max()))
    ax.plot([lo, hi], [lo, hi], color=PLOT_PALETTE[5], linewidth=0.5, linestyle="--")
    ax.set_xlabel("Additive per-gene ridge prediction")
    ax.set_ylabel("Transformer prediction (c7671wgj)")
    ax.set_title(f"010 inference space, Pearson r = {r:.3f}", fontsize=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()

    stem = osp.join(
        ASSET_IMAGES_DIR, "010-kuzmin-tmi", "additive_vs_transformer_inference_space"
    )
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
