# experiments/010-kuzmin-tmi/scripts/additive_baseline_gene_interaction.py
# [[experiments.010-kuzmin-tmi.scripts.additive_baseline_gene_interaction]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/additive_baseline_gene_interaction
"""Additive / low-order null baselines for the 010 trigenic interaction score.

Motivated by Visani, Verma & DeWitt (bioRxiv 2026.04.23.719915), "Additive
baselines furnish no evidence for epistasis learning by MULTI-evolve", which
shows that a neural network claimed to learn epistasis is reproduced almost
exactly by a ridge-regularized additive model over mutation indicators.

The analogous question for experiment 010 is: can the equivariant cell graph
transformer's held-out performance on the trigenic interaction score be matched
by a model with no capacity to represent gene-gene interaction beyond what is
already written into its feature space?

Baselines fit here, all on the SAME train/val/test split the transformer used
(``index_seed_42.json``), all predicting the same label (``gene_interaction``):

    B0  train-mean               intercept only, no capacity at all
    B1  additive (per-gene)      y = b0 + sum_i beta_i x_i, ridge
    B2  additive + pair          B1 plus one coefficient per observed unordered
                                 gene pair, ridge
    B3  hierarchical mean        unregularized empirical mean encoding, pair
                                 mean backing off to gene mean backing off to
                                 the global mean
    B4  query-pair only          ridge on recurring gene pairs alone, with no
                                 per-gene term; isolates how much of the signal
                                 is carried by the Kuzmin query-double identity
                                 (i.e. screen batch) rather than by gene content
    B5  nonlinear, same features an embedding-sum MLP over the identical
                                 one-hot gene features as B1; adds nonlinearity
                                 and learned gene embeddings but still sees no
                                 interaction network

B1 is the direct analogue of the DeWitt additive null. It is a strictly
harder null here than in the protein setting: the trigenic interaction score
is itself defined as a residual after the single- and double-mutant
expectations have been subtracted, so the additive-in-single-gene-effects
component of fitness is removed by construction before the model sees it.

B5 is the analogue of DeWitt's zero-hidden-layer extension of the MULTI-evolve
grid, run in the other direction: it holds the feature space fixed at B1's and
asks what nonlinearity alone buys. The gap B5 -> transformer is then
attributable to the nine interaction graphs, not to model capacity.

Inputs are read from the pinned 010 build, not through the dataset loaders:
    processed/label_df.parquet                  index -> gene_interaction
    processed/is_any_perturbed_gene_index.json  gene  -> [record index, ...]
    data_module_cache/index_seed_42.json        split -> [record index, ...]

Transformer reference numbers are read from the three checkpoints' own
re-evaluation runs under ``$DATA_ROOT/wandb-experiments`` rather than retyped.
"""

import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from scipy.sparse.linalg import lsqr
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

BUILD_DIR = osp.join(
    DATA_ROOT,
    "data/torchcell/experiments/010-kuzmin-tmi/001-small-build",
)
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

# Re-evaluation runs of the three 010 best-Pearson checkpoints. These carry both
# val and test metrics for the transformer, which the calibration table does not.
CGT_EVAL_RUNS = {
    "CGT_M01_lzs9pcj3": (
        "eval_compute-3-3-2027905_a1260b50c3d74b6b7acea919b89416feb6f"
        "c957b3023c9ac866f90378df82625/wandb/run-20260107_185127-leodrxht"
    ),
    "CGT_M02_yv4r30bi": (
        "eval_compute-3-3-2027907_a1260b50c3d74b6b7acea919b89416feb6f"
        "c957b3023c9ac866f90378df82625/wandb/run-20260109_105320-cvu2ryfw"
    ),
    "CGT_M03_c7671wgj": (
        "eval_compute-3-3-2036902_bd9e6c666ea1c0e7d1bbb6321fbc4d3bd5f6"
        "0f100d6dc0e0288cd97e366fc15e/wandb/run-20260107_201234-0psour3n"
    ),
}
WANDB_ROOT = osp.join(DATA_ROOT, "wandb-experiments")

MLP_SEEDS = [0, 1, 2]

# Ridge penalties swept; the value is chosen on val and then reported on test.
ALPHA_GRID = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]


def load_records() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Return (row_genes, y, splits, gene_names) aligned on a dense row index."""
    label_df = pd.read_parquet(osp.join(BUILD_DIR, "processed", "label_df.parquet"))
    with open(
        osp.join(BUILD_DIR, "processed", "is_any_perturbed_gene_index.json")
    ) as f:
        gene_index = json.load(f)
    with open(
        osp.join(BUILD_DIR, "data_module_cache", "index_seed_42.json")
    ) as f:
        split_index = json.load(f)

    # Dense row numbering over the record indices that carry a label.
    record_ids = label_df["index"].to_numpy()
    order = np.argsort(record_ids)
    record_ids = record_ids[order]
    y = label_df["gene_interaction"].to_numpy()[order]
    id_to_row = {int(r): i for i, r in enumerate(record_ids)}

    gene_names = sorted(gene_index.keys())
    gene_to_col = {g: j for j, g in enumerate(gene_names)}

    # Each record carries exactly three perturbed genes; fill a (n, 3) table.
    row_genes = np.full((len(record_ids), 3), -1, dtype=np.int32)
    fill = np.zeros(len(record_ids), dtype=np.int8)
    for gene, ids in gene_index.items():
        col = gene_to_col[gene]
        for rid in ids:
            row = id_to_row[int(rid)]
            row_genes[row, fill[row]] = col
            fill[row] += 1
    assert (fill == 3).all(), "every 010 record must carry exactly 3 perturbed genes"

    splits = {
        name: np.array([id_to_row[int(r)] for r in ids], dtype=np.int64)
        for name, ids in split_index.items()
    }
    return row_genes, y, splits, gene_names


def gene_matrix(row_genes: np.ndarray, n_genes: int) -> sp.csr_matrix:
    """One-hot gene incidence, three ones per row."""
    n = row_genes.shape[0]
    rows = np.repeat(np.arange(n), 3)
    cols = row_genes.reshape(-1)
    data = np.ones(rows.size, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n_genes))


def pair_keys(row_genes: np.ndarray) -> np.ndarray:
    """Unordered gene-pair keys, shape (n, 3), encoded as lo * BASE + hi."""
    base = np.int64(row_genes.max()) + 1
    a, b, c = row_genes[:, 0], row_genes[:, 1], row_genes[:, 2]
    pairs = np.stack(
        [
            np.minimum(a, b) * base + np.maximum(a, b),
            np.minimum(a, c) * base + np.maximum(a, c),
            np.minimum(b, c) * base + np.maximum(b, c),
        ],
        axis=1,
    ).astype(np.int64)
    return pairs


def pair_matrix(
    pairs: np.ndarray, vocab: dict[int, int]
) -> sp.csr_matrix:
    """Incidence over a fixed pair vocabulary; pairs outside it are dropped."""
    n = pairs.shape[0]
    rows, cols = [], []
    for k in range(3):
        col = np.array([vocab.get(int(p), -1) for p in pairs[:, k]], dtype=np.int64)
        keep = col >= 0
        rows.append(np.nonzero(keep)[0])
        cols.append(col[keep])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(rows.size, dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, len(vocab)))


def ridge_fit(
    x: sp.csr_matrix, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    """Ridge with an unpenalized intercept, solved by LSQR on the augmented system."""
    intercept = float(y.mean())
    target = y - intercept
    beta = lsqr(x, target, damp=np.sqrt(alpha), atol=1e-10, btol=1e-10, iter_lim=2000)[0]
    return beta, intercept


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.allclose(y_pred, y_pred[0]):
        pearson, spearman = 0.0, 0.0
    else:
        pearson = float(pearsonr(y_true, y_pred)[0])
        spearman = float(spearmanr(y_true, y_pred)[0])
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def hierarchical_mean(
    row_genes: np.ndarray,
    pairs: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    """Pair mean -> gene mean -> global mean, all estimated on train only."""
    global_mean = float(y[tr].mean())

    gene_sum = np.zeros(int(row_genes.max()) + 1)
    gene_cnt = np.zeros_like(gene_sum)
    np.add.at(gene_sum, row_genes[tr].reshape(-1), np.repeat(y[tr], 3))
    np.add.at(gene_cnt, row_genes[tr].reshape(-1), 1.0)

    pair_sum: dict[int, float] = {}
    pair_cnt: dict[int, int] = {}
    for p, val in zip(pairs[tr].reshape(-1), np.repeat(y[tr], 3)):
        key = int(p)
        pair_sum[key] = pair_sum.get(key, 0.0) + float(val)
        pair_cnt[key] = pair_cnt.get(key, 0) + 1

    out = np.empty(idx.size)
    for i, row in enumerate(idx):
        vals = []
        for p in pairs[row]:
            key = int(p)
            # Require a pair to recur before trusting its mean.
            if pair_cnt.get(key, 0) >= 5:
                vals.append(pair_sum[key] / pair_cnt[key])
        if not vals:
            for g in row_genes[row]:
                if gene_cnt[g] >= 5:
                    vals.append(gene_sum[g] / gene_cnt[g])
        out[i] = float(np.mean(vals)) if vals else global_mean
    return out


def embedding_mlp(
    row_genes: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    n_genes: int,
    dim: int = 128,
    hidden: int = 256,
    epochs: int = 60,
    seed: int = 0,
) -> tuple[np.ndarray, int]:
    """Sum-pooled gene embeddings + MLP, on exactly B1's feature space.

    Returns predictions for the whole record set and the early-stopping epoch.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu, sd = float(y[tr].mean()), float(y[tr].std(ddof=0))
    genes = torch.from_numpy(row_genes.astype(np.int64))
    target = torch.from_numpy(((y - mu) / sd).astype(np.float32))

    model = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, 1),
    ).to(device)
    emb = nn.Embedding(n_genes, dim).to(device)
    nn.init.normal_(emb.weight, std=0.02)
    opt = torch.optim.AdamW(
        list(emb.parameters()) + list(model.parameters()), lr=3e-3, weight_decay=1e-4
    )

    g_tr = genes[tr].to(device)
    t_tr = target[tr].to(device)
    g_all = genes.to(device)
    batch = 4096

    best = (-1.0, None, -1)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(g_tr.shape[0], device=device)
        for start in range(0, perm.numel(), batch):
            sel = perm[start : start + batch]
            pooled = emb(g_tr[sel]).sum(dim=1)
            loss = nn.functional.mse_loss(model(pooled).squeeze(-1), t_tr[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            chunks = []
            for start in range(0, g_all.shape[0], 65536):
                pooled = emb(g_all[start : start + 65536]).sum(dim=1)
                chunks.append(model(pooled).squeeze(-1).cpu())
            pred = torch.cat(chunks).numpy() * sd + mu
        r = float(pearsonr(y[va], pred[va])[0])
        if r > best[0]:
            best = (r, pred, epoch)
        if epoch % 10 == 0:
            print(f"B5_mlp epoch {epoch:>3d}  val pearson {r:.4f}")

    print(f"B5_mlp best val pearson {best[0]:.4f} at epoch {best[2]}")
    return best[1], best[2]


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    row_genes, y, splits, gene_names = load_records()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    n_genes = len(gene_names)

    print(f"records {y.size}  genes {n_genes}")
    print(f"train {tr.size}  val {va.size}  test {te.size}")
    print(f"label sd (train) {y[tr].std(ddof=0):.6f}")

    seen = np.unique(row_genes[tr])
    for name, idx in (("val", va), ("test", te)):
        unseen = np.setdiff1d(np.unique(row_genes[idx]), seen)
        print(f"{name}: genes unseen in train = {unseen.size}")

    pairs = pair_keys(row_genes)
    train_pairs, train_counts = np.unique(pairs[tr].reshape(-1), return_counts=True)
    print(
        f"distinct gene pairs in train {train_pairs.size}  "
        f"recurring (>=5 obs) {int((train_counts >= 5).sum())}  "
        f"max pair count {int(train_counts.max())}"
    )

    xg = gene_matrix(row_genes, n_genes)
    # Only pairs that recur can support a coefficient; singletons are noise.
    vocab = {int(p): j for j, p in enumerate(train_pairs[train_counts >= 5])}
    xp = pair_matrix(pairs, vocab)
    xgp = sp.hstack([xg, xp]).tocsr()
    print(f"B2 design: {xgp.shape[1]} columns ({n_genes} gene + {len(vocab)} pair)")
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        covered = np.asarray(xp[idx].sum(axis=1)).ravel() > 0
        print(f"{name}: records carrying a recurring pair = {covered.mean():.3%}")

    rows: list[dict[str, object]] = []

    # --- B0: train mean -------------------------------------------------
    mean_pred = np.full(y.size, y[tr].mean())
    for name, idx in (("val", va), ("test", te)):
        rows.append(
            {"model": "B0_train_mean", "alpha": None, "split": name}
            | score(y[idx], mean_pred[idx])
        )

    # --- B1 / B2: ridge, alpha chosen on val ----------------------------
    for tag, design in (
        ("B1_additive_gene", xg),
        ("B2_additive_plus_pair", xgp),
        ("B4_query_pair_only", xp),
    ):
        best = None
        for alpha in ALPHA_GRID:
            beta, b0 = ridge_fit(design[tr], y[tr], alpha)
            pred = design @ beta + b0
            val = score(y[va], pred[va])
            print(f"{tag} alpha={alpha:>8g}  val pearson {val['pearson']:.4f}")
            if best is None or val["pearson"] > best[0]:
                best = (val["pearson"], alpha, pred)
        _, alpha, pred = best
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            rows.append(
                {"model": tag, "alpha": alpha, "split": name}
                | score(y[idx], pred[idx])
            )
        np.save(osp.join(RESULTS_DIR, f"additive_baseline_pred_{tag}.npy"), pred)
        if tag == "B1_additive_gene":
            # Persist the additive coefficients so the same model can be scored
            # over the 010 inference design space by a downstream script.
            beta, b0 = ridge_fit(xg[tr], y[tr], alpha)
            np.savez(
                osp.join(RESULTS_DIR, "additive_baseline_B1_coefficients.npz"),
                gene_names=np.array(gene_names),
                beta=beta,
                intercept=b0,
                alpha=alpha,
            )

    # --- B3: hierarchical empirical mean --------------------------------
    for name, idx in (("val", va), ("test", te)):
        pred = hierarchical_mean(row_genes, pairs, y, tr, idx)
        rows.append(
            {"model": "B3_hierarchical_mean", "alpha": None, "split": name}
            | score(y[idx], pred)
        )

    # --- B5: nonlinear model on B1's feature space, replicated ----------
    for seed in MLP_SEEDS:
        pred, stop_epoch = embedding_mlp(row_genes, y, tr, va, te, n_genes, seed=seed)
        for name, idx in (("train", tr), ("val", va), ("test", te)):
            rows.append(
                {
                    "model": "B5_gene_embedding_mlp",
                    "alpha": None,
                    "split": name,
                    "seed": seed,
                }
                | score(y[idx], pred[idx])
            )
        np.save(
            osp.join(
                RESULTS_DIR, f"additive_baseline_pred_B5_gene_embedding_mlp_s{seed}.npy"
            ),
            pred,
        )
        print(f"B5 seed {seed} early-stopped at epoch {stop_epoch}")

    df = pd.DataFrame(rows)

    # --- transformer reference, read from its own eval runs -------------
    cgt_rows = []
    for tag, rel in CGT_EVAL_RUNS.items():
        path = osp.join(WANDB_ROOT, rel, "files", "wandb-summary.json")
        with open(path) as f:
            summary = json.load(f)
        for split in ("val", "test"):
            cgt_rows.append(
                {
                    "model": tag,
                    "alpha": None,
                    "split": split,
                    "pearson": summary[f"{split}/gene_interaction/Pearson"],
                    # Spearman is logged under the sampled-plot namespace only.
                    "spearman": summary[f"{split}_sample/Spearman_target_0"],
                    "mse": summary[f"{split}/gene_interaction/MSE"],
                    "rmse": summary[f"{split}/gene_interaction/RMSE"],
                }
            )
    df = pd.concat([df, pd.DataFrame(cgt_rows)], ignore_index=True)

    out_csv = osp.join(RESULTS_DIR, "additive_baseline_gene_interaction.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print(df.to_string(index=False))

    plot(df)


def plot(df: pd.DataFrame) -> None:
    val = df[df["split"] == "test"].copy()
    val = val.groupby("model", as_index=False).agg(
        pearson=("pearson", "mean"), sd=("pearson", "std")
    )
    labels = {
        "B0_train_mean": "Train mean",
        "B4_query_pair_only": "Query pair\nonly",
        "B3_hierarchical_mean": "Hierarchical\nempirical mean",
        "B1_additive_gene": "Additive\n(per-gene ridge)",
        "B2_additive_plus_pair": "Additive\n+ gene-pair ridge",
        "B5_gene_embedding_mlp": "Nonlinear MLP\n(same features)",
        "CGT_M01_lzs9pcj3": "CGT M01",
        "CGT_M02_yv4r30bi": "CGT M02",
        "CGT_M03_c7671wgj": "CGT M03",
    }
    val = val[val["model"].isin(labels)].copy()
    val["order"] = val["model"].map({m: i for i, m in enumerate(labels)})
    val = val.sort_values("order")
    val["label"] = val["model"].map(labels)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(
        figsize=(mm_to_in(PANEL_WIDTHS_MM["half"]), mm_to_in(55.0))
    )
    tier = {
        "B0_train_mean": PLOT_PALETTE[5],
        "B4_query_pair_only": PLOT_PALETTE[5],
        "B3_hierarchical_mean": PLOT_PALETTE[3],
        "B1_additive_gene": PLOT_PALETTE[0],
        "B2_additive_plus_pair": PLOT_PALETTE[1],
        "B5_gene_embedding_mlp": PLOT_PALETTE[2],
        "CGT_M01_lzs9pcj3": PLOT_PALETTE[4],
        "CGT_M02_yv4r30bi": PLOT_PALETTE[4],
        "CGT_M03_c7671wgj": PLOT_PALETTE[4],
    }
    colors = [tier[m] for m in val["model"]]
    ax.bar(
        np.arange(len(val)),
        val["pearson"].to_numpy(),
        yerr=val["sd"].fillna(0).to_numpy(),
        error_kw={"elinewidth": 0.5, "capthick": 0.5, "capsize": 1.5},
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(np.arange(len(val)))
    ax.set_xticklabels(val["label"], rotation=45, ha="right")
    ax.set_ylabel("Test Pearson r")
    ax.set_ylim(0, 0.55)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()

    stem = osp.join(
        ASSET_IMAGES_DIR, "010-kuzmin-tmi", "additive_baseline_test_pearson"
    )
    os.makedirs(osp.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


if __name__ == "__main__":
    main()
