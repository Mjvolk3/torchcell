# experiments/019-simb-multimodal/scripts/plot_residual_covariance.py
# [[experiments.019-simb-multimodal.scripts.plot_residual_covariance]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/plot_residual_covariance
"""Plot the residual gene-gene covariance diagnostic, and test whether it is CONDITIONAL.

`residual_covariance_diagnostic.py` established that residual gene-gene correlation is
REPRODUCIBLE (split-half r = 0.869 against a within-gene-shuffled null of 0.0001, effective
rank 32.8). That was the gate for adding the `energy` arm with a rank-32 covariance factor.

It did NOT establish that the structure is CONDITIONAL, and that distinction decides what a
joint head can buy:

    Sigma = D + V V^T  is GLOBAL -- one covariance shared by every strain. It changes the
    predictive DISTRIBUTION. It does NOT enter mu, so it cannot move a point metric such as
    `pearson_per_feature`.

So a global Sigma is worth its parameters only if the residual dependence is real (settled:
it is) -- but whether the model can do better still by making covariance depend on the
perturbation is a separate question, and one the original diagnostic hints at uncomfortably:
removing a genuine conditional mean (kNN over prot_T5) dropped the reproducible correlation
only from 0.8706 to 0.8687, i.e. by 0.2%. If conditioning on the perturbation barely changes
the correlation pattern, the structure is close to gene-intrinsic (co-expression modules,
array/batch effects) rather than perturbation-specific.

PANELS
  A  Split-half agreement -- C_A vs C_B off-diagonals (hexbin), observed vs shuffled null.
     The headline: does the correlation pattern replicate on held-out strains?
  B  Eigenspectrum -- cumulative variance vs rank, with the chosen k=32 and the
     participation-ratio effective rank marked. Says how much of the structure rank-k can hold.
  C  CONDITIONAL test (NEW) -- split strains by PERTURBED-GENE SIMILARITY rather than at
     random, into two groups whose deleted genes are far apart in prot_T5 space. If residual
     covariance is perturbation-specific, the two groups' correlation matrices should agree
     LESS than two random halves do. If they agree just as well, the covariance is global and
     a single shared Sigma is already the right model -- there is nothing conditional to gain.

Run from repo root (CPU only, no GPU needed):
    python experiments/019-simb-multimodal/scripts/plot_residual_covariance.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any

import lmdb
import numpy as np
from dotenv import load_dotenv

_WT_ENV = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(
    _WT_ENV
    if osp.exists(_WT_ENV)
    else osp.expanduser("~/Documents/projects/torchcell/.env")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)
from torchcell.utils.paths import asset_images_dir, experiment_results_dir  # noqa: E402

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0
KNN_K = 25
EMBEDDING = "prot_T5_all"
RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128]
CHOSEN_RANK = 32
N_SPLITS = 20  # random splits, to give the split-half statistic an error bar

def _apply_plot_style() -> None:
    """Apply the repo figure standards -- CALLED AT PLOT TIME, not at import.

    `torchcell.graph.graph` (pulled in transitively by NodeEmbeddingBuilder) runs
    `plt.style.use(style_file_path)` AT MODULE IMPORT, which silently overwrites
    `font.size` 6 -> 16. Setting rcParams at the top of this file therefore has no effect
    on anything drawn later: titles passed an explicit `fontsize=6` came out right while
    every rcParams-driven label rendered 2.67x too large. Applying the style here, after
    all imports and after the data load, is what makes the 6 pt Nature minimum stick.
    """
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.titlesize": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "svg.fonttype": "none",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _load_expression(base: str) -> tuple[list[str], np.ndarray]:
    """Return (perturbed gene per strain, Y [S, F]) for single-deletion strains."""
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    genes: list[str] = []
    rows: list[np.ndarray] = []
    keys: list[str] | None = None
    with env.begin() as txn:
        for _, value in txn.cursor():
            recs = json.loads(value.decode())
            if isinstance(recs, dict):
                recs = [recs]
            perts = recs[0]["experiment"]["genotype"]["perturbations"]
            if len(perts) != 1:
                continue
            for r in recs:
                ph = r["experiment"]["phenotype"]
                if ph["label_name"] != "expression_log2_ratio":
                    continue
                d = ph["expression_log2_ratio"]
                if keys is None:
                    keys = sorted(d)
                genes.append(perts[0]["systematic_gene_name"])
                rows.append(np.array([d[k] for k in keys], dtype=np.float32))
    env.close()
    return genes, np.stack(rows)


def _knn_loo(emb: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out similarity-weighted kNN prediction of each strain's vector."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    w = np.maximum(sim[rows, idx], 0.0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return np.einsum("nk,nkf->nf", w, Y[idx])


def _corr(R: np.ndarray) -> np.ndarray:
    """Gene-gene correlation across strains. R [n, F] -> [F, F]."""
    X = R - R.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = np.inf
    X = X / sd
    return (X.T @ X) / (X.shape[0] - 1)


def _offdiag(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(C.shape[0], k=1)
    return C[iu]


def _agreement(Ra: np.ndarray, Rb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    a, b = _offdiag(_corr(Ra)), _offdiag(_corr(Rb))
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), a[ok], b[ok]


def _shuffle_within_gene(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = R.copy()
    for f in range(out.shape[1]):
        rng.shuffle(out[:, f])
    return out


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    rng = np.random.default_rng(SEED)

    genes, Y = _load_expression(base)
    S, F = Y.shape
    print(f"expression strains S={S}  reporter genes F={F}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    built = NodeEmbeddingBuilder.build(
        embedding_names=[EMBEDDING], data_root=data_root, genome=genome, graph=None
    )
    lookup = {
        item.id: torch.cat([t.flatten() for t in item.embeddings.values()]).cpu().numpy()
        for item in built[EMBEDDING]
    }
    emb = np.stack([lookup[g] for g in genes])

    R = Y - _knn_loo(emb, Y, KNN_K)  # the honest residual: a real conditional mean removed
    results: dict[str, Any] = {"n_strains": S, "n_genes": F, "knn_k": KNN_K}

    # ---- Panel A: random split-half agreement, repeated for an error bar ----
    perm = rng.permutation(S)
    ia, ib = perm[: S // 2], perm[S // 2 :]
    obs, oa, ob = _agreement(R[ia], R[ib])
    Rn = _shuffle_within_gene(R, rng)
    null, na, nb = _agreement(Rn[ia], Rn[ib])
    print(f"random split-half: observed r={obs:.4f}  null r={null:.4f}")

    reps = []
    for _ in range(N_SPLITS):
        p = rng.permutation(S)
        reps.append(_agreement(R[p[: S // 2]], R[p[S // 2 :]])[0])
    print(f"  over {N_SPLITS} splits: mean {np.mean(reps):.4f}  sd {np.std(reps):.4f}")
    results["random_split"] = {
        "observed": obs,
        "null": null,
        "mean_over_splits": float(np.mean(reps)),
        "sd_over_splits": float(np.std(reps)),
        "n_splits": N_SPLITS,
    }

    # ---- Panel C: CONDITIONAL split -- group strains by perturbed-gene similarity ----
    # Project the deleted-gene embeddings on their leading principal direction and split at
    # the median. The two groups then perturb GENES THAT ARE FAR APART in prot_T5 space. If
    # residual covariance were perturbation-specific, their correlation matrices would agree
    # less than two random halves do; the random-split value is the matched control, since
    # both splits have the same group sizes and therefore the same sampling noise.
    E = emb - emb.mean(axis=0, keepdims=True)
    pc1 = np.linalg.svd(E, full_matrices=False)[2][0]
    score = E @ pc1
    order = np.argsort(score)
    ca, cb = order[: S // 2], order[S // 2 :]
    cond, _, _ = _agreement(R[ca], R[cb])
    sep = float(
        np.linalg.norm(emb[ca].mean(axis=0) - emb[cb].mean(axis=0))
        / np.linalg.norm(emb.std(axis=0))
    )
    print(f"conditional split (by perturbed-gene PC1): r={cond:.4f}  (separation {sep:.2f} sd)")
    results["conditional_split"] = {"agreement": cond, "embedding_separation_sd": sep}

    # ---- Panel B: eigenspectrum of the standardized residuals ----
    X = R - R.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = np.inf
    X = X / sd
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    cum = np.cumsum(ev) / ev.sum()
    eff = float((ev.sum() ** 2) / (ev**2).sum())
    results["effective_rank"] = eff
    results["cumulative_variance_by_rank"] = {
        str(k): float(cum[k - 1]) for k in RANK_GRID if k <= len(cum)
    }

    # ---------------- figure ----------------
    _apply_plot_style()
    # Width is STRICT (full 179 mm so the row tiles the Nature page); height is loose, but
    # three panels each carrying a title + both axis labels need ~68 mm -- at 52 mm
    # tight_layout cannot fit them and the labels collide and clip off the canvas.
    w = PANEL_WIDTHS_MM["full"]
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(w), mm_to_in(58)))

    # A: split-half hexbin
    ax = axes[0]
    sub = rng.choice(oa.size, size=min(300_000, oa.size), replace=False)
    ax.hexbin(na[sub], nb[sub], gridsize=45, cmap="Greys", mincnt=1, linewidths=0)
    ax.hexbin(
        oa[sub], ob[sub], gridsize=45, cmap="Oranges", mincnt=1, linewidths=0, alpha=0.75
    )
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.5, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("gene-gene corr, split half A")
    ax.set_ylabel("gene-gene corr, split half B")
    ax.set_title(f"A  reproducible  r={obs:.3f}\nnull r={null:.4f}", fontsize=6)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # B: eigenspectrum
    ax = axes[1]
    ks = np.arange(1, min(256, len(cum)) + 1)
    ax.plot(ks, 100 * cum[: len(ks)], color=PLOT_PALETTE[0], lw=1.0)
    ax.axvline(CHOSEN_RANK, color=PLOT_PALETTE[1], lw=0.8, ls="--")
    ax.axvline(eff, color=PLOT_PALETTE[2], lw=0.8, ls=":")
    ax.annotate(
        f"k={CHOSEN_RANK}\n{100 * cum[CHOSEN_RANK - 1]:.0f}%",
        xy=(CHOSEN_RANK, 100 * cum[CHOSEN_RANK - 1]),
        xytext=(CHOSEN_RANK * 1.6, 100 * cum[CHOSEN_RANK - 1] - 22),
        fontsize=5,
        color=PLOT_PALETTE[1],
        arrowprops={"arrowstyle": "-", "lw": 0.4, "color": PLOT_PALETTE[1]},
    )
    ax.annotate(
        f"eff. rank {eff:.1f}",
        xy=(eff, 8),
        xytext=(eff * 1.7, 6),
        fontsize=5,
        color=PLOT_PALETTE[2],
    )
    ax.set_xscale("log")
    ax.set_xlabel("rank k")
    ax.set_ylabel("cumulative residual variance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("B  how much rank-k can hold", fontsize=6)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.9")
    ax.set_axisbelow(True)

    # C: conditional vs random split
    ax = axes[2]
    labels = ["random\nsplit", "split by\nperturbed gene", "shuffled\nnull"]
    vals = [float(np.mean(reps)), cond, null]
    errs = [float(np.std(reps)), 0.0, 0.0]
    colors = [PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[5]]
    ax.bar(
        range(3), vals, yerr=errs, color=colors, edgecolor="black", linewidth=0.5,
        capsize=2, error_kw={"lw": 0.5},
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("split-half agreement r")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", lw=0.3, color="0.9")
    ax.set_axisbelow(True)
    ax.set_title("C  conditional on perturbation?\n(split by perturbed-gene PC1)", fontsize=6)

    fig.tight_layout(pad=0.6, w_pad=1.2)
    out_dir = asset_images_dir(__file__, "019-simb-multimodal")
    stem = osp.join(out_dir, "residual_covariance_diagnostic")
    fig.savefig(f"{stem}.png", dpi=300)
    savefig_true_size_svg(fig, f"{stem}.svg")
    print(f"\nwrote {stem}.png / .svg")

    out = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "residual_covariance_plots.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")

    print("\nINTERPRETATION")
    print(f"  random split         r = {np.mean(reps):.4f} +/- {np.std(reps):.4f}")
    print(f"  perturbed-gene split r = {cond:.4f}")
    d = (np.mean(reps) - cond) / max(np.std(reps), 1e-9)
    print(f"  difference = {np.mean(reps) - cond:+.4f}  ({d:+.1f} sd of the random split)")
    if abs(d) < 2:
        print("  -> covariance does NOT depend on which gene was perturbed: a single GLOBAL")
        print("     Sigma is the right model, and there is no conditional structure to gain.")
    else:
        print("  -> covariance DOES vary with the perturbation: a conditional Sigma could")
        print("     capture structure a global V V^T cannot.")


if __name__ == "__main__":
    main()
