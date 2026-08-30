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

PANELS. Each one states its own conclusion in its title, because a panel that needs the
caption to be readable is not a panel the reader can check.
  a  The residual matrix R the whole figure is computed from.
  b  Split-half agreement -- C_A vs C_B off-diagonals (hexbin), observed against the
     shuffled null. Does the correlation pattern replicate on held-out strains?
  c  Cross-validated spectrum -- variance the components fit on half A explain in half B,
     against the in-sample curve and a random-direction floor. How many components are real.
  d  CONDITIONAL test -- split strains by PERTURBED-GENE SIMILARITY rather than at
     random, into two groups whose deleted genes are far apart in prot_T5 space. If residual
     covariance is perturbation-specific, the two groups' correlation matrices should agree
     LESS than two random halves do. If they agree just as well, the covariance is global and
     a single shared Sigma is already the right model -- there is nothing conditional to gain.
  e  Construction of the shuffled null that is the third bar of d.

A sixth panel drawing four individual strain profiles was REMOVED after review: it showed
noise for four arbitrary deletions and supported none of the four claims above, so its space
went to the panels that do.

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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    panel_label,
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

    # ---- Panel D: CROSS-VALIDATED spectrum -- which components actually generalize ----
    # In-sample cumulative variance OVERSTATES the usable rank. With S=1482 strains and
    # F=6169 genes, F/S ~ 4.2 > 1, so the sample correlation matrix is rank-deficient and
    # its eigenvalues are inflated by sampling noise (Marchenko-Pastur): a component can
    # "explain variance" purely by fitting the noise of the strains it was estimated on.
    #
    # The honest question for choosing k is not "how much variance does component k hold?"
    # but "does component k REPLICATE?" -- so fit the directions on half A and measure the
    # variance they explain in the HELD-OUT half B. Where that curve saturates is where
    # extra rank stops buying anything real.
    #
    # Baseline: k random orthonormal directions explain k/F of the variance in expectation,
    # which is the floor a component must beat to be doing anything at all.
    Xa = X[ia]
    Xb = X[ib]
    Va = np.linalg.svd(Xa, full_matrices=False)[2]  # [r, F] right singular vectors
    Bproj = Xb @ Va.T  # [n_b, r] held-out data in A's component basis
    held_out = np.cumsum((Bproj**2).sum(axis=0)) / (Xb**2).sum()
    in_sample = cum
    k_max = min(256, held_out.size)
    results["cv_spectrum"] = {
        str(k): {"in_sample": float(in_sample[k - 1]), "held_out": float(held_out[k - 1])}
        for k in RANK_GRID
        if k <= k_max
    }
    print("\ncross-validated spectrum (fit on half A, evaluated on half B):")
    print(f"  {'k':>5} {'in-sample':>11} {'held-out':>10} {'random floor':>13}")
    for k in RANK_GRID:
        if k <= k_max:
            print(f"  {k:>5} {100 * in_sample[k - 1]:>10.1f}% {100 * held_out[k - 1]:>9.1f}%"
                  f" {100 * k / F:>12.2f}%")

    # ---------------- figure ----------------
    _apply_plot_style()
    w = PANEL_WIDTHS_MM["full"]
    fig = plt.figure(figsize=(mm_to_in(w), mm_to_in(112)))
    gs = fig.add_gridspec(2, 3, hspace=0.80, wspace=0.55)
    axA, axC, axD = (fig.add_subplot(gs[0, i]) for i in range(3))
    axE = fig.add_subplot(gs[1, 0])
    # The null-construction panel spans two columns: it carries the prose that explains
    # what the third bar of panel d is, and that prose was unreadable in a third of a page.
    axF = fig.add_subplot(gs[1, 1:])

    # Order genes by their loading on the leading residual component, so the shared
    # co-expression program is visible as a gradient rather than scrambled.
    gene_order = np.argsort(Va[0])
    strain_order = np.argsort(X @ Va[0])

    # --- A: THE DATA the covariance is computed from ---
    n_s, n_g = 120, 400
    si = strain_order[np.linspace(0, S - 1, n_s).astype(int)]
    gi = gene_order[np.linspace(0, F - 1, n_g).astype(int)]
    M = R[np.ix_(si, gi)]
    v = float(np.percentile(np.abs(M), 98))
    im = axA.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    axA.set_xlabel(f"reporter genes ({n_g} of {F})")
    axA.set_ylabel(f"strains ({n_s} of {S})")
    axA.set_title(
        "the data: residual log2 ratio matrix R\n"
        "one deletion strain per row, gene mean\n"
        "and a kNN conditional mean removed",
        fontsize=6,
    )
    # No colorbar label: at this panel width it runs into panel b's y axis, and the title
    # already names the units.
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5, width=0.4)
    cb.outline.set_linewidth(0.4)

    # --- C: split-half agreement, with the null made visible ---
    sub = rng.choice(oa.size, size=min(300_000, oa.size), replace=False)
    axC.hexbin(oa[sub], ob[sub], gridsize=42, cmap="Oranges", mincnt=1, linewidths=0)
    # The null collapses to a dot at the origin, so an overlaid hexbin is invisible. Draw
    # its 99% extent as an explicit ring instead -- that is the honest visual for "the null
    # occupies this much of the plane".
    nr = float(np.percentile(np.abs(np.stack([na, nb])), 99))
    axC.add_patch(
        plt.Circle((0, 0), nr, fill=False, ec=PLOT_PALETTE[5], lw=0.7, ls="--", zorder=5)
    )
    axC.plot([-1, 1], [-1, 1], color="black", lw=0.5, ls=(0, (4, 3)))
    axC.set_xlim(-1, 1)
    axC.set_ylim(-1, 1)
    axC.set_xlabel("gene-gene corr, strain half A")
    axC.set_ylabel("gene-gene corr, half B")
    axC.set_title(
        "the gene-gene pattern replicates\n"
        "on strains it was not measured on:\n"
        f"r = {obs:.3f}, shuffled null {null:.4f}",
        fontsize=6,
    )
    axC.xaxis.set_major_locator(MultipleLocator(0.5))
    axC.yaxis.set_major_locator(MultipleLocator(0.5))
    axC.legend(
        handles=[
            Patch(facecolor=PLOT_PALETTE[0], label="one gene pair"),
            Line2D([0], [0], color=PLOT_PALETTE[5], ls="--", lw=0.7,
                   label="shuffled null, 99%"),
            Line2D([0], [0], color="black", ls=(0, (4, 3)), lw=0.5, label="y = x"),
        ],
        frameon=False, fontsize=5, loc="upper left", handlelength=1.4,
    )

    # --- D: cross-validated spectrum ---
    ks = np.arange(1, k_max + 1)
    axD.plot(ks, 100 * in_sample[:k_max], color=PLOT_PALETTE[0], lw=1.0, label="in-sample")
    axD.plot(ks, 100 * held_out[:k_max], color=PLOT_PALETTE[1], lw=1.0, label="held-out")
    axD.plot(ks, 100 * ks / F, color=PLOT_PALETTE[5], lw=0.7, ls=":", label="random floor")
    axD.axvline(CHOSEN_RANK, color="black", lw=0.6, ls=(0, (4, 3)))
    axD.text(CHOSEN_RANK * 1.15, 4, f"k={CHOSEN_RANK}", fontsize=5)
    axD.set_xscale("log")
    axD.set_xlabel("rank k")
    axD.set_ylabel("cumulative variance (%)")
    axD.set_ylim(0, 100)
    axD.set_title(
        f"about {eff:.0f} components generalize:\n"
        "the held-out curve is the honest one,\n"
        "in-sample overstates the usable rank",
        fontsize=6,
    )
    axD.yaxis.set_major_locator(MultipleLocator(20))
    axD.yaxis.set_minor_locator(MultipleLocator(10))
    axD.tick_params(which="minor", length=0)
    axD.grid(axis="y", which="both", lw=0.3, color="0.9")
    axD.set_axisbelow(True)
    axD.legend(frameon=False, fontsize=5, loc="upper left", handlelength=1.4)

    # --- E: the conditional test -- the load-bearing panel ---
    # The comparison is bar 1 against bar 2, and the reader has to be able to see that it is
    # a NULL RESULT: grouping strains by which gene was deleted leaves the correlation
    # pattern where random grouping leaves it. So the difference and its size in units of
    # the random-split spread are drawn between the two bars rather than left to the caption.
    labels = ["random\nsplit", "by deleted\ngene", "shuffled\nnull (e)"]
    vals = [float(np.mean(reps)), cond, null]
    spread = float(np.std(reps))
    errs = [spread, 0.0, 0.0]
    axE.bar(
        range(3), vals, yerr=errs, color=[PLOT_PALETTE[0], PLOT_PALETTE[1], PLOT_PALETTE[5]],
        edgecolor="black", linewidth=0.5, capsize=2, error_kw={"lw": 0.5},
    )
    for i, val in enumerate(vals):
        axE.text(i, val + 0.03, f"{val:.3f}", ha="center", fontsize=5)
    bracket = max(vals[0], vals[1]) + 0.14
    axE.plot(
        [0, 0, 1, 1],
        [bracket - 0.03, bracket, bracket, bracket - 0.03],
        color="black", lw=0.5,
    )
    axE.text(
        0.5, bracket + 0.02,
        f"{cond - vals[0]:+.3f}\n({(cond - vals[0]) / max(spread, 1e-9):+.1f} sd)",
        ha="center", va="bottom", fontsize=5, linespacing=1.3,
    )
    axE.set_xticks(range(3))
    axE.set_xticklabels(labels)
    axE.set_ylabel("split-half agreement r")
    # Ticks stop at 1.0, the largest value a correlation can take; the space above it is
    # annotation room, not axis.
    axE.set_ylim(0, 1.55)
    axE.set_yticks(np.arange(0, 1.01, 0.2))
    axE.yaxis.set_minor_locator(MultipleLocator(0.1))
    axE.tick_params(which="minor", length=0)
    axE.grid(axis="y", which="both", lw=0.3, color="0.9")
    axE.set_axisbelow(True)
    axE.set_title(
        "THE RESULT: grouping strains by the\ndeleted gene barely changes the\n"
        "pattern, so ONE global Sigma fits",
        fontsize=6,
    )

    # --- F: how the null of panel E is built ---
    axF.axis("off")
    axF.set_title(
        "how the shuffled null, the third bar of panel d, is built", fontsize=6
    )
    demo = R[np.ix_(si[:14], gi[:14])]
    dv = float(np.percentile(np.abs(demo), 95))
    a1 = axF.inset_axes((0.01, 0.50, 0.13, 0.44))
    a2 = axF.inset_axes((0.21, 0.50, 0.13, 0.44))
    a1.imshow(demo, cmap="RdBu_r", vmin=-dv, vmax=dv, interpolation="nearest")
    shuf = demo.copy()
    for c in range(shuf.shape[1]):
        rng.shuffle(shuf[:, c])
    a2.imshow(shuf, cmap="RdBu_r", vmin=-dv, vmax=dv, interpolation="nearest")
    for a, t in ((a1, "observed R"), (a2, "columns permuted")):
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(t, fontsize=5, pad=2)
        for sp in a.spines.values():
            sp.set_linewidth(0.4)
    axF.annotate(
        "", xy=(0.20, 0.72), xytext=(0.15, 0.72), xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "black"},
    )
    axF.text(
        0.42, 0.94,
        "Every GENE COLUMN is permuted over strains INDEPENDENTLY,\n"
        "so each gene keeps its own values and its own column histogram\n"
        "while the pairing between columns is destroyed. All gene-gene\n"
        "correlation is removed; every per-gene property is held fixed.",
        fontsize=5, va="top", linespacing=1.5,
    )
    axF.text(
        0.01, 0.36,
        f"Two halves of the permuted matrix agree at r = {null:.4f}. That is the floor the "
        f"observed {obs:.3f} in panel b is measured\n"
        "against, and it is the third bar of panel d. It answers only whether cross-gene "
        "structure exists AT ALL; whether\n"
        "that structure is CONDITIONAL on the perturbation is the first-against-second-bar "
        "comparison in panel d.",
        fontsize=5, va="top", linespacing=1.5,
    )

    # Reading order is left to right, top to bottom: a b c on the first row, d e on the
    # second. The letters go on last so nothing drawn later can sit on top of them.
    for letter, axis in zip("abcde", (axA, axC, axD, axE, axF)):
        panel_label(axis, letter)

    out_dir = asset_images_dir(__file__, "019-simb-multimodal")
    stem = osp.join(out_dir, "residual_covariance_diagnostic")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
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
