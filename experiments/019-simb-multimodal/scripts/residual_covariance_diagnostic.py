# experiments/019-simb-multimodal/scripts/residual_covariance_diagnostic.py
# [[experiments.019-simb-multimodal.scripts.residual_covariance_diagnostic]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/residual_covariance_diagnostic
"""Is there reproducible RESIDUAL gene-gene structure worth modelling with a joint head?

GATE for the energy-score arm. The energy score is only worth its cost if the predictive
distribution has dependence to learn -- and the relevant dependence is in the RESIDUALS,
not in the raw response matrix.

    raw:      Cov_s( Y_{s,i}, Y_{s,j} )
    residual: Cov_s( Y_{s,i} - mu_{s,i},  Y_{s,j} - mu_{s,j} )

Raw covariance across strains is dominated by SYSTEMATIC GENOTYPE EFFECTS: two genes
co-vary because different deletions move both of their means. That signal belongs in
mu_theta(s) -- the encoder -- not in Sigma. A joint head fitted against raw covariance
would absorb mean structure the model should be learning instead. So the question is
narrower than "do genes co-vary": it is "do genes co-vary AFTER the mean is removed, and
is that co-variation REPRODUCIBLE rather than sampling noise?"

METHOD -- split-half reproducibility.

  1. Residuals R = Y - muhat, for two baselines of increasing honesty:
       * col_mean -- muhat_{s,f} = mean_s Y_{s,f}. Removes only the per-gene offset; a
         floor, since any real predictor does at least this well.
       * knn -- leave-one-out similarity-weighted average of the k nearest OTHER strains
         in prot_T5 embedding space. A genuine conditional mean, and the closest cheap
         stand-in for what mu_theta(s) will be.
  2. Split strains into halves A / B.
  3. C_A = corr(R_A), C_B = corr(R_B), each [F, F].
  4. r = Pearson( offdiag(C_A), offdiag(C_B) ).

  A high r means the residual correlation pattern replicates on held-out strains -- real
  structure a global Sigma could capture. r near the null means the off-diagonals are
  sampling noise and a joint head has nothing to learn.

NULL CONTROL. With F = 6,169 genes and only ~S/2 strains per half, sample correlation
matrices are noisy and the null is NOT zero. So the same statistic is computed after
independently permuting strains WITHIN each gene, which destroys cross-gene structure while
preserving each gene's marginal. Any r above that is real.

CEILING. The reproducible fraction is also an upper bound on what ANY Sigma model can
capture -- the joint analogue of the per-feature noise ceiling.

RANK. The eigenspectrum of the standardized residuals (via SVD, since a 6169^2
eigendecomposition is not tractable and rank <= n_strains anyway) says how many components
carry the reproducible structure -- i.e. what `k` in Sigma = D + V V^T should be, instead
of guessing 8.

Run from repo root:
    python experiments/019-simb-multimodal/scripts/residual_covariance_diagnostic.py
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

import torch  # noqa: E402

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0
KNN_K = 25
EMBEDDING = "prot_T5_all"
RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128]


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
    np.fill_diagonal(sim, -np.inf)  # leave-one-out
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    w = np.maximum(sim[rows, idx], 0.0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return np.einsum("nk,nkf->nf", w, Y[idx])


def _corr(R: np.ndarray) -> np.ndarray:
    """Gene-gene correlation across strains. R [n, F] -> [F, F]."""
    X = R - R.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = np.inf  # constant genes -> zero correlation, not NaN
    X = X / sd
    return (X.T @ X) / (X.shape[0] - 1)


def _offdiag_agreement(Ra: np.ndarray, Rb: np.ndarray) -> float:
    ca, cb = _corr(Ra), _corr(Rb)
    iu = np.triu_indices(ca.shape[0], k=1)
    a, b = ca[iu], cb[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _shuffle_within_gene(R: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute strains independently per gene: kills cross-gene structure, keeps marginals."""
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
        item.id: torch.cat([t.flatten() for t in item.embeddings.values()])
        .cpu()
        .numpy()
        for item in built[EMBEDDING]
    }
    emb = np.stack([lookup[g] for g in genes])

    mu_col = np.repeat(Y.mean(axis=0, keepdims=True), S, axis=0)
    mu_knn = _knn_loo(emb, Y, KNN_K)
    residuals = {
        "raw (no mean removed)": np.zeros_like(Y),
        "resid: col_mean": mu_col,
        f"resid: knn(prot_T5,k={KNN_K})": mu_knn,
    }

    perm = rng.permutation(S)
    ia, ib = perm[: S // 2], perm[S // 2 :]
    print(f"split halves: {len(ia)} / {len(ib)} strains\n")

    results: dict[str, Any] = {"n_strains": S, "n_genes": F, "knn_k": KNN_K}
    print(f"{'baseline':<32}{'split-half r':>14}{'null r':>10}{'excess':>10}")
    for label, mu in residuals.items():
        R = Y - mu
        obs = _offdiag_agreement(R[ia], R[ib])
        Rn = _shuffle_within_gene(R, rng)
        null = _offdiag_agreement(Rn[ia], Rn[ib])
        print(f"{label:<32}{obs:>14.4f}{null:>10.4f}{obs - null:>+10.4f}")
        results[label] = {"split_half_r": obs, "null_r": null, "excess": obs - null}

    # Rank: eigenspectrum via SVD of standardized residuals (rank <= S, so 6169^2 eigh
    # is neither tractable nor necessary).
    R = Y - mu_knn
    X = R - R.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1)
    sd[sd < 1e-12] = np.inf
    X = X / sd
    sv = np.linalg.svd(X, compute_uv=False)
    ev = sv**2
    cum = np.cumsum(ev) / ev.sum()
    print("\ncumulative variance of knn residual correlation, top-k components:")
    spec = {}
    for k in RANK_GRID:
        if k <= len(cum):
            print(f"   k={k:>4}: {100 * cum[k - 1]:.1f}%")
            spec[str(k)] = float(cum[k - 1])
    results["cumulative_variance_by_rank"] = spec
    eff = float((ev.sum() ** 2) / (ev**2).sum())
    print(f"   participation-ratio effective rank: {eff:.1f}")
    results["effective_rank"] = eff

    out = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "residual_covariance_diagnostic.json",
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
