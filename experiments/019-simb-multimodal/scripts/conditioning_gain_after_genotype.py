# experiments/019-simb-multimodal/scripts/conditioning_gain_after_genotype.py
# [[experiments.019-simb-multimodal.scripts.conditioning_gain_after_genotype]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/conditioning_gain_after_genotype
"""Is the masked-conditioning signal REACHABLE FROM GENOTYPE, or orthogonal to it?

THE PROBLEM WITH THE FIRST ORACLE. `masked_conditioning_oracle.py` measured that observing
m genes predicts the rest extremely well: val pearson_per_feature 0.4084 / 0.6756 / 0.7932
at m = 10 / 100 / 1000, against 0.198 for the trained model. But it took residuals against
each gene's MEAN, so it answers "how much gene-gene structure exists", not the question the
experiment actually turns on.

WHY THAT DISTINCTION DECIDES THE BUILD. Our validation metric is computed with EVERYTHING
masked (m = 0), because at inference we predict a strain's profile from genotype alone. A
masked-training objective can only improve that number if learning to exploit gene-gene
structure teaches the model something it can also USE at m = 0 -- i.e. if the structure is
correlated with what genotype can predict. If instead the structure is orthogonal to
genotype, then masked training is learning a channel that is switched off at evaluation,
and the m = 0 metric cannot benefit no matter how well the objective is optimized. The
capability would still be real (imputation from a partial measurement), but it would be a
different product, not a better score.

THE TEST. Remove a genotype-based conditional mean first, then re-run the conditioning
oracle on what is left:

    R_mu   = Y - mu                      (per-gene mean; what the first oracle used)
    R_knn  = Y - muhat_knn               (leave-one-out similarity-weighted mean over the
                                          k nearest OTHER strains in prot_T5 space)

muhat_knn is a genuine genotype-conditional predictor -- it uses only the perturbed gene's
sequence embedding, exactly the information our model has -- and it is the cheap stand-in
for mu_theta(s) that `residual_covariance_diagnostic.py` already established. Then compare

    gain(m) on R_mu     vs     gain(m) on R_knn .

READING THE RESULT.
  * If gain(m) is essentially UNCHANGED after removing the genotype prediction, the
    gene-gene structure is orthogonal to genotype. Masked training then cannot be expected
    to move the m = 0 metric, and should be justified (if at all) as an imputation
    capability -- with the val protocol changed to report m > 0 explicitly.
  * If gain(m) DROPS substantially, the structure overlaps what genotype reaches, and
    masked training is a plausible route to a better m = 0 score.

There is a prior. `residual_covariance_diagnostic.json` reports split-half reproducibility
of the residual correlation pattern at 0.8706 (col_mean) vs 0.8687 (knn) -- a drop of
0.002. So a genotype-based predictor removes almost none of the gene-gene CORRELATION
STRUCTURE. This script asks whether that also holds for the quantity we care about, the
recoverable per-feature correlation, which is not implied: a correlation pattern can
survive while its exploitable magnitude shrinks.

HONEST LIMIT. kNN in embedding space is weaker than the trained model, so this bounds the
orthogonality question rather than settling it; the exact statement needs the model's own
residuals, which requires a checkpoint inference pass. Recorded as the follow-up.

Usage
-----
    python experiments/019-simb-multimodal/scripts/conditioning_gain_after_genotype.py

Writes ``results/conditioning_gain_after_genotype.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import numpy as np
import torch
from dotenv import load_dotenv
from masked_conditioning_oracle import (  # noqa: E402
    M_GRID,
    N_DRAWS,
    N_TUNE,
    N_VAL,
    RIDGE_GRID,
    SEED,
    _load_expression,
    conditional_mean,
    per_feature_pearson,
)

from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()

DATASET_TAG = "019-simb-multimodal/fig3_core"
EMBEDDING = "prot_T5_all"
KNN_K = 25


def _knn_loo(emb: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out similarity-weighted mean over the k nearest OTHER strains.

    Copied in form from residual_covariance_diagnostic.py so the two agree on what "the
    genotype-based prediction" means. The diagonal is -inf so a strain never predicts
    itself, which would make the residual identically zero.
    """
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    w = np.maximum(sim[rows, idx], 0.0)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    return np.einsum("nk,nkf->nf", w, Y[idx])


def _sweep(
    R_tr: np.ndarray, R_tu: np.ndarray, R_va: np.ndarray, n_gene: int, rng
) -> list[dict[str, object]]:
    """Conditioning gain at each m, ridge tuned on the TUNE split only."""
    out = []
    for m in M_GRID:
        if m == 0:
            out.append({"m": 0, "val_mean": 0.0, "val_sd": 0.0})
            continue
        scores, lams = [], []
        for _ in range(N_DRAWS):
            obs = rng.choice(n_gene, size=m, replace=False)
            mask = np.ones(n_gene, dtype=bool)
            mask[obs] = False
            unobs = np.flatnonzero(mask)
            best_lam, best = RIDGE_GRID[0], -np.inf
            for lam in RIDGE_GRID:
                sc = per_feature_pearson(
                    conditional_mean(R_tr, R_tu, obs, unobs, lam), R_tu[:, unobs]
                )
                if sc > best:
                    best_lam, best = lam, sc
            scores.append(
                per_feature_pearson(
                    conditional_mean(R_tr, R_va, obs, unobs, best_lam), R_va[:, unobs]
                )
            )
            lams.append(best_lam)
        a = np.array(scores)
        out.append(
            {
                "m": int(m),
                "val_mean": float(a.mean()),
                "val_sd": float(a.std(ddof=1)),
                "lam": sorted({float(x) for x in lams}),
            }
        )
    return out


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    rng = np.random.default_rng(SEED)

    genes, Y = _load_expression(base)
    n_strain, n_gene = Y.shape
    print(f"strains={n_strain} genes={n_gene}")

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
    print(f"embedding {EMBEDDING}: {emb.shape}")

    # Same split as masked_conditioning_oracle.py (same SEED, same sizes) so the two
    # baselines are directly comparable rather than merely similar.
    perm = rng.permutation(n_strain)
    val_idx, tune_idx, train_idx = (
        perm[:N_VAL],
        perm[N_VAL : N_VAL + N_TUNE],
        perm[N_VAL + N_TUNE :],
    )

    # muhat_knn is computed over ALL strains leave-one-out: it is a property of the
    # predictor, not of the split, and each strain's prediction already excludes itself.
    mu_knn = _knn_loo(emb, Y, KNN_K)
    mu_col = Y[train_idx].mean(axis=0, keepdims=True)

    baselines = {"col_mean": Y - mu_col, f"knn(prot_T5,k={KNN_K})": Y - mu_knn}

    out: dict[str, object] = {
        "n_strains": int(n_strain),
        "n_genes": int(n_gene),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "seed": SEED,
        "knn_k": KNN_K,
        "embedding": EMBEDDING,
        "baselines": {},
    }

    for name, R in baselines.items():
        res = _sweep(
            R[train_idx], R[tune_idx], R[val_idx], n_gene, np.random.default_rng(SEED)
        )
        out["baselines"][name] = res
        print(f"\n=== residual baseline: {name} ===")
        for r in res:
            print(f"  m={r['m']:>5}  {r['val_mean']:.4f} +/- {r['val_sd']:.4f}")

    # The comparison the script exists for.
    a = {r["m"]: r["val_mean"] for r in out["baselines"]["col_mean"]}
    b = {r["m"]: r["val_mean"] for r in out["baselines"][f"knn(prot_T5,k={KNN_K})"]}
    print("\n=== conditioning gain retained after removing the genotype prediction ===")
    retained = {}
    for m in M_GRID:
        if m == 0 or a[m] <= 0:
            continue
        retained[m] = b[m] / a[m]
        print(
            f"  m={m:>5}  col_mean {a[m]:.4f} -> knn {b[m]:.4f}   "
            f"retained {100 * b[m] / a[m]:.1f}%"
        )
    out["retained_fraction"] = {str(k): float(v) for k, v in retained.items()}

    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    path = osp.join(results_dir, "conditioning_gain_after_genotype.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    print(
        "\nRETAINED NEAR 100% -> the gene-gene structure is ORTHOGONAL to genotype, so a\n"
        "masked objective cannot be expected to move the m=0 metric and must be justified\n"
        "as an imputation capability with m>0 reported explicitly.\n"
        "RETAINED WELL BELOW 100% -> the structure overlaps what genotype reaches, and\n"
        "masked training is a plausible route to a better m=0 score."
    )


if __name__ == "__main__":
    main()
