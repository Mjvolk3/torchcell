# experiments/019-simb-multimodal/scripts/masked_conditioning_oracle.py
# [[experiments.019-simb-multimodal.scripts.masked_conditioning_oracle]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/masked_conditioning_oracle
"""UPPER BOUND on what a masked-label objective can buy, with no training and no GPU.

THE QUESTION. A masked objective hides a subset of measured genes, feeds the rest in as
input, and asks the model to predict the held-out ones. It can only help if observing some
genes actually constrains the others. Before building the objective -- per-batch masking,
an unmasking schedule, cross-gene routing, and a loss restricted to hidden entries -- this
script bounds what is available to be won.

THE BOUND. Write the residual after removing each gene's TRAIN mean as R = Y - mu. Model
it as R ~ N(0, Sigma) with Sigma estimated on TRAIN strains only. Given an observed gene
set M, the minimum-MSE estimate of the unobserved set U is the conditional mean

    E[R_U | R_M] = Sigma_UM Sigma_MM^{-1} R_M .

Evaluating that on HELD-OUT strains and scoring it with the project metric
(`pearson_per_feature`: correlate across strains per gene, then average over genes) gives
an upper bound on any masked-conditioning architecture whose leverage comes from LINEAR
residual structure. A network could in principle beat it with nonlinear structure; nothing
that uses only the linear part can.

WHY IT SHOULD BE NONZERO. `residual_covariance_diagnostic.py` already measured that the
residual gene-gene correlation pattern REPLICATES on held-out strains: split-half
r = 0.8687 against a permutation null of 8.45e-05, concentrated in effective rank 32.78
(the rank-32 subspace carries 59.1% of residual variance). This script turns that
qualitative fact into the quantity we actually care about -- how much of the metric it is
worth -- and calibrates the reveal schedule: identifying ~33 components needs |M| >~ 33,
so m = 10 is under-determined, m = 100 over-determines by ~3x, m = 1000 saturates.

WHAT THIS IS NOT. The residual is taken against the per-gene MEAN, not against the model's
prediction. So this measures the total linear gene-gene structure, of which the trained
model may already capture some through the perturbation. It is therefore an upper bound on
the ABSOLUTE conditioning signal, not on the INCREMENT over the current model. Read it as:
if this is near zero, stop; if it is large, the increment is worth measuring properly.

RIDGE, AND WHY IT IS TUNED ON TRAIN. Sigma_MM is estimated from n_train strains, so at
m = 1000 it is badly conditioned and the plain inverse is meaningless. A ridge
(Sigma_MM + lam I)^{-1} fixes that, but choosing lam on the evaluation split would inflate
the bound. lam is therefore selected on an INNER split of the training strains and then
applied unchanged to the evaluation strains.

Usage
-----
    python experiments/019-simb-multimodal/scripts/masked_conditioning_oracle.py

Writes ``results/masked_conditioning_oracle.json``.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import lmdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DATASET_TAG = "019-simb-multimodal/fig3_core"
SEED = 0
# The reveal schedule from the design note. 0 is the floor (predict each gene's mean, which
# is constant across strains and therefore has zero correlation by construction).
M_GRID = [0, 10, 100, 1000]
# Independent draws of the observed set at each m, so the reported spread is over WHICH
# genes were revealed rather than over a single lucky subset.
N_DRAWS = 5
RIDGE_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
# Matches the model's supervised expression split sizes (1244 / 155).
N_VAL = 155
N_TUNE = 155


def _load_expression(base: str) -> tuple[list[str], np.ndarray]:
    """Return (perturbed gene per strain, Y [S, F]) for single-deletion strains.

    Identical traversal to residual_covariance_diagnostic.py so the two are measuring the
    same matrix -- this script's whole argument rests on that diagnostic's numbers.
    """
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


def per_feature_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean over FEATURES of the across-STRAIN correlation. Matches the project metric.

    A feature with zero predicted variance contributes 0, not NaN: a constant prediction
    has no correlation with anything, and dropping such features would silently score the
    model only on the genes it happened to move.
    """
    p = pred - pred.mean(axis=0, keepdims=True)
    t = true - true.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    den = np.sqrt((p**2).sum(axis=0)) * np.sqrt((t**2).sum(axis=0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return float(r.mean())


def conditional_mean(
    R_fit: np.ndarray, R_eval: np.ndarray, obs: np.ndarray, unobs: np.ndarray, lam: float
) -> np.ndarray:
    """E[R_U | R_M] = Sigma_UM (Sigma_MM + lam I)^{-1} R_M, Sigma from R_fit only.

    The full [F, F] covariance is never formed -- only the two blocks the formula needs,
    which keeps this to tens of MB instead of ~300.
    """
    n = R_fit.shape[0]
    A = R_fit[:, obs].astype(np.float64)
    B = R_fit[:, unobs].astype(np.float64)
    s_mm = (A.T @ A) / (n - 1)
    s_um = (B.T @ A) / (n - 1)
    s_mm.flat[:: s_mm.shape[0] + 1] += lam
    # solve(S_mm, R_M^T) rather than an explicit inverse: same answer, better conditioned.
    w = np.linalg.solve(s_mm, R_eval[:, obs].astype(np.float64).T)
    return (s_um @ w).T


def main() -> None:
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    rng = np.random.default_rng(SEED)

    genes, Y = _load_expression(base)
    n_strain, n_gene = Y.shape
    print(f"strains={n_strain}  genes={n_gene}  (perturbed genes: {len(set(genes))})")

    perm = rng.permutation(n_strain)
    val_idx = perm[:N_VAL]
    tune_idx = perm[N_VAL : N_VAL + N_TUNE]
    train_idx = perm[N_VAL + N_TUNE :]
    print(f"split: train={len(train_idx)} tune={len(tune_idx)} val={len(val_idx)}")

    # Gene means come from TRAIN ONLY. Centering with a mean that saw the evaluation
    # strains would leak, and the leak is exactly the kind that manufactures correlation.
    mu = Y[train_idx].mean(axis=0, keepdims=True)
    R_tr = Y[train_idx] - mu
    R_tu = Y[tune_idx] - mu
    R_va = Y[val_idx] - mu

    out: dict[str, object] = {
        "n_strains": int(n_strain),
        "n_genes": int(n_gene),
        "n_train": int(len(train_idx)),
        "n_tune": int(len(tune_idx)),
        "n_val": int(len(val_idx)),
        "seed": SEED,
        "n_draws": N_DRAWS,
        "ridge_grid": RIDGE_GRID,
        "results": [],
    }

    for m in M_GRID:
        if m == 0:
            # Predicting the per-gene mean is a constant across strains, so every
            # per-feature correlation is identically 0. Recorded explicitly as the floor.
            out["results"].append(
                {"m": 0, "lam": None, "val_mean": 0.0, "val_sd": 0.0, "per_draw": []}
            )
            print(f"m={0:>5}  val pearson_per_feature = 0.0000  (floor, by construction)")
            continue

        per_draw = []
        chosen_lams = []
        for draw in range(N_DRAWS):
            obs = rng.choice(n_gene, size=m, replace=False)
            mask = np.ones(n_gene, dtype=bool)
            mask[obs] = False
            unobs = np.flatnonzero(mask)

            # lam selected on the TUNE strains, never on the evaluation strains.
            best_lam, best_score = RIDGE_GRID[0], -np.inf
            for lam in RIDGE_GRID:
                pred_tu = conditional_mean(R_tr, R_tu, obs, unobs, lam)
                sc = per_feature_pearson(pred_tu, R_tu[:, unobs])
                if sc > best_score:
                    best_lam, best_score = lam, sc

            pred_va = conditional_mean(R_tr, R_va, obs, unobs, best_lam)
            score = per_feature_pearson(pred_va, R_va[:, unobs])
            per_draw.append(score)
            chosen_lams.append(best_lam)

        arr = np.array(per_draw)
        out["results"].append(
            {
                "m": int(m),
                "lam": [float(x) for x in chosen_lams],
                "val_mean": float(arr.mean()),
                "val_sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "per_draw": [float(x) for x in arr],
            }
        )
        print(
            f"m={m:>5}  val pearson_per_feature = {arr.mean():.4f} "
            f"+/- {arr.std(ddof=1):.4f}  (n_draws={N_DRAWS}, lam={sorted(set(chosen_lams))})"
        )

    results_dir = osp.join(
        os.environ["EXPERIMENT_ROOT"], "019-simb-multimodal", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    path = osp.join(results_dir, "masked_conditioning_oracle.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")

    print(
        "\nREAD THIS AS: the value of observing m genes, over and above each gene's mean.\n"
        "Near 0 at every m -> a masked objective has nothing to exploit and should not be\n"
        "built. Rising steeply in m -> the residual structure is worth the build, and the\n"
        "slope between m=10 and m=100 is the number to compare the trained model against."
    )


if __name__ == "__main__":
    main()
