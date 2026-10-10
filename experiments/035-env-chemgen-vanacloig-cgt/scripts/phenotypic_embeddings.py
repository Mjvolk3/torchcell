# experiments/035-env-chemgen-vanacloig-cgt/scripts/phenotypic_embeddings.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.phenotypic_embeddings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/phenotypic_embeddings
"""A compound's PREDICTED chemical-genetic profile in another store, as an embedding.

For each auxiliary source of ``aux_matrices.py`` (every Vanacloig compound already
removed), kernel ridge from the FCFP4 min/max Tanimoto kernel to the source's gene
profile is fit on that source's compounds, and every compound of the embedded library
(5,472 InChIKeys) gets a predicted profile: for a source compound, out of fold (5 folds
over its compounds), for any other compound, from the full fit. The profile is reduced to
its top principal directions, fit on the source's true profiles, so the embedding is a
``pca_dim``-vector per compound. Penalty by leave-one-out on the source compounds.

The result is a structure-only map, inductive for any molecule, that carries what the
other store learned about which structures hit which genes. It is written beside the 031
embeddings so ``baseline_ladder.py`` reads it as one more compound representation:
``$DATA_ROOT/experiments/035-env-chemgen-vanacloig-cgt/embeddings/pheno_<source>.npz``.
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(__file__))
from baseline_ladder import EMBEDDING_DIR, tanimoto  # noqa: E402

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
AUX_DIR = osp.join(DATA_ROOT, "experiments", "035-env-chemgen-vanacloig-cgt", "aux")
OUT = osp.join(DATA_ROOT, "experiments", "035-env-chemgen-vanacloig-cgt", "embeddings")
RESULTS = osp.join(EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results")
LAMBDAS = np.logspace(-3, 2, 11)


def loo_penalty(k: np.ndarray, y: np.ndarray) -> float:
    """The penalty with the lowest leave-one-out squared error, closed form."""
    evals, evecs = np.linalg.eigh(k)
    evals = np.clip(evals, 0, None)
    best, best_err = LAMBDAS[0], np.inf
    yc = evecs.T @ y
    for lam in LAMBDAS:
        shrink = evals / (evals + lam * len(k))
        hat_diag = (evecs**2 * shrink).sum(1)
        fitted = evecs @ (yc * shrink[:, None])
        loo = (y - fitted) / (1 - hat_diag)[:, None]
        err = float(np.nanmean(loo**2))
        if err < best_err:
            best, best_err = lam, err
    return float(best)


def solve(k_fit: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(k_fit + lam * len(k_fit) * np.eye(len(k_fit)), y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["wildenhain", "hillenmeyer_het", "hoepfner_hip"],
    )
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)

    library = np.load(osp.join(EMBEDDING_DIR, "fcfp4_count.npz"), allow_pickle=True)
    keys = list(library["inchikey"])
    row = {k: i for i, k in enumerate(keys)}
    kernel = tanimoto(library["X"].astype(np.float64))
    print(f"library kernel over {len(keys):,} compounds", flush=True)

    report = []
    for source in args.sources:
        a = np.load(osp.join(AUX_DIR, f"{source}.npz"))
        present = [i for i, k in enumerate(a["inchikeys"]) if k in row]
        rows = np.array([row[a["inchikeys"][i]] for i in present])
        y = a["Y"][:, present].T.astype(np.float64)  # compounds x genes
        gene_mean = np.nanmean(y, axis=0)
        y = np.where(np.isnan(y), gene_mean, y) - gene_mean
        # the profile's top directions, from the true profiles
        _, _, vt = np.linalg.svd(y, full_matrices=False)
        w = vt[: args.pca_dim].T  # genes x dim
        scores = y @ w
        k_src = kernel[np.ix_(rows, rows)]
        lam = loo_penalty(k_src, scores)

        embedding = np.zeros((len(keys), args.pca_dim))
        alpha = solve(k_src, scores, lam)
        embedding[:] = kernel[:, rows] @ alpha
        # source compounds: out of fold
        order = rng.permutation(len(rows))
        oof = np.zeros_like(scores)
        for fold in np.array_split(order, args.n_folds):
            fit = np.setdiff1d(order, fold)
            alpha_f = solve(k_src[np.ix_(fit, fit)], scores[fit], lam)
            oof[fold] = k_src[np.ix_(fold, fit)] @ alpha_f
        embedding[rows] = oof
        held = [
            float(np.corrcoef(oof[:, j], scores[:, j])[0, 1]) for j in range(w.shape[1])
        ]
        np.savez_compressed(
            osp.join(OUT, f"pheno_{source}.npz"),
            inchikey=np.array(keys),
            X=embedding.astype(np.float32),
        )
        report.append(
            {
                "source": source,
                "compounds_fit": len(rows),
                "genes": y.shape[1],
                "pca_dim": args.pca_dim,
                "penalty": lam,
                "oof_pearson_first_direction": held[0],
                "oof_pearson_median_direction": float(np.median(held)),
            }
        )
        print(report[-1], flush=True)
    pd.DataFrame(report).to_csv(
        osp.join(RESULTS, "phenotypic_embeddings.csv"), index=False
    )


if __name__ == "__main__":
    main()
