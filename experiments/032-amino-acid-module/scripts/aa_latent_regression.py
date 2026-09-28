# experiments/032-amino-acid-module/scripts/aa_latent_regression.py
# [[experiments.032-amino-acid-module.scripts.aa_latent_regression]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/aa_latent_regression

r"""Does predicting an autoencoder's latent beat predicting the amino acids directly?

``aa_profile_reconstruction.py`` (2026-09-27) found the M\"ulleder profile compressible
only slowly (four dimensions hold 0.58 to 0.62 of a held-out strain) and betaxanthin in
its minor directions. This script asks the question that decides whether an
autoencoder helps a model PREDICT the profile of a strain nobody measured: fit the
latent on the training strains' profiles, regress the latent from the gene's
representation, decode, and compare with regressing the nineteen amino acids directly.
If the profile's shared structure is what the genotype can reach, the latent route
wins; if not, it loses the per-amino-acid detail and ties or falls behind.

Representations of the deleted gene: the GO multi-hot (``SM_gem_go``, the ladder's
GO), and the propagated-label store of the profile (``SM_aa_graphlabel``, one hop over
the nine graphs, leave-one-out, no pinned label read). Models: ridge (alpha chosen on a
validation fifth of the training genes) and 25 cosine nearest neighbors. Bottlenecks:
PCA with k components and an autoencoder (19 -> 64 -> k -> 64 -> 19, sklearn MLP) whose
encoder gives the latent and whose decoder maps a predicted latent back. Score: pinned
held-out mean Spearman over the nineteen amino acids, the 032 study's statistic.
Writes ``results/aa_latent_regression.json``. Run from the repo root::

    PYTHONHASHSEED=0 ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/aa_latent_regression.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
SPLIT = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_nested_split.json"
)
KS = [2, 4, 8, 12]
ALPHAS = np.logspace(-2, 4, 13)


def load(name: str) -> dict[str, np.ndarray]:
    z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{name}.npz"), allow_pickle=False)
    return {str(g): v for g, v in zip(z["genes"], z["v_deletion"].astype(np.float64), strict=True)}


def mean_spearman(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean([spearmanr(pred[:, j], true[:, j])[0] for j in range(true.shape[1])]))


def main() -> None:
    started = time.time()
    pool = load("YPD_bx_aapool")
    reps = {"go": load("SM_gem_go"), "graph_label": load("SM_aa_graphlabel")}
    with open(SPLIT) as fh:
        pinned = set(json.load(fh)["split"]["test"])
    genes = sorted(g for g in pool if all(g in r for r in reps.values()))
    test = np.array([g in pinned for g in genes])
    rng = np.random.default_rng(0)
    train_idx = np.flatnonzero(~test)
    val_idx = rng.choice(train_idx, size=len(train_idx) // 5, replace=False)
    fit_idx = np.setdiff1d(train_idx, val_idx)
    y = np.log1p(np.stack([pool[g] for g in genes]))
    mu, sd = y[fit_idx].mean(axis=0), y[fit_idx].std(axis=0)
    y = (y - mu) / sd
    print(
        f"  {len(genes)} strains with a profile and every representation; fit "
        f"{len(fit_idx)}, validation {len(val_idx)}, pinned {int(test.sum())}",
        flush=True,
    )

    # Bottlenecks fitted on the fitting strains' profiles only.
    codecs: dict[str, tuple] = {"direct": (lambda v: v, lambda v: v)}
    for k in KS:
        pca = PCA(n_components=k).fit(y[fit_idx])
        codecs[f"pca{k}"] = (pca.transform, pca.inverse_transform)
        ae = MLPRegressor(
            hidden_layer_sizes=(64, k, 64),
            activation="relu",
            max_iter=2000,
            early_stopping=True,
            random_state=0,
        ).fit(y[fit_idx], y[fit_idx])
        w, b = ae.coefs_, ae.intercepts_

        def enc(v: np.ndarray, w=w, b=b) -> np.ndarray:
            h = np.maximum(v @ w[0] + b[0], 0)
            return np.maximum(h @ w[1] + b[1], 0)

        def dec(zz: np.ndarray, w=w, b=b) -> np.ndarray:
            h = np.maximum(zz @ w[2] + b[2], 0)
            return h @ w[3] + b[3]

        codecs[f"ae{k}"] = (enc, dec)

    rows = []
    for rep_name, rep in reps.items():
        x = np.stack([rep[g] for g in genes])
        if rep_name == "graph_label":
            x = np.log1p(x)
        xm, xs = x[fit_idx].mean(axis=0), x[fit_idx].std(axis=0)
        xs[xs == 0] = 1.0
        x = (x - xm) / xs
        for codec_name, (enc, dec) in codecs.items():
            target = enc(y)
            for model in ("ridge", "knn"):
                if model == "ridge":
                    best = max(
                        ALPHAS,
                        key=lambda a: mean_spearman(
                            dec(Ridge(alpha=a).fit(x[fit_idx], target[fit_idx]).predict(x[val_idx])),
                            y[val_idx],
                        ),
                    )
                    m = Ridge(alpha=best).fit(x[train_idx], target[train_idx])
                else:
                    best = 25
                    m = KNeighborsRegressor(n_neighbors=25, metric="cosine").fit(
                        x[train_idx], target[train_idx]
                    )
                pred = dec(m.predict(x[test]))
                score = mean_spearman(pred, y[test])
                rows.append(
                    {
                        "representation": rep_name,
                        "codec": codec_name,
                        "model": model,
                        "param": float(best),
                        "pinned_mean_spearman": score,
                    }
                )
                print(
                    f"  {rep_name:12s} {codec_name:7s} {model:5s} pinned {score:+.4f}",
                    flush=True,
                )
    out = {
        "generated_by": "experiments/032-amino-acid-module/scripts/aa_latent_regression.py",
        "split": SPLIT,
        "n_strains": len(genes),
        "n_pinned": int(test.sum()),
        "statistic": (
            "pinned held-out mean Spearman over the 19 amino acids of the decoded "
            "prediction; 'direct' regresses the standardized log1p profile itself"
        ),
        "rows": rows,
        "seconds": time.time() - started,
    }
    path = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results", "aa_latent_regression.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
