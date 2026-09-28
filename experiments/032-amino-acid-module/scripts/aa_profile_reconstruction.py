# experiments/032-amino-acid-module/scripts/aa_profile_reconstruction.py
# [[experiments.032-amino-acid-module.scripts.aa_profile_reconstruction]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/032-amino-acid-module/scripts/aa_profile_reconstruction

r"""How compressible is the Mulleder amino-acid profile, and does betaxanthin survive the squeeze.

Before training a model that PREDICTS the profile (a genotype encoder whose latent must
decode the nineteen amino acids and betaxanthin both), two properties of the profile set
what such a latent can hold: how many dimensions reconstruct a held-out strain's profile,
and how much of the profile's betaxanthin read survives when the profile is forced
through that bottleneck. This script measures both on the betaxanthin study's pinned
split, genes pinned to test held out of every fit.

- Reconstruction: the profile as $\log(1 + \text{mM})$, standardized on the training
  genes, through PCA with $k$ components and through an autoencoder
  (19 -> 64 -> k -> 64 -> 19, ReLU, sklearn MLP, five seeds), scored on the held-out
  genes as the fraction of variance reconstructed, pooled and per amino acid.
- Betaxanthin through the bottleneck: the Cachera betaxanthin score of the strain
  (the ``SM_gem_bxscore`` store, the loader's corrected 24 h colony fluorescence) read
  from the $k$ PCA scores by ridge and by 25 cosine nearest neighbors, pinned-test
  Spearman, against the same models on all nineteen columns.

This is a diagnostic of the profile, not of any genotype model: every row here is the
strain's own measurement. Writes ``results/aa_profile_reconstruction.json``. Run from
the repo root::

    PYTHONHASHSEED=0 ~/miniconda3/envs/torchcell/bin/python \
        experiments/032-amino-acid-module/scripts/aa_profile_reconstruction.py
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
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
SPLIT = osp.join(
    EXPERIMENT_ROOT, "020-cachera-betaxanthin", "results", "merzbacher_nested_split.json"
)
KS = [1, 2, 3, 4, 6, 8, 12, 19]
AE_SEEDS = [0, 1, 2, 3, 4]


def load(name: str) -> tuple[list[str], np.ndarray, list[str]]:
    z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{name}.npz"), allow_pickle=False)
    return [str(g) for g in z["genes"]], z["v_deletion"].astype(np.float64), [
        str(r) for r in z["rxn_ids"]
    ]


def r2(pred: np.ndarray, true: np.ndarray) -> tuple[float, list[float]]:
    """Fraction of variance reconstructed, pooled over columns and per column."""
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return float(1 - ss_res.sum() / ss_tot.sum()), [float(x) for x in 1 - ss_res / ss_tot]


def main() -> None:
    started = time.time()
    genes, pool, aa = load("YPD_bx_aapool")
    bx_genes, bx, _ = load("SM_gem_bxscore")
    with open(SPLIT) as fh:
        pinned = set(json.load(fh)["split"]["test"])
    bx_of = {g: float(v[0]) for g, v in zip(bx_genes, bx, strict=True)}
    x = np.log1p(pool)
    test = np.array([g in pinned for g in genes])
    mu, sd = x[~test].mean(axis=0), x[~test].std(axis=0)
    z = (x - mu) / sd
    ztr, zte = z[~test], z[test]
    has_bx = np.array([g in bx_of for g in genes])
    y = np.array([bx_of.get(g, np.nan) for g in genes])
    tr_b, te_b = (~test) & has_bx, test & has_bx
    print(
        f"  pool: {len(genes)} strains, {len(aa)} amino acids; held out {int(test.sum())} "
        f"pinned; betaxanthin score on {int(tr_b.sum())} train / {int(te_b.sum())} pinned",
        flush=True,
    )

    def bx_reads(feat: np.ndarray) -> dict[str, float]:
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(feat[tr_b], y[tr_b])
        knn = KNeighborsRegressor(n_neighbors=25, metric="cosine").fit(
            feat[tr_b], y[tr_b]
        )
        return {
            "ridge": float(spearmanr(ridge.predict(feat[te_b]), y[te_b])[0]),
            "knn": float(spearmanr(knn.predict(feat[te_b]), y[te_b])[0]),
        }

    full = bx_reads(z)
    print(f"  betaxanthin from all 19 columns: ridge {full['ridge']:+.3f} knn {full['knn']:+.3f}")
    rows = []
    for k in KS:
        pca = PCA(n_components=k).fit(ztr)
        rec = pca.inverse_transform(pca.transform(zte))
        pca_r2, pca_per = r2(rec, zte)
        ae_r2 = []
        for s in AE_SEEDS:
            if k == len(aa):
                break
            ae = MLPRegressor(
                hidden_layer_sizes=(64, k, 64),
                activation="relu",
                max_iter=2000,
                early_stopping=True,
                random_state=s,
            ).fit(ztr, ztr)
            ae_r2.append(r2(ae.predict(zte), zte)[0])
        scores = pca.transform(z)
        reads = bx_reads(scores)
        row = {
            "k": k,
            "pca_explained_train": float(pca.explained_variance_ratio_.sum()),
            "pca_heldout_r2": pca_r2,
            "pca_heldout_r2_per_aa": dict(zip(aa, pca_per, strict=True)),
            "ae_heldout_r2_mean": float(np.mean(ae_r2)) if ae_r2 else None,
            "ae_heldout_r2_sd": float(np.std(ae_r2, ddof=1)) if len(ae_r2) > 1 else None,
            "bx_from_scores": reads,
        }
        rows.append(row)
        ae_txt = (
            f"AE {row['ae_heldout_r2_mean']:.3f} (sd {row['ae_heldout_r2_sd']:.3f})"
            if ae_r2
            else "AE --"
        )
        print(
            f"  k={k:2d}: PCA held-out R2 {pca_r2:.3f} (train {row['pca_explained_train']:.3f}), "
            f"{ae_txt}; betaxanthin ridge {reads['ridge']:+.3f} knn {reads['knn']:+.3f}",
            flush=True,
        )
    out = {
        "generated_by": "experiments/032-amino-acid-module/scripts/aa_profile_reconstruction.py",
        "store": "YPD_bx_aapool",
        "betaxanthin_store": "SM_gem_bxscore",
        "split": SPLIT,
        "n_strains": len(genes),
        "n_heldout": int(test.sum()),
        "n_bx_train": int(tr_b.sum()),
        "n_bx_heldout": int(te_b.sum()),
        "amino_acids": aa,
        "betaxanthin_from_all_columns": full,
        "rows": rows,
        "statistic": (
            "held-out fraction of variance of the standardized log1p profile reconstructed "
            "through a k-dimensional PCA or autoencoder bottleneck; pinned-test Spearman "
            "of the betaxanthin score read from the k PCA scores"
        ),
        "seconds": time.time() - started,
    }
    path = osp.join(EXPERIMENT_ROOT, "032-amino-acid-module", "results", "aa_profile_reconstruction.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)


if __name__ == "__main__":
    main()
