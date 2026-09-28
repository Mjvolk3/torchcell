# experiments/031-env-chemgen-inhibitor-tolerance/scripts/baseline_ceilings.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.baseline_ceilings]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/baseline_ceilings
"""How close do linear models and k-nearest neighbors get to the reliability ceiling?

The reliability work put a ceiling on any model's correlation with the Vanacloig response.
This script measures what simple models actually reach against that ceiling, so the gap
between "what the data allows" and "what a feature set delivers" is a measured number
rather than an assumption.

THE TASK. The Vanacloig screen is a gene-by-compound matrix. Two cold-start splits matter
and they ask different questions.

* ``compound_cold`` holds out one whole compound at a time, all 41 folds. Predicting it
  requires saying something about a molecule never dosed, so ONLY a molecular feature can
  help. This is the split that matters for hydrolysate generalization.
* ``gene_cold`` holds out a fifth of the genes at a time. Predicting those requires a gene
  feature, so this is where a strain representation earns its place.

TWO TARGETS, AND THE SECOND IS THE HONEST ONE. On the raw response a model scores well by
learning only which genes are sick in general, because the gene main effect dominates the
matrix. Subtracting each gene's mean over the TRAINING compounds removes that main effect
and leaves the compound-specific signal, which is the thing a tolerance model has to get
right. Both are reported. Centering is refit inside every fold, so no held-out value ever
informs the centering.

THE CEILING IS RECOMPUTED PER TARGET. Centering removes signal and keeps noise, so the raw
ceiling does not apply to the centered target. For each compound and each target the
reliability is recomputed as ``1 - mean(SE^2) / Var(target)`` over genes, and the ceiling on
correlation with the noise-free response is its square root. A model scored on the centered
target is compared with the centered ceiling.

BASELINES THAT CANNOT USE THE FEATURES. ``gene_mean`` predicts each gene's training-compound
mean, identical for every held-out compound, so it carries no molecular information at all.
Any encoder that does not beat it has contributed nothing. On the centered target its
prediction is zero by construction, which makes the comparison a clean test of whether the
molecule features carry compound-specific signal.

Writes ``results/baseline_ceilings_<split>.csv`` (one row per fold x feature set x model x
target) and ``results/baseline_ceilings_summary.csv`` (medians over folds).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
from glob import glob
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV

from torchcell.molecule.similarity import cosine_matrix, tanimoto_matrix

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMB_DIR = osp.join(RESULTS_DIR, "embeddings")
GENE_EMB_ROOT = osp.join(DATA_ROOT, "data", "scerevisiae")

# the drug-sensitized host: every Vanacloig genotype carries these three on top of the
# queried deletion, so they are not the queried gene
HOST_GENES = {"YBL005W", "YDR011W", "YGL013C"}
FINGERPRINTS = {"ecfp4_count", "ecfp4_bit", "fcfp4_count", "maccs"}
ALPHAS = np.logspace(-1, 6, 22)
KNN_KS = (1, 3, 5)
N_RANDOM_DRAWS = 25

GENE_EMBEDDINGS = {
    "codon_freq": ("codon_frequency_embedding", "cds_codon_frequency"),
    "calm": ("calm_embedding", "calm"),
    "prott5": ("protT5_embedding", "prot_t5_xl_uniref50_all"),
    "esm2": ("esm2_embedding", "esm2_t33_650M_UR50D_all"),
}


# --------------------------------------------------------------------------- data
def queried_gene(genotype: str) -> str:
    """The one deletion of a Vanacloig genotype that is not part of the host."""
    genes = [g for g in genotype.split("|") if g not in HOST_GENES]
    return genes[0] if len(genes) == 1 else ""


def load_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gene-by-compound response, its standard error, and the compound identity table."""
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, "records_vanacloig2022.parquet"),
        columns=["gene", "compound", "inchikey", "response", "response_se"],
    )
    df["qgene"] = df["gene"].map(queried_gene)
    df = df[df["qgene"] != ""]
    resp = df.pivot_table(
        index="qgene", columns="compound", values="response", aggfunc="mean"
    )
    se = df.pivot_table(
        index="qgene", columns="compound", values="response_se", aggfunc="mean"
    )
    ident = (
        df[["compound", "inchikey"]]
        .drop_duplicates("compound")
        .set_index("compound")
        .loc[resp.columns]
    )
    return resp, se.loc[resp.index, resp.columns], ident


def load_molecule_embeddings(ident: pd.DataFrame) -> dict[str, NDArray[np.float64]]:
    """Per encoder, the compound-by-feature matrix aligned to the response columns."""
    out: dict[str, NDArray[np.float64]] = {}
    for path in sorted(glob(osp.join(EMB_DIR, "*.npz"))):
        enc = osp.basename(path)[:-4]
        z = np.load(path, allow_pickle=True)
        idx = {str(k): i for i, k in enumerate(z["inchikey"])}
        X = np.asarray(z["X"], dtype=np.float64)
        keys = ident["inchikey"].to_numpy()
        if not all(k in idx for k in keys):
            continue  # an encoder that cannot represent every Vanacloig compound
        A = X[[idx[k] for k in keys]]
        # the RDKit descriptor block has undefined entries for some structures; impute by
        # the column median over these compounds so the feature set stays usable
        if np.isnan(A).any():
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(A, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            A = np.where(np.isnan(A), med, A)
        out[enc] = A
    return out


def load_gene_embeddings(genes: pd.Index) -> dict[str, NDArray[np.float64]]:
    """Per gene encoder, the gene-by-feature matrix aligned to the response rows."""
    out: dict[str, NDArray[np.float64]] = {}
    for name, (subdir, key) in GENE_EMBEDDINGS.items():
        path = osp.join(GENE_EMB_ROOT, subdir, "processed", f"{key}.pt")
        if not osp.exists(path):
            continue
        data, _ = torch.load(path, map_location="cpu", weights_only=False)
        ids = [str(g) for g in data["id"]]
        mat = data["embeddings"][key]
        arr = np.asarray(mat.reshape(len(ids), -1), dtype=np.float64)
        idx = {g: i for i, g in enumerate(ids)}
        keep = [g for g in genes if g in idx]
        if len(keep) < 0.9 * len(genes):
            continue
        full = np.full((len(genes), arr.shape[1]), np.nan)
        for i, g in enumerate(genes):
            if g in idx:
                full[i] = arr[idx[g]]
        col_med = np.nanmedian(full, axis=0)
        out[name] = np.where(np.isnan(full), col_med, full)
    return out


# ----------------------------------------------------------------------- ceilings
def ceiling(target: NDArray[np.float64], se: NDArray[np.float64]) -> float:
    """Sqrt of 1 - mean(SE^2)/Var(target): the limit on r against the noise-free value."""
    ok = np.isfinite(target) & np.isfinite(se)
    if ok.sum() < 10:
        return float("nan")
    var = float(np.var(target[ok], ddof=1))
    rel = 1.0 - float(np.mean(se[ok] ** 2)) / var if var > 0 else float("nan")
    return float(np.sqrt(rel)) if rel > 0 else 0.0


# ------------------------------------------------------------------------- models
def standardize(
    train: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def similarity(
    enc: str, X: NDArray[np.float64], ia: NDArray[np.int_], ib: NDArray[np.int_]
) -> NDArray[np.float64]:
    if enc in FINGERPRINTS:
        return tanimoto_matrix(X[ia], X[ib])
    return cosine_matrix(X[ia], X[ib])


def knn_predict(
    sims: NDArray[np.float64], train_profiles: NDArray[np.float64], k: int
) -> NDArray[np.float64]:
    """Similarity-weighted mean of the k most similar training profiles."""
    order = np.argsort(-sims)[:k]
    w = np.clip(sims[order], 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    block = train_profiles[order]
    masked = np.ma.masked_invalid(block)
    return np.ma.average(masked, axis=0, weights=w).filled(np.nan)


def ridge_predict(
    Xtr: NDArray[np.float64], Ytr: NDArray[np.float64], Xte: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Multi-output ridge with the penalty chosen by efficient leave-one-out on train."""
    # a gene unobserved in every training column has no column mean; it falls back to the
    # overall mean, which makes it uninformative rather than dropping the whole fit
    with np.errstate(invalid="ignore"):
        col_mean = np.nanmean(Ytr, axis=0)
    overall = float(np.nanmean(Ytr))
    col_mean = np.where(np.isfinite(col_mean), col_mean, overall)
    Y = np.where(np.isnan(Ytr), col_mean, Ytr)
    model = RidgeCV(alphas=ALPHAS)
    model.fit(Xtr, Y)
    return np.asarray(model.predict(Xte), dtype=np.float64)


def score(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> tuple[float, float]:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 10 or np.std(pred[ok]) == 0:
        return float("nan"), float("nan")
    return (
        float(spearmanr(pred[ok], obs[ok])[0]),
        float(pearsonr(pred[ok], obs[ok])[0]),
    )


# -------------------------------------------------------------------------- splits
def run_compound_cold(
    resp: pd.DataFrame, se: pd.DataFrame, mol: dict[str, NDArray[np.float64]]
) -> pd.DataFrame:
    R = resp.to_numpy()
    S = se.to_numpy()
    compounds = list(resp.columns)
    rows: list[dict[str, Any]] = []
    for j, comp in enumerate(compounds):
        tr = np.array([i for i in range(len(compounds)) if i != j])
        gene_mean = np.nanmean(R[:, tr], axis=1)
        for target in ("raw", "centered"):
            obs = R[:, j] if target == "raw" else R[:, j] - gene_mean
            tgt_tr = R[:, tr] if target == "raw" else R[:, tr] - gene_mean[:, None]
            cap = ceiling(obs, S[:, j])
            base = gene_mean if target == "raw" else np.zeros_like(gene_mean)
            rho, r = score(base, obs)
            rows.append(
                {
                    "split": "compound_cold",
                    "fold": comp,
                    "features": "none",
                    "model": "gene_mean",
                    "target": target,
                    "spearman": rho,
                    "pearson": r,
                    "ceiling": cap,
                }
            )
            # the control that decides whether the encoder is doing anything: k RANDOM
            # training compounds instead of the k most chemically similar ones. On the
            # centered target the gene-mean baseline predicts a constant and has no
            # correlation, so this is the null a neighbor method has to beat.
            rng = np.random.default_rng(1000 + j)
            for k in KNN_KS:
                draws = [
                    score(
                        np.ma.masked_invalid(
                            tgt_tr.T[rng.choice(len(tr), size=k, replace=False)]
                        )
                        .mean(axis=0)
                        .filled(np.nan),
                        obs,
                    )
                    for _ in range(N_RANDOM_DRAWS)
                ]
                rows.append(
                    {
                        "split": "compound_cold",
                        "fold": comp,
                        "features": "random",
                        "model": f"knn{k}",
                        "target": target,
                        "spearman": float(np.nanmedian([d[0] for d in draws])),
                        "pearson": float(np.nanmedian([d[1] for d in draws])),
                        "ceiling": cap,
                    }
                )
            pred = np.ma.masked_invalid(tgt_tr.T).mean(axis=0).filled(np.nan)
            rho, r = score(pred, obs)
            rows.append(
                {
                    "split": "compound_cold",
                    "fold": comp,
                    "features": "none",
                    "model": "mean_train_profile",
                    "target": target,
                    "spearman": rho,
                    "pearson": r,
                    "ceiling": cap,
                }
            )
            for enc, X in mol.items():
                sims = similarity(enc, X, np.array([j]), tr)[0]
                for k in KNN_KS:
                    pred = knn_predict(sims, tgt_tr.T, k)
                    rho, r = score(pred, obs)
                    rows.append(
                        {
                            "split": "compound_cold",
                            "fold": comp,
                            "features": enc,
                            "model": f"knn{k}",
                            "target": target,
                            "spearman": rho,
                            "pearson": r,
                            "ceiling": cap,
                        }
                    )
                Xtr, Xte = standardize(X[tr], X[[j]])
                pred = ridge_predict(Xtr, tgt_tr.T, Xte)[0]
                rho, r = score(pred, obs)
                rows.append(
                    {
                        "split": "compound_cold",
                        "fold": comp,
                        "features": enc,
                        "model": "ridge",
                        "target": target,
                        "spearman": rho,
                        "pearson": r,
                        "ceiling": cap,
                    }
                )
        print(f"  compound_cold fold {comp} done", flush=True)
    return pd.DataFrame(rows)


def run_gene_cold(
    resp: pd.DataFrame,
    se: pd.DataFrame,
    gene_emb: dict[str, NDArray[np.float64]],
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    R = resp.to_numpy()
    S = se.to_numpy()
    rng = np.random.default_rng(seed)
    assign = rng.integers(0, n_folds, size=R.shape[0])
    rows: list[dict[str, Any]] = []
    for f in range(n_folds):
        te = np.where(assign == f)[0]
        tr = np.where(assign != f)[0]
        compound_mean = np.nanmean(R[tr, :], axis=0)
        for target in ("raw", "centered"):
            # NOTE the two targets coincide for this split BY CONSTRUCTION. Centering
            # subtracts a per-compound constant, and a correlation taken over genes inside
            # one compound column is invariant to that constant, so raw and centered score
            # identically here. Both are emitted so the file has one schema across splits.
            # centered here removes the COMPOUND main effect, the analogous nuisance
            obs_mat = R[te, :] if target == "raw" else R[te, :] - compound_mean[None, :]
            tr_mat = R[tr, :] if target == "raw" else R[tr, :] - compound_mean[None, :]
            for enc, G in gene_emb.items():
                Xtr, Xte = standardize(G[tr], G[te])
                pred = ridge_predict(Xtr, tr_mat, Xte)
                for c, comp in enumerate(resp.columns):
                    cap = ceiling(obs_mat[:, c], S[te, c])
                    rho, r = score(pred[:, c], obs_mat[:, c])
                    rows.append(
                        {
                            "split": "gene_cold",
                            "fold": f"{f}:{comp}",
                            "features": enc,
                            "model": "ridge",
                            "target": target,
                            "spearman": rho,
                            "pearson": r,
                            "ceiling": cap,
                        }
                    )
            for c, comp in enumerate(resp.columns):
                cap = ceiling(obs_mat[:, c], S[te, c])
                base = (
                    np.full(len(te), compound_mean[c])
                    if target == "raw"
                    else np.zeros(len(te))
                )
                rho, r = score(base, obs_mat[:, c])
                rows.append(
                    {
                        "split": "gene_cold",
                        "fold": f"{f}:{comp}",
                        "features": "none",
                        "model": "compound_mean",
                        "target": target,
                        "spearman": rho,
                        "pearson": r,
                        "ceiling": cap,
                    }
                )
        print(f"  gene_cold fold {f} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="compound_cold", help="comma separated")
    args = ap.parse_args()
    splits = args.splits.split(",")

    resp, se, ident = load_matrix()
    print(
        f"matrix {resp.shape[0]} genes x {resp.shape[1]} compounds, "
        f"{int(np.isfinite(resp.to_numpy()).sum())} observed cells"
    )
    mol = load_molecule_embeddings(ident)
    print(f"molecule encoders with full coverage: {len(mol)} {sorted(mol)}")

    frames = []
    if "compound_cold" in splits:
        frames.append(run_compound_cold(resp, se, mol))
    if "gene_cold" in splits:
        gene_emb = load_gene_embeddings(resp.index)
        print(f"gene encoders loaded: {sorted(gene_emb)}")
        frames.append(run_gene_cold(resp, se, gene_emb))

    out = pd.concat(frames, ignore_index=True)
    for split, block in out.groupby("split"):
        block.to_csv(
            osp.join(RESULTS_DIR, f"baseline_ceilings_{split}.csv"), index=False
        )
    summary = (
        out.groupby(["split", "features", "model", "target"])
        .agg(
            folds=("spearman", "size"),
            spearman_median=("spearman", "median"),
            pearson_median=("pearson", "median"),
            ceiling_median=("ceiling", "median"),
        )
        .reset_index()
    )
    summary["frac_of_ceiling"] = summary["pearson_median"] / summary["ceiling_median"]
    summary = summary.sort_values(
        ["split", "target", "spearman_median"], ascending=[True, True, False]
    )
    path = osp.join(RESULTS_DIR, "baseline_ceilings_summary.csv")
    summary.to_csv(path, index=False)
    for (split, target), block in summary.groupby(["split", "target"]):
        print(f"\n=== {split} / {target} (median over folds)")
        print(block.head(14).to_string(index=False))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
