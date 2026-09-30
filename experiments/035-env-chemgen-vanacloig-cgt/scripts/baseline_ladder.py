# experiments/035-env-chemgen-vanacloig-cgt/scripts/baseline_ladder.py
# [[experiments.035-env-chemgen-vanacloig-cgt.scripts.baseline_ladder]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/035-env-chemgen-vanacloig-cgt/scripts/baseline_ladder
"""How far do simple compound-similarity models go on Vanacloig, compound-cold?

Every model here maps a compound's representation to the whole gene profile, so a
compound never seen in training can be predicted from its structure alone. That is the
inductive case: any small molecule with an embedding can be scored.

THE DATA is the genes by compounds matrix of ``vanacloig_data.load_cells``. For a fold,
the 32 or 33 compounds outside the test group are the model's compounds (training plus
the neural models' validation compounds: every pipeline gets the same non-test data and
uses it as it likes). Missing cells in the training block take their gene's training
mean, so they carry no signal.

THE MODELS, each over one compound kernel ``K``:

``krr``   kernel ridge regression of the gene-centered profile on ``K``, predicting
          ``gene mean + k(x)^T (K + lambda I)^-1 (Y - gene mean)``. Options: ``rank``
          truncates the centered training profiles to their top singular directions
          before the fit (a reduced-rank, gene-side denoising), ``clip`` winsorizes the
          centered training values at that many standard deviations.
``knn``   the similarity-weighted mean of the ``k`` most similar training compounds'
          centered profiles, weights ``softmax(K / tau)``.

THE KERNELS: ``linear`` on standardized features and ``rbf`` (median heuristic width)
for every embedding; ``tanimoto`` (min/max for counts) for the fingerprints; and
``combo:<set>``, the mean of the cosine-normalized kernels of a set of embeddings.

SELECTION IS NESTED. Inside each outer fold, every candidate (model, kernel, options) is
scored by leave-one-compound-out over the fold's non-test compounds, on the centered
target (``inner`` columns). Each (model, kernel) pair keeps its best options by that
inner score and is then scored once on the test compounds. The row
``model == "selected"`` is the pipeline that picks the (model, kernel, options) with the
best inner score across everything, per fold: that is the honest score of the whole
ladder, since the test compounds never influence any choice.

Writes ``results/ladder/ladder_scores.csv`` (per compound) and
``results/ladder/ladder_summary.csv``.
"""

from __future__ import annotations

import argparse
import itertools
import os
import os.path as osp
import sys
import time
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.stats import rankdata

sys.path.insert(0, osp.dirname(__file__))
from vanacloig_data import (  # noqa: E402
    VanacloigCells,
    load_cells,
    make_folds,
    score_compounds,
)

# a gene measured under no fit compound has no mean; it is predicted NaN and unscored
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "035-env-chemgen-vanacloig-cgt", "results", "ladder"
)
PREDICTIONS = osp.join(
    os.environ["DATA_ROOT"],
    "experiments",
    "035-env-chemgen-vanacloig-cgt",
    "predictions",
)
EMBEDDING_DIR = (
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/"
    "031-env-chemgen-vanacloig-hillenmeyer/experiments/"
    "031-env-chemgen-inhibitor-tolerance/results/embeddings"
)
EMBEDDING_DIR_035 = osp.join(
    os.environ["DATA_ROOT"],
    "experiments",
    "035-env-chemgen-vanacloig-cgt",
    "embeddings",
)
EMBEDDINGS = (
    "fcfp4_count",
    "ecfp4_count",
    "ecfp4_bit",
    "maccs",
    "rdkit_2d",
    "mol2vec",
    "chemberta2_mlm",
    "chemberta2_mtr",
    "molformer_xl",
    "roberta_zinc_480m",
    "mole_static",
    "unimol_v1",
)
FINGERPRINTS = ("fcfp4_count", "ecfp4_count", "ecfp4_bit", "maccs")
COMBOS = {
    "fingerprints": FINGERPRINTS,
    "pretrained": (
        "chemberta2_mlm",
        "chemberta2_mtr",
        "molformer_xl",
        "roberta_zinc_480m",
        "mole_static",
        "mol2vec",
    ),
}
LAMBDAS = np.logspace(-3, 3, 13)
RANKS = (2, 4, 8, 16, None)
CLIPS = (None, 3.0)
KS = (1, 3, 5, 10, None)
TAUS = (0.02, 0.05, 0.1, 0.3)


class Candidate(BaseModel):
    """One model configuration over one kernel."""

    model: str  # krr | knn
    kernel: str
    lam: float | None = None
    rank: int | None = None
    clip: float | None = None
    k: int | None = None
    tau: float | None = None

    def options(self) -> str:
        """The non-default options as ``name=value`` pairs."""
        return ",".join(
            f"{name}={value}"
            for name, value in self.model_dump(exclude={"model", "kernel"}).items()
            if value is not None
        )


def embedding_path(name: str) -> str:
    """A 031 embedding, or a 035 one (``pheno_*``) written by ``phenotypic_embeddings.py``."""
    directory = EMBEDDING_DIR_035 if name.startswith("pheno_") else EMBEDDING_DIR
    return osp.join(directory, f"{name}.npz")


def load_features(cells: VanacloigCells, name: str) -> NDArray[np.float64] | None:
    """[41, d] features in ``cells.compounds`` order; None if a compound has none."""
    data = np.load(embedding_path(name), allow_pickle=True)
    row = {key: i for i, key in enumerate(data["inchikey"])}
    if not all(key in row for key in cells.inchikeys):
        return None
    x = np.stack([data["X"][row[key]] for key in cells.inchikeys]).astype(np.float64)
    finite = np.isfinite(x).all(axis=0)
    return x[:, finite]


def tanimoto(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Min/max Tanimoto, which is the bit Tanimoto on binary input."""
    num = np.minimum(x[:, None, :], x[None, :, :]).sum(-1)
    den = np.maximum(x[:, None, :], x[None, :, :]).sum(-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)


def fold_kernel(
    kind: str, x: NDArray[np.float64], fit: NDArray[np.int64]
) -> NDArray[np.float64]:
    """A 41 x 41 kernel whose data-dependent parameters use the ``fit`` compounds only."""
    if kind == "tanimoto":
        return tanimoto(x)
    mu, sd = x[fit].mean(0), x[fit].std(0)
    keep = sd > 0
    z = (x[:, keep] - mu[keep]) / sd[keep]
    if kind == "linear":
        return z @ z.T / z.shape[1]
    assert kind == "rbf", kind
    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    fit_sq = sq[np.ix_(fit, fit)]
    width = np.median(fit_sq[np.triu_indices(len(fit), 1)])
    return np.exp(-sq / width)


def cosine_normalize(k: NDArray[np.float64]) -> NDArray[np.float64]:
    d = np.sqrt(np.clip(np.diag(k), 1e-12, None))
    return k / d[:, None] / d[None, :]


def centered_profiles(
    y: NDArray[np.float64], fit: list[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gene means over ``fit`` and the [len(fit), genes] centered, NaN-filled block."""
    mean = np.nanmean(y[:, fit], axis=1)
    block = y[:, fit] - mean[:, None]
    return mean, np.where(np.isnan(block), 0.0, block).T


def predict_candidates(
    candidates: list[Candidate],
    kernel: NDArray[np.float64],
    y: NDArray[np.float64],
    fit: list[int],
    target: list[int],
) -> dict[int, NDArray[np.float64]]:
    """Predicted [genes, len(target)] centered profiles for every candidate on one kernel.

    Candidates are indexed by their position; all share the kernel.
    """
    mean, yc = centered_profiles(y, fit)
    out: dict[int, NDArray[np.float64]] = {}
    k_fit = kernel[np.ix_(fit, fit)]
    k_new = kernel[np.ix_(target, fit)]
    evals, evecs = np.linalg.eigh(k_fit)
    evals = np.clip(evals, 0.0, None)
    sd = yc.std()
    cache: dict[tuple, NDArray[np.float64]] = {}
    for i, c in enumerate(candidates):
        if c.model == "krr":
            key = (c.rank, c.clip)
            if key not in cache:
                block = yc if c.clip is None else np.clip(yc, -c.clip * sd, c.clip * sd)
                if c.rank is not None and c.rank < min(block.shape):
                    u, s, vt = np.linalg.svd(block, full_matrices=False)
                    block = (u[:, : c.rank] * s[: c.rank]) @ vt[: c.rank]
                cache[key] = evecs.T @ block
            assert c.lam is not None
            alpha = evecs @ (cache[key] / (evals + c.lam * len(fit))[:, None])
            out[i] = (k_new @ alpha).T
        else:
            assert c.model == "knn" and c.tau is not None
            sim = k_new.copy()
            if c.k is not None and c.k < len(fit):
                cut = np.sort(sim, axis=1)[:, -c.k][:, None]
                sim = np.where(sim >= cut, sim, -np.inf)
            w = np.exp((sim - sim.max(1, keepdims=True)) / c.tau)
            w = w / w.sum(1, keepdims=True)
            out[i] = (w @ yc).T
    return {i: p + mean[:, None] for i, p in out.items()}


def fast_spearman(pred: NDArray[np.float64], obs: NDArray[np.float64]) -> float:
    ok = np.isfinite(pred) & np.isfinite(obs)
    if ok.sum() < 10 or np.std(pred[ok]) < 1e-10:
        return float("nan")
    a, b = rankdata(pred[ok]), rankdata(obs[ok])
    return float(np.corrcoef(a, b)[0, 1])


def inner_scores(
    candidates: list[Candidate],
    kernel: NDArray[np.float64],
    y: NDArray[np.float64],
    pool: list[int],
) -> NDArray[np.float64]:
    """Mean centered Spearman of every candidate, leaving one ``pool`` compound out."""
    scores = np.zeros((len(pool), len(candidates)))
    for n, left in enumerate(pool):
        fit = [j for j in pool if j != left]
        pred = predict_candidates(candidates, kernel, y, fit, fit + [left])
        measured_mean = np.nanmean(y[:, fit], axis=1)
        obs = y[:, left] - measured_mean
        for i, p in pred.items():
            centered = p[:, -1] - p[:, :-1].mean(axis=1)
            scores[n, i] = fast_spearman(centered, obs)
    return np.nanmean(scores, axis=0)


def grid(kernel: str) -> list[Candidate]:
    out = [
        Candidate(model="krr", kernel=kernel, lam=float(lam), rank=r, clip=c)
        for lam, r, c in itertools.product(LAMBDAS, RANKS, CLIPS)
    ]
    out += [
        Candidate(model="knn", kernel=kernel, k=k, tau=t)
        for k, t in itertools.product(KS, TAUS)
    ]
    return out


def run_fold(
    cells: VanacloigCells,
    features: dict[str, NDArray[np.float64]],
    fold_seed: int,
    fold_index: int,
    n_folds: int,
) -> pd.DataFrame:
    started = time.time()
    fold = make_folds(len(cells.compounds), n_folds, 4, fold_seed)[fold_index]
    pool = sorted(fold.train + fold.val)
    fit_idx = np.array(pool)
    y = cells.matrix(cells.response)

    kernels: dict[str, NDArray[np.float64]] = {}
    for name, x in features.items():
        kinds = ("linear", "rbf") + (("tanimoto",) if name in FINGERPRINTS else ())
        for kind in kinds:
            kernels[f"{kind}:{name}"] = fold_kernel(kind, x, fit_idx)
    # a combination is formed only over at least two of its members that were loaded
    for combo, members in COMBOS.items():
        parts = [
            cosine_normalize(kernels[f"rbf:{m}"]) for m in members if m in features
        ]
        if len(parts) >= 2:
            kernels[f"combo:{combo}"] = np.mean(parts, axis=0)
    if len(features) >= 2:
        kernels["combo:all"] = np.mean(
            [cosine_normalize(kernels[f"rbf:{m}"]) for m in features], axis=0
        )

    frames = []
    best_overall: tuple[float, str, Candidate] | None = None
    tested: dict[str, pd.DataFrame] = {}
    for name, kernel in kernels.items():
        candidates = grid(name)
        inner = inner_scores(candidates, kernel, y, pool)
        for model in ("krr", "knn"):
            idx = [i for i, c in enumerate(candidates) if c.model == model]
            best = idx[int(np.nanargmax(inner[idx]))]
            chosen = candidates[best]
            pred = predict_candidates([chosen], kernel, y, pool, pool + fold.test)[0]
            prediction = np.full(y.shape, np.nan)
            prediction[:, pool + fold.test] = pred
            scores = score_compounds(cells, prediction, pool, fold.test).assign(
                model=model,
                kernel=name,
                options=chosen.options(),
                inner_centered_spearman=float(inner[best]),
            )
            tested[f"{model}|{name}"] = scores
            frames.append(scores)
            if model == "krr" and name == "linear:fcfp4_count":
                # the ridge reference's full prediction, for stacking offline
                np.save(
                    osp.join(
                        PREDICTIONS, f"ridge_fold{fold_index}_seed{fold_seed}.npy"
                    ),
                    prediction.astype(np.float32),
                )
            if best_overall is None or inner[best] > best_overall[0]:
                best_overall = (float(inner[best]), f"{model}|{name}", chosen)
    assert best_overall is not None
    selected = tested[best_overall[1]].assign(
        model="selected",
        kernel=best_overall[2].kernel,
        options=f"{best_overall[2].model}:{best_overall[2].options()}",
    )
    frames.append(selected)
    out = pd.concat(frames, ignore_index=True).assign(
        fold_seed=fold_seed, fold=fold_index, n_fit_compounds=len(pool)
    )
    print(
        f"fold seed {fold_seed} fold {fold_index}: {len(kernels)} kernels, "
        f"selected {best_overall[1]} ({best_overall[2].options()}) inner "
        f"{best_overall[0]:.3f}, {time.time() - started:.0f} s",
        flush=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-table", required=True)
    parser.add_argument("--fold-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--embeddings", nargs="+", default=list(EMBEDDINGS))
    parser.add_argument("--tag", default="ladder")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS, exist_ok=True)

    cells = load_cells(args.cell_table, osp.join(EMBEDDING_DIR, "fcfp4_count.npz"))
    features: dict[str, NDArray[np.float64]] = {}
    for name in args.embeddings:
        x = load_features(cells, name)
        if x is None:
            print(
                f"{name}: a Vanacloig compound has no embedding; left out", flush=True
            )
            continue
        features[name] = x
    print(f"embeddings used: {sorted(features)}", flush=True)

    jobs = [(seed, k) for seed in args.fold_seeds for k in range(args.n_folds)]
    frames = Parallel(n_jobs=args.workers)(
        delayed(run_fold)(cells, features, seed, k, args.n_folds) for seed, k in jobs
    )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(osp.join(RESULTS_DIR, f"{args.tag}_scores.csv"), index=False)
    summary = (
        out.groupby(["model", "kernel", "target"])
        .agg(
            compounds=("spearman", "size"),
            spearman_median=("spearman", "median"),
            spearman_mean=("spearman", "mean"),
            pearson_median=("pearson", "median"),
            inner_mean=("inner_centered_spearman", "mean"),
        )
        .reset_index()
        .sort_values(["target", "spearman_median"], ascending=[True, False])
    )
    summary.to_csv(osp.join(RESULTS_DIR, f"{args.tag}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summary[summary["target"] == "centered"].head(40).to_string(index=False))
    print(summary[summary["model"] == "selected"].to_string(index=False))


if __name__ == "__main__":
    main()
