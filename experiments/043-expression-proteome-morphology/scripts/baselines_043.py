# experiments/043-expression-proteome-morphology/scripts/baselines_043.py
# [[experiments.043-expression-proteome-morphology.scripts.baselines_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/baselines_043
"""Baselines on exactly the trainer's validation records, scored by the trainer's metrics.

Owner, 2026-10-26: "can you run baselines for the methods we are comparing against so we
can see if they are helping at all?" The comparison is the three-seed recipe-v2 run
(``gs1024_w4_bs16``, ``_s1``, ``_s2``; arm ``all``, prep_v4, ontology conditioning, batch
16) read as the mean over epochs 41-50 of each seed.

Records, targets and metrics come from ``train_043.py`` itself, imported, not rewritten:

* ``PreparedStore043.load`` of ``prep_v4_routeA`` (store v2, ``global_split_v2``; the
  split sha256 is checked against the prepared store as ``train_043.run`` checks it);
* ``build_view`` with ``conf/arm/all.yaml``, ``target_transform=zscore`` and
  ``target_stats_by=dataset`` (the recipe: the legacy z, fitted on training rows);
* the validation records of each seed are ``eval_positions(view, VAL, 2000, seed)``, the
  call the trainer makes (only Vaishnav exceeds the cap, so only its subset moves with the
  seed); every baseline is scored on each seed's subset and the three are averaged;
* ``gene_level_metrics`` (median per-gene Pearson across held-out records, genes with at
  least 3 finite pairs, plus the centered per-strain Pearson against the training-mean
  profile) for gene-level datasets, ``column_pearson`` over records for Vaishnav, and the
  doubles statistic of ``evaluate_doubles`` (released Kemmeren units, centered per double
  on Kemmeren's training-mean profile, over all genes).

Fitting uses training records only. Baselines (``--stage fit``):

* ``constant``: the per-gene training mean ``train_mean_z``. It is constant across
  records, so every Pearson the trainer reports is undefined for it (NaN); it is listed,
  not plotted. Nadal-Ribelles has two environments in validation, so its per-environment
  training mean (``env_mean``) is a real ordering and is scored.
* ``additive_measured`` (doubles): measured single x plus measured single y, the
  trainer's ``val/doubles/additive_measured_...`` key, recomputed here as a check.
* ``ridge``: multi-output ridge (``linear_evals_043.ridge_fold``: kernel form, inputs and
  targets z-scored on the fitting rows, missing targets at z 0, lambda = (trace K / n) x
  10^g for g in -5..5 by 0.5) with g chosen on an INNER split of the training records (20%
  of training units held out, rng seed 0, grouped by unit) by the median per-gene Pearson,
  then refit on all training records. Vaishnav uses ``primal_ridge_fold`` (n >> p).
* ``knn_k`` for k in 5, 20, 50: the nanmean measured profile of the k nearest training
  records by cosine on the training-z-scored features, within the record's environment.
* ``hgb``: sklearn ``HistGradientBoostingRegressor`` per target gene (fixed, not tuned:
  200 iterations, learning rate 0.05, 15 leaves, min(20, n_train // 5) records per
  leaf so the two smallest panels can split at all, l2 1.0, no early
  stopping) on the first 64 principal components of the training-z-scored features (plus
  the environment indicator); Vaishnav fits one model on the raw k-mer counts.

Features per dataset (the frozen inputs the 043 linear ladder already uses):

* deletion panels (Kemmeren, Messner, Zelezniak, Hu, Nadal-Ribelles): the composite token
  of the deleted gene (``linear_evals_043.load_tokens``: FUDT upstream 768 + FUDT downstream
  768 + ProtT5-XL 1,024 + CaLM 768, each LayerNorm-standardized per gene, 3,328 wide); a
  record whose deleted ORF has no index position gets the training-mean feature (the
  intercept prediction) and is counted as ``n_val_no_token``; Nadal-Ribelles adds its
  environment indicator.
* doubles: every learned baseline is fitted on Kemmeren and predicts each single from its
  gene's token; the double is the sum of the two predicted singles in released units (the
  same additive form as the null). For kNN the query gene's own Kemmeren record is
  excluded from its neighbors; ridge and HGB are in-sample on the singles (the trained
  model also trains on those singles).
* isolates (Caudal, Muenzner protein and turnover): the per-gene call indicators
  (variant, loss of function, absence; 3 x 6,607 columns) of the GBLUP kinship in
  ``linear_evals_043.run_isolates``; ridge on the standardized indicators is GBLUP up to
  the penalty scale.
* segregants (Boocock pooled, N'Guessan): the parent-of-origin vector of
  ``linear_evals_043.run_segregants`` (+1 non-reference parent, -1 reference, 0
  unassigned, at each gene's midpoint).
* Vaishnav: k-mer counts (k 1 to 6, 5,460 columns) of the 80-mer plus the library
  indicator (``linear_evals_043.kmer_matrix``).

Jackson 2020 has no validation records under ``global_split_v2`` (its 12 TF genes and HO
are training units), so the trainer scores it on no validation key and it is listed as not
scored.

Stages: ``fit`` (CPU, slurm: ``gh_baselines_043.slurm``) writes
``results/baselines_043.json``; ``trained`` reads the three runs' unsampled W&B history
over epochs 41-50 for every ``val/`` key the table needs and adds it to the json;
``report`` writes the figure to ``ASSET_IMAGES_DIR/043-expression-proteome-morphology/``
and prints the markdown table.

    python experiments/043-expression-proteome-morphology/scripts/baselines_043.py --stage fit
    python experiments/043-expression-proteome-morphology/scripts/baselines_043.py --stage trained
    python experiments/043-expression-proteome-morphology/scripts/baselines_043.py --stage report
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import subprocess
import sys
import time
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from dotenv import load_dotenv

SCRIPTS = osp.dirname(osp.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from global_split_043 import Role, load_split  # noqa: E402
from linear_evals_043 import ALL as RAW_NAMES  # noqa: E402
from linear_evals_043 import (  # noqa: E402
    LinearEvalConfig,
    kmer_matrix,
    load_tokens,
    primal_ridge_fold,
    ridge_fold,
)
from store_version_043 import store_paths  # noqa: E402
from train_043 import (  # noqa: E402
    ROLE_CODE,
    ArmConfig,
    PreparedStore043,
    TrainingView,
    _nanmedian,
    build_view,
    column_pearson,
    eval_positions,
    gene_level_metrics,
)

EXPERIMENT = "043-expression-proteome-morphology"
STORE_VERSION = "v2"
PREP_VERSION = "prep_v4"
SEEDS = (0, 1, 2)
VAL_CAP = 2000
MIN_STRAINS = 3
RIDGE_GRID = [x / 2 for x in range(-10, 11)]  # log10 of lambda / (trace K / n): -5..5
INNER_FRAC = 0.2
INNER_SEED = 0
KS = (5, 20, 50)
HGB_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}
HGB_PCS = 64
HGB_MIN_TRAIN = 10
DELETION = ("kemmeren", "messner", "zelezniak", "hu", "nadal")
ISOLATES = ("caudal", "muenzner", "turnover")
SEGREGANTS = ("boocock", "nguessan")
#: Order of the table and the figure (the trainer's datasets, then the doubles).
ORDER = (
    "kemmeren",
    "messner",
    "zelezniak",
    "caudal",
    "muenzner",
    "turnover",
    "boocock",
    "nguessan",
    "vaishnav",
    "hu",
    "nadal",
    "doubles",
)
NOT_SCORED = {
    "jackson": "no validation records under global_split_v2 (its 12 TF genes and HO are "
    "training units); the trainer logs no val/jackson key"
}
TRAINED_RUNS = ("gs1024_w4_bs16", "gs1024_w4_bs16_s1", "gs1024_w4_bs16_s2")
TRAINED_WINDOW = (41, 50)
WANDB_ENTITY = "zhao-group"
WANDB_PROJECT = "torchcell_043-expression-proteome-morphology"
LABEL = {
    "kemmeren": "Kemmeren 2014",
    "messner": "Messner 2023",
    "zelezniak": "Zelezniak 2018",
    "caudal": "Caudal 2024",
    "muenzner": "Muenzner 2024 protein",
    "turnover": "Muenzner 2024 turnover",
    "boocock": "Boocock 2025 pooled",
    "nguessan": "N'Guessan 2025",
    "vaishnav": "Vaishnav 2022",
    "hu": "Hu 2007",
    "nadal": "Nadal-Ribelles 2025",
    "doubles": "Sameith 2015 doubles",
}


# ============================================================================ helpers
def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=SCRIPTS,
    ).stdout.strip()


def _zfit_rows(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """``X`` z-scored with the mean and sd of ``rows`` (constant columns divided by 1)."""
    m = X[rows].mean(0)
    s = X[rows].std(0)
    s = np.where(s > 0, s, 1.0)
    return (X - m) / s


def inner_split(units: npt.NDArray[np.str_], seed: int = INNER_SEED) -> np.ndarray:
    """Positions (into ``units``) of the inner held-out records: ``INNER_FRAC`` of the
    distinct units, drawn once, every record of a drawn unit held out together.
    """
    uniq = np.unique(units)
    rng = np.random.default_rng(seed)
    held = set(
        rng.choice(uniq, max(1, int(round(INNER_FRAC * len(uniq)))), replace=False)
    )
    return np.nonzero([u in held for u in units])[0].astype(np.int64)


def knn_predict(
    F: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    env: np.ndarray,
    ks: tuple[int, ...],
    exclude: dict[int, np.ndarray] | None = None,
    chunk: int = 1024,
) -> dict[int, np.ndarray]:
    """Nanmean of the measured rows ``Y`` of the k most cosine-similar training rows of
    the same environment. ``F`` are z-scored features (a zero row has no direction and
    gets the training mean). ``exclude[i]`` lists training rows barred for query ``i``
    (an index into ``te``).
    """
    norm = np.linalg.norm(F, axis=1, keepdims=True)
    Fn = np.where(norm > 0, F / np.where(norm > 0, norm, 1.0), 0.0)
    out = {k: np.full((len(te), Y.shape[1]), np.nan) for k in ks}
    kmax = max(ks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_all = np.nanmean(Y[tr], 0)
        for e in np.unique(env[te]):
            pool = tr[env[tr] == e]
            q_all = np.nonzero(env[te] == e)[0]
            mean_env = np.nanmean(Y[pool], 0)
            for s in range(0, len(q_all), chunk):
                q = q_all[s : s + chunk]
                sim = Fn[te[q]] @ Fn[pool].T
                if exclude is not None:
                    for j, qi in enumerate(q):
                        bar = exclude.get(int(qi))
                        if bar is not None:
                            sim[j, np.isin(pool, bar)] = -np.inf
                kk = min(kmax, len(pool))
                top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
                order = np.take_along_axis(sim, top, 1).argsort(1)[:, ::-1]
                top = np.take_along_axis(top, order, 1)
                for k in ks:
                    nb = pool[top[:, : min(k, kk)]]
                    out[k][q] = np.nanmean(Y[nb], axis=1)
                dead = norm[te[q], 0] == 0
                for k in ks:
                    out[k][q[dead]] = (
                        mean_env if len(np.unique(env[tr])) > 1 else mean_all
                    )
    return out


def _hgb_columns(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, cols: list[int]
) -> list[tuple[int, np.ndarray]]:
    """One HGB per target column (a joblib task; one thread per worker)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    res = []
    with threadpool_limits(1):
        for j, c in enumerate(cols):
            y = Ytr[:, j]
            ok = np.isfinite(y)
            leaf = min(HGB_PARAMS["min_samples_leaf"], max(2, int(ok.sum()) // 5))
            m = HistGradientBoostingRegressor(
                **{**HGB_PARAMS, "min_samples_leaf": leaf}
            )
            m.fit(Xtr[ok], y[ok])
            res.append((c, m.predict(Xte)))
    return res


def hgb_predict(
    F: np.ndarray,
    Y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    cols: np.ndarray,
    extra: np.ndarray | None,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """HGB per column of ``cols`` on the top ``HGB_PCS`` principal components of ``F``
    (fitted on ``tr``) plus ``extra`` columns; other columns stay NaN.
    """
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA

    n_pc = min(HGB_PCS, len(tr) - 1, F.shape[1])
    pca = PCA(n_components=n_pc, svd_solver="randomized", random_state=0)
    pca.fit(F[tr])
    Z = pca.transform(F)
    if extra is not None:
        Z = np.concatenate([Z, extra], 1)
    P = np.full((len(te), Y.shape[1]), np.nan)
    chunks = [list(cols[i : i + 32]) for i in range(0, len(cols), 32)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_hgb_columns)(Z[tr], Y[np.ix_(tr, c)], Z[te], c) for c in chunks
    )
    for part in results:
        for c, p in part:
            P[:, c] = p
    return P, {
        **HGB_PARAMS,
        "n_principal_components": int(n_pc),
        "min_samples_leaf_rule": "min(20, n_finite_train // 5), at least 2",
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_extra_columns": 0 if extra is None else int(extra.shape[1]),
        "n_target_columns_fitted": int(len(cols)),
    }


# ============================================================================ inputs
class Inputs:
    """The prepared store, the trainer's view and the validation subsets of each seed."""

    def __init__(self, data_root: str, exp_dir: str) -> None:
        """Load prep_v4, check its split, build the recipe view and the val subsets."""
        gen = store_paths(STORE_VERSION)
        self.prep_dir = osp.join(
            data_root,
            "data/torchcell/experiments",
            EXPERIMENT,
            "train_043_prep",
            f"{PREP_VERSION}_routeA",
        )
        self.split_file = osp.join(
            exp_dir, "conf", "splits", f"{gen.split_version}.json"
        )
        self.cache = osp.join(
            data_root, "data/torchcell/experiments", EXPERIMENT, gen.linear_evals_cache
        )
        self.store = PreparedStore043.load(self.prep_dir)
        split = load_split(self.split_file)
        if (split.version, split.store_sha256) != (
            self.store.meta.split_version,
            self.store.meta.split_store_sha256,
        ):
            raise ValueError("the prepared store was built under another split file")
        with open(osp.join(exp_dir, "conf", "arm", "all.yaml")) as f:
            arm_yaml = yaml.safe_load(f)
        self.arm = ArmConfig(**arm_yaml)
        self.view: TrainingView = build_view(
            self.store, self.arm, None, "zscore", "dataset", "dataset"
        )
        self.val = {
            s: eval_positions(self.view, ROLE_CODE[Role.VAL], VAL_CAP, s) for s in SEEDS
        }
        self.genes = list(self.store.meta.genes)
        with open(osp.join(self.cache, "genes.json")) as f:
            if json.load(f) != self.genes:
                raise ValueError(
                    "linear-evals cache genes differ from the prepared store"
                )
        self.cfg = LinearEvalConfig(
            data_root=data_root, experiment_dir=exp_dir, split_path=self.split_file
        )
        self._tokens: np.ndarray | None = None

    @property
    def tokens(self) -> np.ndarray:
        """Composite token per gene (6,607 x 3,328), ``load_tokens``."""
        if self._tokens is None:
            self._tokens = load_tokens(self.cfg, self.genes)["composite"]
        return self._tokens

    def cache_rows(self, short: str) -> dict[int, dict[str, Any]]:
        """linear_evals cache rows of one dataset keyed by raw index."""
        with open(osp.join(self.cache, f"{RAW_NAMES[short]}.pkl"), "rb") as f:
            rows = pickle.load(f)
        return {int(r["raw_index"]): r for r in rows}

    def positions(self, d: str) -> tuple[np.ndarray, np.ndarray]:
        """Training positions of ``d`` and the union of the seeds' validation positions."""
        v = self.view
        tr = v.positions(ROLE_CODE[Role.TRAIN], d)
        sel = [p[v.local_dataset[p] == v.datasets.index(d)] for p in self.val.values()]
        return tr, np.unique(np.concatenate(sel))

    def gene_targets(self, pos: np.ndarray) -> np.ndarray:
        """``view.Z`` rows of arm positions (the trainer's scoring target)."""
        v = self.view
        return v.Z[self.store.rec_yrow[v.rec[pos]]].astype(np.float64)

    def deleted_genes(self, pos: np.ndarray) -> list[list[int]]:
        """Index positions of every deleted gene of each record (empty = off index)."""
        s, v = self.store, self.view
        out = []
        for p in pos:
            r = v.rec[p]
            m = np.arange(s.mark_ptr[r], s.mark_ptr[r + 1])
            out.append([int(g) for g in s.mark_gene[m][s.mark_on_graph[m]]])
        return out


def token_features(inp: Inputs, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean composite token of each record's deleted genes; NaN rows = no indexed gene."""
    T = inp.tokens
    genes = inp.deleted_genes(pos)
    X = np.full((len(pos), T.shape[1]), np.nan)
    has = np.zeros(len(pos), bool)
    for i, gs in enumerate(genes):
        if gs:
            X[i] = T[gs].mean(0)
            has[i] = True
    return X, has


def check_units(inp: Inputs, pos: np.ndarray) -> int:
    """Deletion records: the marked gene must be the record's unit (its released ORF).

    Returns how many single-gene records name a unit absent from the index (no token).
    """
    s, v = inp.store, inp.view
    genes = inp.deleted_genes(pos)
    gidx = {g: i for i, g in enumerate(inp.genes)}
    off = 0
    for p, gs in zip(pos, genes, strict=True):
        unit = str(s.rec_unit[v.rec[p]])
        if unit in gidx:
            if gs != [gidx[unit]]:
                raise ValueError(f"record {v.rec[p]}: unit {unit} but marks {gs}")
        else:
            if gs:
                raise ValueError(
                    f"record {v.rec[p]}: off-index unit {unit} has marks {gs}"
                )
            off += 1
    return off


# ============================================================================ scoring
def score_gene(
    inp: Inputs, d: str, val_pos: np.ndarray, P_val: np.ndarray
) -> dict[str, Any]:
    """Trainer metrics of predictions ``P_val`` (rows = ``val_pos``) on each seed's subset."""
    Y = inp.gene_targets(val_pos)
    tm = inp.view.train_mean_z[d]
    per_seed = []
    for s in SEEDS:
        sub = inp.val[s]
        sub = sub[inp.view.local_dataset[sub] == inp.view.datasets.index(d)]
        idx = np.searchsorted(val_pos, sub)
        m = gene_level_metrics(P_val[idx], Y[idx], tm, MIN_STRAINS)
        r = column_pearson(P_val[idx], Y[idx], MIN_STRAINS)
        m["n_genes_scored"] = float(np.isfinite(r).sum())
        per_seed.append(m)
    return _seed_mean(per_seed)


def score_record(inp: Inputs, val_pos: np.ndarray, p_val: np.ndarray) -> dict[str, Any]:
    """Vaishnav: Pearson over records on each seed's capped subset."""
    v = inp.view
    y = v.z_scalar[v.rec[val_pos]].astype(np.float64)
    per_seed = []
    for s in SEEDS:
        sub = inp.val[s]
        sub = sub[v.local_dataset[sub] == v.datasets.index("vaishnav")]
        idx = np.searchsorted(val_pos, sub)
        per_seed.append(
            {
                "pearson": float(column_pearson(p_val[idx, None], y[idx, None], 3)[0]),
                "n_strains": float(len(idx)),
            }
        )
    return _seed_mean(per_seed)


def _seed_mean(per_seed: list[dict[str, float]]) -> dict[str, Any]:
    keys = per_seed[0].keys()
    out: dict[str, Any] = {}
    for k in keys:
        vals = np.asarray([m[k] for m in per_seed], np.float64)
        out[k] = float(np.mean(vals)) if np.isfinite(vals).all() else float("nan")
        out[f"{k}_per_seed_subset"] = [float(x) for x in vals]
    return out


# ============================================================================ fitting
def fit_gene_dataset(
    inp: Inputs,
    d: str,
    F_tr: np.ndarray,
    F_val: np.ndarray,
    env_tr: np.ndarray,
    env_val: np.ndarray,
    tr: np.ndarray,
    val: np.ndarray,
    n_jobs: int,
    F_query: np.ndarray | None = None,
    query_exclude: dict[int, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Every baseline on one gene-level dataset; also predictions for ``F_query`` rows
    (the doubles' singles), returned per baseline in z.
    """
    Ytr = inp.gene_targets(tr)
    Yval = inp.gene_targets(val)
    cols = np.nonzero(np.isfinite(Ytr).any(0))[0]
    nq = 0 if F_query is None else len(F_query)
    F = np.concatenate([F_tr, F_val] + ([F_query] if F_query is not None else []), 0)
    n_tr, n_val = len(tr), len(val)
    env = np.concatenate([env_tr, env_val, np.full(nq, env_tr[0])])
    # rows with no feature (an off-index deletion): the training-mean feature, which is
    # exactly 0 after the z-scoring below (the intercept prediction; kNN gives the
    # training mean for a zero row)
    miss = ~np.isfinite(F).all(1)
    F[miss] = np.nanmean(F[:n_tr], 0)
    Fz = _zfit_rows(F, np.arange(n_tr))
    Fz[miss] = 0.0
    multi_env = len(np.unique(env_tr)) > 1
    Ez = None
    if multi_env:
        envs = np.unique(env_tr)
        E = (env[:, None] == envs[None, 1:]).astype(np.float64)
        Ez = _zfit_rows(E, np.arange(n_tr))
    Fr = Fz if Ez is None else np.concatenate([Fz, Ez], 1)  # ridge input
    Yall = np.full((len(F), len(cols)), np.nan)
    Yall[:n_tr] = Ytr[:, cols]
    tr_i = np.arange(n_tr)
    te_i = np.arange(n_tr, len(F))
    units = inp.store.rec_unit[inp.view.rec[tr]]
    va_i = inner_split(units)
    G = Ytr.shape[1]

    def expand(Pc: np.ndarray) -> np.ndarray:
        P = np.full((Pc.shape[0], G), np.nan)
        P[:, cols] = Pc
        return P

    out: dict[str, Any] = {}
    preds_q: dict[str, np.ndarray] = {}
    tm = inp.view.train_mean_z[d]
    # constant
    m = score_gene(inp, d, val, np.broadcast_to(tm, Yval.shape).copy())
    out["constant"] = {
        **m,
        "hyperparameter": None,
        "note": "per-gene training mean; constant across records, so every Pearson is "
        "undefined (NaN; linear_evals_043 scores it 0 by convention)",
    }
    if F_query is not None:
        preds_q["constant"] = np.broadcast_to(tm, (nq, G)).copy()
    if multi_env:
        Pv = np.full(Yval.shape, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for e in np.unique(env_val):
                Pv[env_val == e] = np.nanmean(Ytr[env_tr == e], 0)
        out["env_mean"] = {
            **score_gene(inp, d, val, Pv),
            "hyperparameter": None,
            "note": "per-gene training mean within the record's environment",
        }
    # ridge
    t0 = time.time()
    P, best, scores = ridge_fold(Fr, None, Yall, tr_i, va_i, te_i, RIDGE_GRID)
    out["ridge"] = {
        **score_gene(inp, d, val, expand(P[:n_val])),
        "hyperparameter": {
            "log10_lambda_rel": best,
            "lambda_rule": "(trace K / n) x 10^g on the inner split",
            "inner_scores_median_per_gene_pearson": scores,
            "grid_edge": bool(best in (RIDGE_GRID[0], RIDGE_GRID[-1])),
            "n_inner_heldout_records": int(len(va_i)),
        },
        "seconds": time.time() - t0,
    }
    if F_query is not None:
        preds_q["ridge"] = expand(P[n_val:])
    print(
        f"{d} ridge g={best} {out['ridge'].get('pearson_per_gene_median'):.4f}",
        flush=True,
    )
    # kNN
    t0 = time.time()
    excl = None
    if query_exclude is not None:
        excl = {n_val + i: v for i, v in query_exclude.items()}
    kp = knn_predict(Fz, Yall, tr_i, te_i, env, KS, excl)
    for k in KS:
        out[f"knn_{k}"] = {
            **score_gene(inp, d, val, expand(kp[k][:n_val])),
            "hyperparameter": {
                "k": k,
                "metric": "cosine on training-z-scored features",
            },
            **(
                {
                    "note": f"k >= {n_tr} training records: every record gets the same neighbors, a constant, so undefined"
                }
                if k >= n_tr
                else {}
            ),
            "seconds": time.time() - t0,
        }
        if F_query is not None:
            preds_q[f"knn_{k}"] = expand(kp[k][n_val:])
    # HGB
    t0 = time.time()
    n_val_fin = np.isfinite(Yval[:, cols]).sum(0)
    n_tr_fin = np.isfinite(Ytr[:, cols]).sum(0)
    fit_cols = np.nonzero(
        (n_tr_fin >= HGB_MIN_TRAIN) & ((n_val_fin >= MIN_STRAINS) | (nq > 0))
    )[0]
    Ph, info = hgb_predict(Fz, Yall, tr_i, te_i, fit_cols, Ez, n_jobs)
    out["hgb"] = {
        **score_gene(inp, d, val, expand(Ph[:n_val])),
        "hyperparameter": {**info, "tuned": False},
        "seconds": time.time() - t0,
    }
    if F_query is not None:
        preds_q["hgb"] = expand(Ph[n_val:])
    print(
        f"{d} hgb {out['hgb'].get('pearson_per_gene_median'):.4f} "
        f"({time.time() - t0:.0f} s)",
        flush=True,
    )
    return out, preds_q


def run_gene_dataset(inp: Inputs, d: str, n_jobs: int) -> dict[str, Any]:
    """Features for one gene-level dataset, then every baseline."""
    tr, val = inp.positions(d)
    s, v = inp.store, inp.view
    env_tr = s.rec_env[v.rec[tr]].astype(np.int64)
    env_val = s.rec_env[v.rec[val]].astype(np.int64)
    info: dict[str, Any] = {
        "level": "gene",
        "statistic": "median over genes of the Pearson r across validation records "
        "(gene_level_metrics; genes with >= 3 finite pairs)",
        "n_train_records": int(len(tr)),
        "n_val_records_union": int(len(val)),
        "n_train_environments": int(len(np.unique(env_tr))),
    }
    if d in DELETION:
        info["features"] = "composite token of the deleted gene (3,328)"
        info["n_train_no_token"] = check_units(inp, tr)
        info["n_val_no_token"] = check_units(inp, val)
        F_tr, _ = token_features(inp, tr)
        F_val, _ = token_features(inp, val)
    elif d in ISOLATES:
        rows = inp.cache_rows(d)
        G = len(inp.genes)

        def ind(pos: np.ndarray) -> np.ndarray:
            Z = np.zeros((len(pos), 3 * G))
            for i, p in enumerate(pos):
                gi, pt = rows[int(s.rec_raw_index[v.rec[p]])]["marked"]
                for t, off in ((1, 0), (2, G), (3, 2 * G)):
                    Z[i, off + gi[pt == t]] = 1.0
            return Z

        F_tr, F_val = ind(tr), ind(val)
        keep = F_tr.std(0) > 0
        F_tr, F_val = F_tr[:, keep], F_val[:, keep]
        info["features"] = (
            "per-gene call indicators (variant, loss of function, absence), the GBLUP "
            f"kinship input of linear_evals_043; {int(keep.sum())} non-constant columns"
        )
    elif d in SEGREGANTS:
        rows = inp.cache_rows(d)
        F_tr = np.stack(
            [rows[int(s.rec_raw_index[v.rec[p]])]["markers"] for p in tr]
        ).astype(np.float64)
        F_val = np.stack(
            [rows[int(s.rec_raw_index[v.rec[p]])]["markers"] for p in val]
        ).astype(np.float64)
        keep = F_tr.std(0) > 0
        F_tr, F_val = F_tr[:, keep], F_val[:, keep]
        info["features"] = (
            "parent-of-origin vector of linear_evals_043 (+1 non-reference, -1 "
            f"reference, 0 unassigned); {int(keep.sum())} non-constant columns"
        )
    else:
        raise ValueError(d)
    res, _ = fit_gene_dataset(inp, d, F_tr, F_val, env_tr, env_val, tr, val, n_jobs)
    return {**info, "baselines": res}


def run_doubles(inp: Inputs, n_jobs: int) -> dict[str, Any]:
    """Kemmeren-fitted baselines, additive over the two singles, as evaluate_doubles."""
    s, v = inp.store, inp.view
    tr, _ = inp.positions("kemmeren")
    _, val = inp.positions("kemmeren")
    dbl = v.doubles
    singles = v.doubles_singles
    n = len(dbl)
    pairs = s.rec_genes2[v.rec[dbl]]
    T = inp.tokens
    F_q = np.concatenate([T[pairs[:, 0]], T[pairs[:, 1]]], 0)
    # kNN: the query gene's own Kemmeren record is not its neighbor
    kem_gene = s.rec_genes2[v.rec[tr], 0]
    excl = {
        i: np.nonzero(kem_gene == g)[0]
        for i, g in enumerate(np.concatenate([pairs[:, 0], pairs[:, 1]]))
    }
    F_tr, _ = token_features(inp, tr)
    F_val, _ = token_features(inp, val)
    env_tr = s.rec_env[v.rec[tr]].astype(np.int64)
    env_val = s.rec_env[v.rec[val]].astype(np.int64)
    kem, preds_q = fit_gene_dataset(
        inp, "kemmeren", F_tr, F_val, env_tr, env_val, tr, val, n_jobs, F_q, excl
    )
    st = v.stats.get("kemmeren")
    mean_profile = st.invert(v.train_mean_z["kemmeren"])
    measured = st.invert(
        v.Z[s.rec_yrow[v.rec[np.concatenate([dbl, singles[:, 0], singles[:, 1]])]]]
    )
    m_xy, m_x, m_y = measured[:n], measured[n : 2 * n], measured[2 * n :]

    def centered(a: np.ndarray) -> float:
        return _nanmedian(
            column_pearson((a - mean_profile).T, (m_xy - mean_profile).T, 3)
        )

    def n_scored(a: np.ndarray) -> int:
        r = column_pearson((a - mean_profile).T, (m_xy - mean_profile).T, 3)
        return int(np.isfinite(r).sum())

    out: dict[str, Any] = {
        "additive_measured": {
            "pearson_per_strain_centered_median": centered(m_x + m_y),
            "n_doubles_scored": n_scored(m_x + m_y),
            "hyperparameter": None,
            "note": "measured single x + measured single y (the trainer's "
            "val/doubles/additive_measured key, recomputed)",
        }
    }
    for name, P in preds_q.items():
        rel = st.invert(P)
        a = rel[:n] + rel[n:]
        hp = kem[name].get("hyperparameter")
        out[name] = {
            "pearson_per_strain_centered_median": centered(a),
            "n_doubles_scored": n_scored(a),
            "hyperparameter": hp
            if name != "ridge"
            else {k: hp[k] for k in ("log10_lambda_rel", "lambda_rule", "grid_edge")},
            "note": "Kemmeren-fitted baseline; predicted single x + predicted single y "
            "in released units"
            + (
                ", query gene's own record excluded from its neighbors"
                if name.startswith("knn")
                else ""
            )
            + ("; constant, so undefined" if name == "constant" else ""),
        }
    return {
        "level": "split B",
        "statistic": "median over doubles of the centered per-double Pearson r over genes, "
        "released Kemmeren units, centered on Kemmeren's training-mean profile "
        "(evaluate_doubles)",
        "n_doubles": int(n),
        "features": "composite tokens of the two deleted genes, through Kemmeren-fitted "
        "baselines",
        "baselines": out,
    }


def run_vaishnav(inp: Inputs, n_jobs: int) -> dict[str, Any]:
    """Record-level: k-mer ridge, kNN and HGB on the Vaishnav validation subsets."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    s, v = inp.store, inp.view
    tr, val = inp.positions("vaishnav")
    rows = inp.cache_rows("vaishnav")
    pos = np.concatenate([tr, val])
    mers = [str(s.rec_unit[v.rec[p]]) for p in pos]
    libs_all = [rows[int(s.rec_raw_index[v.rec[p]])]["library"] for p in pos]
    for p, m in zip(pos, mers, strict=True):
        if rows[int(s.rec_raw_index[v.rec[p]])]["mer"] != m:
            raise ValueError("Vaishnav unit is not the cached 80-mer")
    libs = sorted(set(libs_all))
    L = np.asarray([[lb == x for x in libs[1:]] for lb in libs_all], np.float32)
    X = np.concatenate([kmer_matrix(mers), L], 1)
    X = X[:, X[: len(tr)].std(0) > 0]
    y = v.z_scalar[v.rec[pos]].astype(np.float64)
    fin_tr = np.nonzero(np.isfinite(y[: len(tr)]))[0]
    n_tr = len(tr)
    te_i = np.arange(n_tr, len(pos))
    units = s.rec_unit[v.rec[tr[fin_tr]]]
    va_i = fin_tr[inner_split(units)]
    # primal_ridge_fold needs finite y on fitting rows: fit on finite training rows only
    t0 = time.time()
    yf = np.where(np.isfinite(y), y, 0.0)
    p, best, scores = primal_ridge_fold(X, yf, fin_tr, va_i, te_i, RIDGE_GRID)
    out: dict[str, Any] = {
        "constant": {
            **score_record(inp, val, np.full(len(val), np.nanmean(y[fin_tr]))),
            "hyperparameter": None,
            "note": "training mean; constant across promoters, Pearson undefined",
        },
        "ridge": {
            **score_record(inp, val, p),
            "hyperparameter": {
                "log10_lambda_rel": best,
                "inner_scores_pearson": scores,
                "grid_edge": bool(best in (RIDGE_GRID[0], RIDGE_GRID[-1])),
            },
            "seconds": time.time() - t0,
        },
    }
    print(f"vaishnav ridge g={best} {out['ridge']['pearson']:.4f}", flush=True)
    t0 = time.time()
    Xz = _zfit_rows(X.astype(np.float64), fin_tr)
    Y1 = y[:, None].copy()
    env = np.zeros(len(pos), np.int64)
    kp = knn_predict(Xz, Y1, fin_tr, te_i, env, KS)
    for k in KS:
        out[f"knn_{k}"] = {
            **score_record(inp, val, kp[k][:, 0]),
            "hyperparameter": {
                "k": k,
                "metric": "cosine on training-z-scored k-mer counts",
            },
            "seconds": time.time() - t0,
        }
    del Xz
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(**{**HGB_PARAMS, "max_iter": 300})
    hgb.fit(X[fin_tr], y[fin_tr])
    out["hgb"] = {
        **score_record(inp, val, hgb.predict(X[te_i])),
        "hyperparameter": {
            **HGB_PARAMS,
            "max_iter": 300,
            "features": "raw k-mer counts + library",
            "tuned": False,
        },
        "seconds": time.time() - t0,
    }
    print(f"vaishnav hgb {out['hgb']['pearson']:.4f}", flush=True)
    return {
        "level": "record",
        "statistic": "Pearson r over validation promoters (each seed's 2,000-record "
        "subset, eval_positions)",
        "n_train_records": int(len(fin_tr)),
        "n_val_records_union": int(len(val)),
        "features": f"k-mer counts (k 1 to 6) + library indicator, {X.shape[1]} columns",
        "baselines": out,
    }


def stage_fit(inp: Inputs, out_path: str, n_jobs: int, only: list[str] | None) -> None:
    """Every dataset; the json is rewritten after each so a partial run keeps its parts."""
    payload: dict[str, Any] = {}
    if osp.exists(out_path):
        with open(out_path) as f:
            payload = json.load(f)
    payload.update(
        {
            "generated_by": "experiments/043-expression-proteome-morphology/scripts/baselines_043.py",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torchcell_commit": _git_commit(),
            "inputs": {
                "prep_dir": inp.prep_dir,
                "prep_meta_slurm_job": inp.store.meta.slurm_job_id,
                "split_version": inp.store.meta.split_version,
                "split_store_sha256": inp.store.meta.split_store_sha256,
                "arm": inp.arm.model_dump(),
                "target": "legacy z (target_transform zscore, target_stats_by dataset)",
                "val_cap_per_dataset": VAL_CAP,
                "seeds_of_val_subsets": list(SEEDS),
            },
            "protocol": {
                "ridge_log10_grid": RIDGE_GRID,
                "inner_split": f"{INNER_FRAC:.0%} of training units, rng seed {INNER_SEED}",
                "knn_k": list(KS),
                "hgb": {**HGB_PARAMS, "n_principal_components": HGB_PCS},
                "aggregation": "each baseline scored on each seed's validation subset; "
                "the value is the mean over the three subsets (identical subsets except "
                "Vaishnav)",
            },
            "not_scored": NOT_SCORED,
        }
    )
    payload.setdefault("datasets", {})
    for d in ORDER:
        if only and d not in only:
            continue
        t0 = time.time()
        print(f"=== {d}", flush=True)
        if d == "doubles":
            res = run_doubles(inp, n_jobs)
        elif d == "vaishnav":
            res = run_vaishnav(inp, n_jobs)
        else:
            res = run_gene_dataset(inp, d, n_jobs)
        res["seconds"] = time.time() - t0
        payload["datasets"][d] = res
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=1, allow_nan=True)
        print(f"=== {d} done in {res['seconds']:.0f} s", flush=True)


# ============================================================================ trained
def metric_key(d: str) -> str:
    """The trainer's primary validation key of each table row."""
    if d == "doubles":
        return "val/doubles/pearson_per_strain_centered_median"
    if d == "vaishnav":
        return "val/vaishnav/pearson"
    return f"val/{d}/pearson_per_gene_median"


def stage_trained(out_path: str) -> None:
    """Mean over epochs 41-50 of each run's unsampled history, then over the 3 seeds."""
    import wandb

    api = wandb.Api(timeout=120)
    keys = [metric_key(d) for d in ORDER]
    keys += [
        f"val/{d}/pearson_per_strain_centered_median"
        for d in ORDER
        if d not in ("doubles", "vaishnav")
    ]
    keys += [
        "val/doubles/additive_measured_pearson_per_strain_centered_median",
        "val/doubles/additive_model_pearson_per_strain_centered_median",
    ]
    per_run: dict[str, dict[str, Any]] = {}
    lo, hi = TRAINED_WINDOW
    for rid in TRAINED_RUNS:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{rid}")
        rows = [
            r for r in run.scan_history(page_size=5000) if r.get("epoch") is not None
        ]
        win = [r for r in rows if lo <= r["epoch"] <= hi]
        means = {}
        for k in keys:
            vals = [float(r[k]) for r in win if r.get(k) is not None]
            means[k] = {
                "mean": float(np.mean(vals)) if vals else None,
                "n_evaluations": len(vals),
            }
        per_run[rid] = {
            "seed": int(run.config["seed"]),
            "state": run.state,
            "epochs_reached": int(max(r["epoch"] for r in rows)),
            "url": f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{rid}",
            "means_41_50": means,
        }
    summary = {}
    for k in keys:
        vals = [per_run[r]["means_41_50"][k]["mean"] for r in TRAINED_RUNS]
        ok = [x for x in vals if x is not None]
        summary[k] = {
            "mean": float(np.mean(ok)) if len(ok) == 3 else None,
            "sd": float(np.std(ok, ddof=1)) if len(ok) == 3 else None,
            "per_seed": vals,
        }
    with open(out_path) as f:
        payload = json.load(f)
    payload["trained"] = {
        "runs": per_run,
        "window_epochs": list(TRAINED_WINDOW),
        "statistic": "mean over the 10 evaluations of epochs 41-50 of each seed, then the "
        "mean and sample sd (ddof 1) over seeds 0, 1, 2",
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, allow_nan=True)
    for k in keys[: len(ORDER)]:
        print(k, summary[k])


# ============================================================================ report
BASELINE_ORDER = (
    "additive_measured",
    "env_mean",
    "ridge",
    "knn_5",
    "knn_20",
    "knn_50",
    "hgb",
)
BASELINE_LABEL = {
    "constant": "per-gene training mean",
    "additive_measured": "additive measured singles",
    "env_mean": "per-environment mean",
    "ridge": "ridge",
    "knn_5": "kNN, k = 5",
    "knn_20": "kNN, k = 20",
    "knn_50": "kNN, k = 50",
    "hgb": "gradient-boosted trees",
}


def baseline_value(d: str, b: dict[str, Any]) -> float:
    if d == "doubles":
        return float(b["pearson_per_strain_centered_median"])
    if d == "vaishnav":
        return float(b["pearson"])
    return float(b["pearson_per_gene_median"])


def stage_report(out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    from torchcell.timestamp import timestamp
    from torchcell.utils.utils import (
        PANEL_WIDTHS_MM,
        PLOT_PALETTE,
        mm_to_in,
        savefig_true_size_svg,
    )

    with open(out_path) as f:
        payload = json.load(f)
    tr = payload["trained"]["summary"]
    ds = payload["datasets"]
    # ---- table
    hdr = (
        "| dataset | statistic | validation records, n | trained, mean (sd) of 3 seeds | "
        + " | ".join(f"{BASELINE_LABEL[b]}" for b in ("constant", *BASELINE_ORDER))
        + " | best baseline | trained minus best, in seed sd |"
    )
    lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    rows_out = []
    for d in ORDER:
        if d not in ds:
            continue
        t = tr[metric_key(d)]
        bl = ds[d]["baselines"]
        vals = {b: baseline_value(d, bl[b]) for b in bl}
        finite = {b: x for b, x in vals.items() if np.isfinite(x)}
        best = max(finite, key=lambda b: finite[b])
        gap = (t["mean"] - finite[best]) / t["sd"]
        n = ds[d].get("n_doubles") or ds[d].get("n_val_records_union")
        if d == "vaishnav":
            n = f"2,000 per seed ({n:,} across the 3 subsets)"
        stat = {
            "doubles": "median centered per-double Pearson r",
            "vaishnav": "Pearson r over promoters",
        }.get(d, "median per-gene Pearson r")
        cells = []
        for b in ("constant", *BASELINE_ORDER):
            if b not in vals:
                cells.append("--")
            elif not np.isfinite(vals[b]):
                cells.append("undefined")
            else:
                cells.append(f"{vals[b]:.3f}")
        lines.append(
            f"| {LABEL[d]} | {stat} | {n} | {t['mean']:.3f} ({t['sd']:.3f}) | "
            + " | ".join(cells)
            + f" | {BASELINE_LABEL[best]} {finite[best]:.3f} | {gap:+.1f} |"
        )
        rows_out.append((d, t, vals, best, gap))
    table = "\n".join(lines)
    print(table)
    payload["report_table_markdown"] = table
    # ---- figure
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    series = ["trained", *BASELINE_ORDER]
    color = {s: PLOT_PALETTE[i] for i, s in enumerate(series)}
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(70)))
    width = 0.8 / len(series)
    for gi, (d, t, vals, _, _) in enumerate(rows_out):
        for si, s in enumerate(series):
            x = gi - 0.4 + width * (si + 0.5)
            if s == "trained":
                ax.bar(
                    x,
                    t["mean"],
                    width,
                    color=color[s],
                    edgecolor="black",
                    linewidth=0.3,
                    yerr=t["sd"],
                    error_kw={"elinewidth": 0.5, "capsize": 1.2, "capthick": 0.5},
                    zorder=3,
                )
            elif s in vals and np.isfinite(vals[s]):
                ax.bar(
                    x,
                    vals[s],
                    width,
                    color=color[s],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=3,
                )
    ax.set_xticks(range(len(rows_out)))
    ax.set_xticklabels(
        [LABEL[d].replace(" 20", "\n20") for d, *_ in rows_out], rotation=0
    )
    shown = [x for r in rows_out for x in r[2].values() if np.isfinite(x)]
    shown += [r[1]["mean"] - r[1]["sd"] for r in rows_out]
    ax.set_ylim(min(-0.2, float(np.floor(min(shown) * 10) / 10)), 1.0)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="#DDDDDD", zorder=0)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
    ax.set_ylabel(
        "validation Pearson r (per-gene median; Vaishnav over promoters;\n"
        "doubles: centered per-double median)"
    )
    ax.set_xlim(-0.6, len(rows_out) - 0.4)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=color[s], edgecolor="black", linewidth=0.3
        )
        for s in series
    ]
    labels = ["trained, recipe v2, mean ± sd of 3 seeds (epochs 41-50)"] + [
        BASELINE_LABEL[s] for s in BASELINE_ORDER
    ]
    ax.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        handlelength=1.0,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.12)
    d_out = osp.join(os.environ["ASSET_IMAGES_DIR"], EXPERIMENT)
    os.makedirs(d_out, exist_ok=True)
    base = osp.join(d_out, f"baselines_043_{timestamp()}")
    fig.savefig(base + ".png", dpi=300)
    savefig_true_size_svg(fig, base + ".svg")
    plt.close(fig)
    payload["figure"] = base + ".svg"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=1, allow_nan=True)
    print(base + ".svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=["fit", "trained", "report"])
    parser.add_argument("--only", nargs="*", default=None, choices=list(ORDER))
    parser.add_argument(
        "--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    )
    args = parser.parse_args()
    load_dotenv()
    exp_dir = osp.join(os.environ["EXPERIMENT_ROOT"], EXPERIMENT)
    out_path = osp.join(exp_dir, "results", "baselines_043.json")
    if args.stage == "fit":
        inp = Inputs(os.environ["DATA_ROOT"], exp_dir)
        stage_fit(inp, out_path, args.n_jobs, args.only)
    elif args.stage == "trained":
        stage_trained(out_path)
    else:
        stage_report(out_path)


if __name__ == "__main__":
    main()
