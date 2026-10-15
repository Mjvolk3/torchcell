# experiments/030-solid-growth-multi/scripts/lower_order_baselines_030.py
# [[experiments.030-solid-growth-multi.scripts.lower_order_baselines_030]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/030-solid-growth-multi/scripts/lower_order_baselines_030
"""Baselines that see what the joint model sees: the lower-order measurements of a triple.

The published trigenic score is tau_ijk = f_ijk - f_ij f_k - eps_ik - eps_jk. On the 030
S3 pool every term but f_ijk of a held-out triple is a TRAINING record: its three
singles, its three doubles (including the double-mutant query strain's own fitness) and
the three digenic scores. A model trained jointly on fitness and interaction over that
pool therefore knows the triple's null expectation, and the gene-identity ladder of
experiments 010 and 025 (one-hot ridge over the three genes, trained on triples alone)
cannot say how much of its score is that arithmetic. These baselines can. None of them
touches the held-out triple's own fitness or interaction.

Rungs, every one fit on the arm's training triples and scored on its pinned val and test:

    N0  screen mean              tau_hat = mean of training tau in the triple's screen
    N1  multiplicative, query    tau_hat = -(eps_ik + eps_jk)         [f_ijk = f_ij f_k]
    N2  multiplicative, singles  tau_hat = f_i f_j f_k - f_ij f_k - eps_ik - eps_jk
    N1s N1 with the control terms matched on the triple's own array strain
    B1  additive gene ridge      the 010 rung, one-hot over the three genes (reference)
    L1  ridge, lower-order       ridge over the lower-order terms, products, screen
    L2  boosted trees, lower-order  HistGradientBoosting over the same features
    L3  ridge, genes + lower-order  B1's one-hot beside L1's features
    F1  ridge, lower-order -> f_ijk  the fitness side: how much of f_ijk is arithmetic

N1 and N2 are zero-parameter predictors. Where a term is missing the arithmetic rungs
fall back to N0 for that row so every rung is scored on the identical rows; the fraction
of rows the term covered is reported beside the score. Ridge alpha and the tree grid
select on val; test is reported.

Terms are chosen under the same ``LabelPolicy`` (``kuzmin-first``, promoted to the
triple's own year) that the trainer's eval table uses, with the double-mutant query
strain's fitness preferred for f_ij (``select_double``) and the strain-matched control
terms resolved exactly as ``closure_recompute_030.py`` resolves them. The target tau and
the screen token of every pinned val/test triple are asserted equal to
``results/pinned_eval_table_030.parquet``.

Arm Q (query-pair disjoint) withholds every double whose pair is a held-out query pair
from the training pool (``subset_definitions_030_q.py``), so here the features of any
held-out pair are masked to missing on Q: a baseline on Q never sees f_ij or eps_ij of a
held-out triple's query pair, as the model never did. Arm R masks nothing. The masked
counts are recorded.

Inputs (read-only):
    $DATA_ROOT/.../030-solid-growth-multi/closure/entries.parquet, triple_roles.parquet
    results/pinned_splits_from_010_seed_42.json.gz          arm R, key ``pinned``
    results/query_pair_disjoint_splits_030.json.gz          arm Q, ``pinned`` + ``pair_assignment``
    results/pinned_eval_table_030.parquet                   the trainer's val/test targets

Writes (results/):
    lower_order_features_030.parquet            one row per triple, every term and target
    lower_order_baselines_030.csv               the ladder, one row per (arm, rung, split)
    lower_order_baselines_030_pred_<arm>.parquet  idx, split, screen, y, yhat per rung
    lower_order_baselines_030_paired.csv        paired bootstrap on test, 2,000 resamples
    lower_order_baselines_030_summary.json      LadderSummary (pydantic)
    t11-lower-order-baselines-030.tex           the document table
    (ASSET_IMAGES_DIR/030-solid-growth-multi/) lower_order_baselines_030_ladder_<ts>.{svg,png}

The transformer rows come from W&B (``--wandb``): the rank-0 runs tagged with each arm's
config, validation interaction Pearson averaged over the last 10 logged epochs and its
max over epochs. They are validation numbers of a model with no test evaluation yet and
are drawn as such.

    PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/lower_order_baselines_030.py --stage all
    sbatch experiments/030-solid-growth-multi/scripts/gh_lower_order_baselines_030.slurm
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import os.path as osp
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from pydantic import BaseModel, Field
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from torchcell.data.label_policy import (
    LabelEntry,
    LabelPolicy,
    source_key,
    strain_token,
)
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    apply_paper_style,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXP = "030-solid-growth-multi"
CLOSURE = osp.join(DATA_ROOT, "data/torchcell/experiments", EXP, "closure")
RESULTS = osp.join(EXPERIMENT_ROOT, EXP, "results")
IMAGES = osp.join(ASSET_IMAGES_DIR, EXP)
FEATURES = osp.join(RESULTS, "lower_order_features_030.parquet")
B010 = osp.join(
    EXPERIMENT_ROOT, "010-kuzmin-tmi", "scripts", "additive_baseline_gene_interaction.py"
)

POLICY_NAME = "kuzmin-first"
SCREENS = {"TmiKuzmin2018Dataset": "kuzmin2018", "TmiKuzmin2020Dataset": "kuzmin2020"}
LABEL_OF_EXP_TYPE = {"fitness": "fitness", "gene interaction": "gene_interaction"}

ARMS = {
    "R": ("pinned_splits_from_010_seed_42.json.gz", "random over records (010)"),
    "Q": ("query_pair_disjoint_splits_030.json.gz", "query-pair disjoint"),
}
ALPHA_GRID = [1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
HGB_GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.1},
    {"max_leaf_nodes": 31, "learning_rate": 0.1},
    {"max_leaf_nodes": 63, "learning_rate": 0.05},
    {"max_leaf_nodes": 127, "learning_rate": 0.05},
]
HGB_ITER = 600
N_BOOT = 2000
SEED = 0

TERMS = [
    "f_i",
    "f_j",
    "f_k",
    "f_ij_q",
    "f_ij_arr",
    "f_ik",
    "f_jk",
    "eps_ij",
    "eps_ik",
    "eps_jk",
    "eps_ik_s",
    "eps_jk_s",
    "f_k_s",
]
PRODUCTS = {
    "f_ij_q_f_k": ("f_ij_q", "f_k"),
    "f_i_f_j_f_k": ("f_i", "f_j", "f_k"),
    "eps_ij_f_k": ("eps_ij", "f_k"),
    "eps_ik_plus_eps_jk": ("eps_ik", "eps_jk"),
}
RUNGS = {
    "N0_screen_mean": "N0 screen mean",
    "N1_mult_query_double": "N1 mult., query double",
    "N2_mult_singles": "N2 mult., singles",
    "N1s_mult_strain_matched": "N1s strain-matched",
    "B1_additive_gene": "B1 additive gene ridge",
    "L1_ridge_lower_order": "L1 ridge, lower-order",
    "L2_hgb_lower_order": "L2 boosted trees, lower-order",
    "L3_ridge_gene_plus_lower_order": "L3 ridge, genes + lower-order",
}
TICKS = {
    "N0_screen_mean": "N0 screen\nmean",
    "N1_mult_query_double": "N1 digenic\ncontrols",
    "N2_mult_singles": "N2 singles\nproduct",
    "N1s_mult_strain_matched": "N1s strain-\nmatched",
    "B1_additive_gene": "B1 additive\nridge",
    "L1_ridge_lower_order": "L1 ridge\nlower-order",
    "L2_hgb_lower_order": "L2 trees\nlower-order",
    "L3_ridge_gene_plus_lower_order": "L3 ridge\ngenes + terms",
}
COLOR = {"R": PLOT_PALETTE[0], "Q": PLOT_PALETTE[1]}
FILL = {"R": PLOT_PALETTE_FILL[0], "Q": PLOT_PALETTE_FILL[1]}
BAR = {"edgecolor": "black", "linewidth": 0.5}
PAIRED = [
    ("B1_additive_gene", "L1_ridge_lower_order"),
    ("B1_additive_gene", "L2_hgb_lower_order"),
    ("B1_additive_gene", "L3_ridge_gene_plus_lower_order"),
    ("N1_mult_query_double", "L2_hgb_lower_order"),
    ("L1_ridge_lower_order", "L2_hgb_lower_order"),
    ("L2_hgb_lower_order", "L3_ridge_gene_plus_lower_order"),
]
from wandb_seed_histories import (  # noqa: E402  (sibling module, scripts dir)
    WANDB_ARM_TAG,
    WANDB_HEADLINE_SEED,
    WANDB_WINDOW,
    wandb_seed_histories,
)
LIMIT: int | None = None


# ----------------------------------------------------------------------------- models
class TermCoverage(BaseModel):
    n_triples: int
    finite_by_term: dict[str, int]
    masked_by_term: dict[str, int] = Field(
        description="arm Q only: rows whose term was set to missing because its pair is held out"
    )


class RungRow(BaseModel):
    arm: str
    rung: str
    split: str
    n: int
    pearson: float
    spearman: float
    rmse: float
    pearson_within_screen: float
    pearson_kuzmin2018: float
    pearson_kuzmin2020: float
    coverage: float = Field(description="fraction of rows the rung's own term covered")
    selection: str | None = None


class PairedRow(BaseModel):
    arm: str
    a: str
    b: str
    n: int
    gain: float = Field(description="r(y, b) - r(y, a) on test")
    lo: float
    hi: float


class WandbReference(BaseModel):
    """One seed of one arm's transformer, stitched over its rank-0 runs by epoch."""

    arm: str
    config_tag: str
    run_ids: list[str]
    seed: int
    n_epochs_logged: int
    last_epoch: int
    val_pearson_last_window_mean: float
    val_pearson_max: float
    val_pearson_max_epoch: int
    window: int
    partial: bool = Field(
        description="fewer epochs logged than the window; recorded, never drawn"
    )


class LadderSummary(BaseModel):
    policy_name: str
    policy_id: str
    features_path: str
    entries_path: str
    coverage: dict[str, TermCoverage]
    screen_floor: dict[str, dict[str, float]] = Field(
        description="per arm, the N0 screen-mean Pearson on val and test"
    )
    test_pearson: dict[str, dict[str, float]]
    paired: list[PairedRow]
    wandb: list[WandbReference]
    hgb_selected: dict[str, dict[str, Any]]
    alpha_selected: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------- helpers
def _b010():
    spec = importlib.util.spec_from_file_location("b010", B010)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_gz(name: str) -> dict[str, Any]:
    with gzip.open(osp.join(RESULTS, name), "rt") as f:
        return json.load(f)


def _finite(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _label_entries(
    frame: pd.DataFrame, label: str
) -> tuple[dict[str, list[LabelEntry]], dict[str, list[set[str]]]]:
    """``LabelEntry`` pools keyed by gene set, with each entry's strain identifiers."""
    want = "fitness" if label == "fitness" else "gene interaction"
    sub = frame[frame["exp_type"] == want]
    out: dict[str, list[LabelEntry]] = {}
    ids: dict[str, list[set[str]]] = {}
    for genes, src, v, sd, n, p, sid, sids in zip(
        sub["genes"],
        sub["source"],
        sub["value"],
        sub["sd"],
        sub["n_samples"],
        sub["p"],
        sub["strain_id"],
        sub["strain_ids"],
    ):
        value = _finite(v)
        if value is None:
            continue
        out.setdefault(str(genes), []).append(
            LabelEntry(
                source=src,
                label=label,
                value=value,
                sd=_finite(sd),
                n_samples=None if _finite(n) is None else int(n),
                p_value=_finite(p),
                strain_id=None if _finite_str(sid) is None else str(sid),
            )
        )
        ids.setdefault(str(genes), []).append(set(str(sids).split("|")) - {""})
    return out, ids


def _finite_str(s: Any) -> str | None:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    return str(s)


def _array_matched(
    pools: dict[str, list[LabelEntry]],
    ids: dict[str, list[set[str]]],
    key: str,
    array_strain_id: str,
    year: str,
    policy: LabelPolicy,
    label: str,
) -> float:
    """The value of ``key`` measured on the triple's own array strain, else NaN."""
    pool = pools.get(key)
    if not pool:
        return math.nan
    on_strain = [(e, s) for e, s in zip(pool, ids[key]) if array_strain_id in s]
    same_year = [e for e, _ in on_strain if e.source == year]
    if not same_year:
        same_year = [e for e, _ in on_strain]
    if not same_year:
        return math.nan
    choice = policy.select(same_year, label)
    return math.nan if choice is None else choice.value


def _chosen(
    pools: dict[str, list[LabelEntry]], policy: LabelPolicy, label: str
) -> pd.Series:
    values = {
        g: c.value
        for g, pool in pools.items()
        if (c := policy.select(pool, label)) is not None
    }
    return pd.Series(values, dtype=float)


def _arr(series: pd.Series, keys: Any) -> np.ndarray:
    return series.reindex(keys).to_numpy(dtype=float)


def _pair(a: pd.Series, b: pd.Series) -> list[str]:
    return ["|".join(sorted((x, y))) for x, y in zip(a, b)]


# --------------------------------------------------------------------------- features
def build_features() -> None:
    """One row per triple: targets under the policy, every lower-order term, roles."""
    t0 = time.time()
    entries = pd.read_parquet(osp.join(CLOSURE, "entries.parquet"))
    entries["source"] = [
        source_key(d, t) for d, t in zip(entries["dataset"], entries["temp"])
    ]
    roles = pd.read_parquet(osp.join(CLOSURE, "triple_roles.parquet"))
    if LIMIT is not None:
        keep = np.sort(roles["idx"].to_numpy())[:LIMIT]
        roles = roles[roles["idx"].isin(keep)]
        entries = entries[(entries["order"] < 3) | entries["idx"].isin(keep)]
        print(f"SMOKE: limited to {len(roles):,} triples")
    print(f"entries {len(entries):,} roles {len(roles):,} ({time.time() - t0:.0f}s)")

    singles = entries[entries["order"] == 1]
    doubles = entries[entries["order"] == 2]
    triples = entries[entries["order"] == 3]
    double_fit, _ = _label_entries(doubles, "fitness")
    double_gi, double_gi_ids = _label_entries(doubles, "gene_interaction")
    single_fit, single_fit_ids = _label_entries(singles, "fitness")
    double_fit_array = {
        g: [e for e in pool if strain_token(e.strain_id) is None]
        for g, pool in double_fit.items()
    }
    triple_pools: dict[int, list[LabelEntry]] = {}
    for idx, src, et, v, sd, n, p, sid in zip(
        triples["idx"],
        triples["source"],
        triples["exp_type"],
        triples["value"],
        triples["sd"],
        triples["n_samples"],
        triples["p"],
        triples["strain_id"],
    ):
        value = _finite(v)
        if value is None:
            continue
        triple_pools.setdefault(int(idx), []).append(
            LabelEntry(
                source=src,
                label=LABEL_OF_EXP_TYPE[str(et)],
                value=value,
                sd=_finite(sd),
                n_samples=None if _finite(n) is None else int(n),
                p_value=_finite(p),
                strain_id=_finite_str(sid),
            )
        )
    print(f"pools built ({time.time() - t0:.0f}s)")

    # targets: the trainer's policy, exactly as make_pinned_eval_table_030 applies it
    base = LabelPolicy(name=POLICY_NAME)
    tgt_rows = []
    for idx, pool in triple_pools.items():
        gi = base.select(pool, "gene_interaction")
        fit = base.select(pool, "fitness")
        assert gi is not None and fit is not None, idx
        tgt_rows.append(
            {
                "idx": idx,
                "tau": gi.value,
                "screen": gi.source,
                "f_ijk": fit.value,
                "f_ijk_source": fit.source,
            }
        )
    targets = pd.DataFrame(tgt_rows).set_index("idx")
    assert set(targets["screen"]) <= set(SCREENS.values()), targets["screen"].unique()
    genes_of = triples.drop_duplicates("idx").set_index("idx")["genes"]
    table = roles.set_index("idx").join(targets, how="inner")
    table["genes"] = genes_of.reindex(table.index).to_numpy()
    assert len(table) == len(roles) == len(targets), (len(table), len(roles))

    # terms under the policy promoted to the triple's own screen
    cols = {t: np.full(len(table), np.nan) for t in TERMS}
    for screen in SCREENS.values():
        policy = LabelPolicy(name=f"{POLICY_NAME}::{screen}").promoted_for_year(screen)
        sel = (table["screen"] == screen).to_numpy()
        sub = table[sel]
        print(f"{screen}: {len(sub):,} triples, policy {policy.policy_id}", flush=True)
        f_single = _chosen(single_fit, policy, "fitness")
        eps_double = _chosen(double_gi, policy, "gene_interaction")
        f_double_arr = _chosen(double_fit_array, policy, "fitness")
        qpair = _pair(sub["qi"], sub["qj"])
        ik = _pair(sub["qi"], sub["array_gene"])
        jk = _pair(sub["qj"], sub["array_gene"])
        f_ij_q = np.full(len(sub), np.nan)
        for n, (pk, sid) in enumerate(zip(qpair, sub["query_strain_id"])):
            pool = double_fit.get(pk)
            if not pool:
                continue
            choice = policy.select_double(pool, sid, "fitness")
            if choice is not None:
                f_ij_q[n] = choice.value
        eps_ik_s = np.full(len(sub), np.nan)
        eps_jk_s = np.full(len(sub), np.nan)
        f_k_s = np.full(len(sub), np.nan)
        for n, (ik_key, jk_key, k_gene, asid) in enumerate(
            zip(ik, jk, sub["array_gene"], sub["array_strain_id"])
        ):
            eps_ik_s[n] = _array_matched(
                double_gi, double_gi_ids, ik_key, asid, screen, policy, "gene_interaction"
            )
            eps_jk_s[n] = _array_matched(
                double_gi, double_gi_ids, jk_key, asid, screen, policy, "gene_interaction"
            )
            f_k_s[n] = _array_matched(
                single_fit, single_fit_ids, k_gene, asid, screen, policy, "fitness"
            )
        values = {
            "f_i": _arr(f_single, sub["qi"]),
            "f_j": _arr(f_single, sub["qj"]),
            "f_k": _arr(f_single, sub["array_gene"]),
            "f_ij_q": f_ij_q,
            "f_ij_arr": _arr(f_double_arr, qpair),
            "f_ik": _arr(f_double_arr, ik),
            "f_jk": _arr(f_double_arr, jk),
            "eps_ij": _arr(eps_double, qpair),
            "eps_ik": _arr(eps_double, ik),
            "eps_jk": _arr(eps_double, jk),
            "eps_ik_s": eps_ik_s,
            "eps_jk_s": eps_jk_s,
            "f_k_s": f_k_s,
        }
        for t, v in values.items():
            cols[t][sel] = v
        print(f"  terms resolved ({time.time() - t0:.0f}s)", flush=True)
    for t, v in cols.items():
        table[t] = v
    table["pair_ij"] = _pair(table["qi"], table["qj"])
    table["pair_ik"] = _pair(table["qi"], table["array_gene"])
    table["pair_jk"] = _pair(table["qj"], table["array_gene"])
    table = table.reset_index()

    # the trainer's val/test targets must be these targets
    ev = pd.read_parquet(osp.join(RESULTS, "pinned_eval_table_030.parquet"))
    ev = ev[ev["idx"].isin(table["idx"])]
    gi = ev[ev["label"] == "gene_interaction"].set_index("idx")
    fit = ev[ev["label"] == "fitness"].set_index("idx")
    t_idx = table.set_index("idx")
    assert np.allclose(t_idx.loc[gi.index, "tau"], gi["value"]), "tau != eval table"
    assert (t_idx.loc[gi.index, "screen"] == gi["source"]).all(), "screen != eval table"
    assert np.allclose(t_idx.loc[fit.index, "f_ijk"], fit["value"]), "f_ijk != eval"
    print(f"eval table agreement asserted on {len(gi):,} pinned triples")

    table.to_parquet(FEATURES, index=False)
    for t in TERMS:
        print(f"  {t:<9} finite {np.isfinite(table[t]).mean():.4f}")
    print(f"wrote {FEATURES} ({len(table):,} rows, {time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------- fit
def _score(y: np.ndarray, yhat: np.ndarray, screen: np.ndarray) -> dict[str, float]:
    assert np.isfinite(yhat).all(), "a prediction is not finite"
    if np.allclose(yhat, yhat[0]):
        pear, spear = 0.0, 0.0
    else:
        pear = float(pearsonr(y, yhat)[0])
        spear = float(spearmanr(y, yhat)[0])
    yc = y.copy()
    hc = yhat.copy()
    per = {}
    for s in SCREENS.values():
        m = screen == s
        yc[m] -= y[m].mean()
        hc[m] -= yhat[m].mean()
        per[s] = (
            0.0 if (m.sum() < 3 or np.allclose(yhat[m], yhat[m][0]))
            else float(pearsonr(y[m], yhat[m])[0])
        )
    within = 0.0 if np.allclose(hc, hc[0]) else float(pearsonr(yc, hc)[0])
    return {
        "pearson": pear,
        "spearman": spear,
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
        "pearson_within_screen": within,
        "pearson_kuzmin2018": per["kuzmin2018"],
        "pearson_kuzmin2020": per["kuzmin2020"],
    }


def _splits(table: pd.DataFrame, arm: str) -> dict[str, np.ndarray]:
    pinned = _load_gz(ARMS[arm][0])["pinned"]
    pos = {int(i): r for r, i in enumerate(table["idx"].to_numpy())}
    out = {}
    for name in ("train", "val", "test"):
        rows = [pos[int(i)] for i in pinned[name] if int(i) in pos]
        assert LIMIT is not None or len(rows) == len(pinned[name]), (
            f"{arm}/{name}: pinned idx outside table"
        )
        out[name] = np.array(sorted(rows), dtype=np.int64)
    n = sum(v.size for v in out.values())
    assert n == len(table), f"{arm}: splits cover {n} of {len(table)} triples"
    return out


def _mask_heldout_pairs(table: pd.DataFrame, arm: str) -> dict[str, int]:
    """Arm Q: a held-out query pair's double never trained, so its terms are missing."""
    masked = {t: 0 for t in TERMS}
    if arm != "Q":
        return masked
    assignment = _load_gz(ARMS["Q"][0])["pair_assignment"]
    held = {
        "|".join(sorted(p.split("+"))) for p, s in assignment.items() if s != "train"
    }
    groups = {
        "pair_ij": ["f_ij_q", "f_ij_arr", "eps_ij"],
        "pair_ik": ["f_ik", "eps_ik", "eps_ik_s"],
        "pair_jk": ["f_jk", "eps_jk", "eps_jk_s"],
    }
    for col, terms in groups.items():
        m = table[col].isin(held).to_numpy()
        for t in terms:
            masked[t] = int((m & np.isfinite(table[t].to_numpy())).sum())
            table.loc[m, t] = np.nan
    return masked


def _raw_terms(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every term and product as float arrays; a missing factor makes the product missing."""
    feats = {t: table[t].to_numpy(dtype=float) for t in TERMS}
    for name, parts in PRODUCTS.items():
        if name == "eps_ik_plus_eps_jk":
            feats[name] = feats["eps_ik"] + feats["eps_jk"]
            continue
        v = np.ones(len(table))
        for p in parts:
            v = v * feats[p]
        feats[name] = v
    return feats


def _design(table: pd.DataFrame, tr: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Dense lower-order design: terms, products, screen; standardized on train,
    missing filled with the train median beside an indicator column."""
    feats = _raw_terms(table)
    names: list[str] = []
    cols: list[np.ndarray] = []
    for name, v in feats.items():
        miss = ~np.isfinite(v)
        med = float(np.nanmedian(v[tr]))
        filled = np.where(miss, med, v)
        mu, sd = filled[tr].mean(), filled[tr].std()
        cols.append((filled - mu) / (sd if sd > 0 else 1.0))
        names.append(name)
        if miss[tr].any() or miss.any():
            cols.append(miss.astype(float))
            names.append(f"{name}_missing")
    cols.append((table["screen"] == "kuzmin2020").to_numpy(dtype=float))
    names.append("screen_kuzmin2020")
    return np.column_stack(cols), names


def _ridge_dense(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    b0 = float(y.mean())
    xtx = x.T @ x + alpha * np.eye(x.shape[1])
    beta = np.linalg.solve(xtx, x.T @ (y - b0))
    return beta, b0


def _select_ridge_dense(
    x: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray
) -> tuple[np.ndarray, float]:
    best = None
    for alpha in ALPHA_GRID:
        beta, b0 = _ridge_dense(x[tr], y[tr], alpha)
        pred = x @ beta + b0
        r = pearsonr(y[va], pred[va])[0]
        if best is None or r > best[0]:
            best = (r, alpha, pred)
    return best[2], best[1]


def _select_ridge_sparse(
    b010: Any, x: sp.csr_matrix, y: np.ndarray, tr: np.ndarray, va: np.ndarray
) -> tuple[np.ndarray, float]:
    best = None
    for alpha in ALPHA_GRID:
        beta, b0 = b010.ridge_fit(x[tr], y[tr], alpha)
        pred = x @ beta + b0
        r = pearsonr(y[va], pred[va])[0]
        if best is None or r > best[0]:
            best = (r, alpha, pred)
    return best[2], best[1]


def _paired_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    n = y.size
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = pearsonr(y[idx], b[idx])[0] - pearsonr(y[idx], a[idx])[0]
    point = float(pearsonr(y, b)[0] - pearsonr(y, a)[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, float(lo), float(hi)


def fit_arm(
    arm: str, b010: Any
) -> tuple[list[RungRow], list[PairedRow], TermCoverage, dict[str, Any], dict[str, float]]:
    table = pd.read_parquet(FEATURES)
    masked = _mask_heldout_pairs(table, arm)
    splits = _splits(table, arm)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    y = table["tau"].to_numpy(dtype=float)
    f_ijk = table["f_ijk"].to_numpy(dtype=float)
    screen = table["screen"].to_numpy()
    print(
        f"\n=== arm {arm} ({ARMS[arm][1]}): train {tr.size:,} val {va.size:,} test {te.size:,}"
    )
    coverage = TermCoverage(
        n_triples=len(table),
        finite_by_term={t: int(np.isfinite(table[t]).sum()) for t in TERMS},
        masked_by_term=masked,
    )
    rows: list[RungRow] = []
    preds: dict[str, np.ndarray] = {}
    cover: dict[str, np.ndarray] = {}
    selection: dict[str, str] = {}
    all_rows = np.ones(len(table), dtype=bool)

    # N0: per-screen training mean, the floor every pooled Pearson carries
    n0 = np.empty(len(table))
    for s in SCREENS.values():
        m = screen == s
        n0[m] = y[tr][screen[tr] == s].mean()
    preds["N0_screen_mean"] = n0
    cover["N0_screen_mean"] = all_rows

    f = {t: table[t].to_numpy(dtype=float) for t in TERMS}
    arithmetic = {
        "N1_mult_query_double": -(f["eps_ik"] + f["eps_jk"]),
        "N2_mult_singles": f["f_i"] * f["f_j"] * f["f_k"]
        - f["f_ij_q"] * f["f_k"]
        - f["eps_ik"]
        - f["eps_jk"],
        "N1s_mult_strain_matched": -(f["eps_ik_s"] + f["eps_jk_s"]),
    }
    for name, v in arithmetic.items():
        fin = np.isfinite(v)
        preds[name] = np.where(fin, v, n0)
        cover[name] = fin

    # B1: the 010 rung, one-hot over the three genes
    genes3 = table[["qi", "qj", "array_gene"]].to_numpy()
    gene_names = sorted(set(genes3.ravel().tolist()))
    col = {g: j for j, g in enumerate(gene_names)}
    row_genes = np.vectorize(col.__getitem__)(genes3).astype(np.int32)
    xg = b010.gene_matrix(row_genes, len(gene_names))
    preds["B1_additive_gene"], alpha = _select_ridge_sparse(b010, xg, y, tr, va)
    cover["B1_additive_gene"] = all_rows
    selection["B1_additive_gene"] = f"alpha {alpha:g}"
    alphas = {"B1_additive_gene": alpha}

    # L1: ridge over the lower-order terms
    xd, names = _design(table, tr)
    print(f"lower-order design: {xd.shape[1]} columns")
    preds["L1_ridge_lower_order"], alpha = _select_ridge_dense(xd, y, tr, va)
    cover["L1_ridge_lower_order"] = all_rows
    selection["L1_ridge_lower_order"] = f"alpha {alpha:g}"
    alphas["L1_ridge_lower_order"] = alpha

    # L2: boosted trees over the raw terms (missing handled natively), grid on val
    raw = _raw_terms(table)
    raw_cols = list(raw)
    xr = np.column_stack(
        [raw[c] for c in raw_cols] + [(screen == "kuzmin2020").astype(float)]
    )
    best = None
    for params in HGB_GRID:
        model = HistGradientBoostingRegressor(
            max_iter=HGB_ITER,
            early_stopping=False,
            random_state=SEED,
            **params,
        )
        model.fit(xr[tr], y[tr])
        pred = model.predict(xr)
        r = pearsonr(y[va], pred[va])[0]
        print(f"  L2 {params} val pearson {r:.4f}", flush=True)
        if best is None or r > best[0]:
            best = (r, params, pred)
    preds["L2_hgb_lower_order"] = best[2]
    cover["L2_hgb_lower_order"] = all_rows
    selection["L2_hgb_lower_order"] = json.dumps(best[1])
    hgb_sel = {"params": best[1], "max_iter": HGB_ITER, "columns": raw_cols + ["screen"]}

    # L3: genes beside the lower-order terms, one alpha for both blocks
    xgd = sp.hstack([xg, sp.csr_matrix(xd)]).tocsr()
    preds["L3_ridge_gene_plus_lower_order"], alpha = _select_ridge_sparse(
        b010, xgd, y, tr, va
    )
    cover["L3_ridge_gene_plus_lower_order"] = all_rows
    selection["L3_ridge_gene_plus_lower_order"] = f"alpha {alpha:g}"
    alphas["L3_ridge_gene_plus_lower_order"] = alpha

    # F1: the fitness side, f_ijk from the same lower-order design
    f1_pred, alpha = _select_ridge_dense(xd, f_ijk, tr, va)
    alphas["F1_ridge_lower_order_fitness"] = alpha

    for rung, pred in preds.items():
        for split, idx in (("train", tr), ("val", va), ("test", te)):
            rows.append(
                RungRow(
                    arm=arm,
                    rung=rung,
                    split=split,
                    n=int(idx.size),
                    coverage=float(cover[rung][idx].mean()),
                    selection=selection.get(rung),
                    **_score(y[idx], pred[idx], screen[idx]),
                )
            )
        print(
            f"  {rung:<32s} test pearson {rows[-1].pearson:.4f} "
            f"coverage {rows[-1].coverage:.3f}"
        )
    for split, idx in (("train", tr), ("val", va), ("test", te)):
        rows.append(
            RungRow(
                arm=arm,
                rung="F1_ridge_lower_order_fitness",
                split=split,
                n=int(idx.size),
                coverage=1.0,
                selection=f"alpha {alphas['F1_ridge_lower_order_fitness']:g}",
                **_score(f_ijk[idx], f1_pred[idx], screen[idx]),
            )
        )
    print(f"  {'F1 (f_ijk target)':<32s} test pearson {rows[-1].pearson:.4f}")

    rng = np.random.default_rng(SEED)
    paired = [
        PairedRow(arm=arm, a=a, b=b, n=int(te.size), gain=g, lo=lo, hi=hi)
        for a, b in PAIRED
        for g, lo, hi in [_paired_bootstrap(y[te], preds[a][te], preds[b][te], rng)]
    ]
    for p in paired:
        print(f"  paired {p.b} - {p.a}: {p.gain:+.4f} [{p.lo:+.4f}, {p.hi:+.4f}]")

    split_col = np.empty(len(table), dtype=object)
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        split_col[idx] = name
    out = pd.DataFrame(
        {
            "idx": table["idx"].to_numpy(),
            "split": split_col,
            "screen": screen,
            "tau": y,
            "f_ijk": f_ijk,
            "F1_ridge_lower_order_fitness": f1_pred,
            **preds,
        }
    )
    out.to_parquet(
        osp.join(RESULTS, f"lower_order_baselines_030_pred_{arm}.parquet"), index=False
    )
    return rows, paired, coverage, hgb_sel, alphas


# ----------------------------------------------------------------------------- wandb
def wandb_references(arm: str) -> list[WandbReference]:
    """The arm's transformer, one reference per seed; the headline seed must be present."""
    tag = WANDB_ARM_TAG[arm]
    hists = wandb_seed_histories(tag, "val/gene_interaction/Pearson")
    assert WANDB_HEADLINE_SEED in hists, f"{tag}: headline seed {WANDB_HEADLINE_SEED} absent"
    refs = []
    for seed, hist in hists.items():
        last = hist.iloc[-WANDB_WINDOW:]
        refs.append(
            WandbReference(
                arm=arm,
                config_tag=tag,
                run_ids=hist.attrs["run_ids"],
                seed=seed,
                n_epochs_logged=int(len(hist)),
                last_epoch=int(hist.index[-1]),
                val_pearson_last_window_mean=float(last.mean()),
                val_pearson_max=float(hist.max()),
                val_pearson_max_epoch=int(hist.idxmax()),
                window=WANDB_WINDOW,
                partial=bool(len(hist) < WANDB_WINDOW),
            )
        )
        print(
            f"  wandb {tag} seed {seed}: {len(hist)} epochs to {int(hist.index[-1])}, "
            f"last-window mean {float(last.mean()):.4f}"
            + (" (partial, not drawn)" if len(hist) < WANDB_WINDOW else "")
        )
    return refs


# ------------------------------------------------------------------------------ table
def write_tex(ladder: pd.DataFrame, paired: pd.DataFrame, path: str) -> None:
    lines = [
        "%% SOURCE: experiments/030-solid-growth-multi/scripts/lower_order_baselines_030.py",
        "%% results/lower_order_baselines_030.csv; regenerate, do not edit",
        r"\begin{tabular}{@{}l l rr rr r@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{pooled $r$} & \multicolumn{2}{c}{within-screen $r$} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"arm & rung & val & test & val & test & coverage \\",
        r"\midrule",
    ]
    for arm in ARMS:
        sub = ladder[ladder["arm"] == arm]
        for rung, label in RUNGS.items():
            v = sub[(sub["rung"] == rung) & (sub["split"] == "val")].iloc[0]
            t = sub[(sub["rung"] == rung) & (sub["split"] == "test")].iloc[0]
            lines.append(
                f"{arm} & {label} & {v['pearson']:.3f} & {t['pearson']:.3f} & "
                f"{v['pearson_within_screen']:.3f} & {t['pearson_within_screen']:.3f} & "
                f"{t['coverage']:.2f} \\\\"
            )
        if arm == "R":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- figure
def style_metric_axis(ax, top: float) -> None:
    """The 025 panels' metric axis: tenth gridlines, labels every 0.2, boxed."""
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", which="both", linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)


def plot_ladder(
    ladder: pd.DataFrame, refs: list[WandbReference], stem: str
) -> None:
    """The 025 ladder look: one test bar per rung in the arm's light fill, the
    transformer's validation mean in the arm's darker line color, legend above the axes."""
    apply_paper_style()
    rungs = list(RUNGS)
    fig, axes = plt.subplots(
        1, 2, figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(62.0))
    )
    for ax, arm, letter in zip(axes, ARMS, "ab"):
        color, fill = COLOR[arm], FILL[arm]
        sub = ladder[ladder["arm"] == arm].set_index(["rung", "split"])["pearson"]
        # one bar per seed that has logged a full window; a partial seed is in the
        # summary but not on the axes
        ref = [r for r in refs if r.arm == arm and not r.partial]
        cats = [TICKS[r] for r in rungs]
        vals = [sub.loc[(r, "test")] for r in rungs]
        x = np.arange(len(cats) + len(ref))
        ax.bar(x[: len(rungs)], vals, 0.7, color=fill, **BAR)
        for i, r in enumerate(ref):
            seed = f", seed {r.seed}" if len(ref) > 1 else ""
            cats.append(f"CGT 030{seed}\n(val, ep {r.last_epoch - r.window + 1}-{r.last_epoch})")
            ax.bar(x[len(rungs) + i], r.val_pearson_last_window_mean, 0.7, color=color, **BAR)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("Held-out Pearson r")
        style_metric_axis(ax, 0.8)
        # Nothing but data inside the axes: the legend sits above the panel and names the arm.
        handles = [
            Patch(facecolor=fill, label=f"arm {arm}, {ARMS[arm][1]}, test score", **BAR),
            Patch(facecolor=color, label="validation, mean of last synced epochs", **BAR),
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=2,
            handlelength=1.4,
            borderpad=0.4,
            columnspacing=1.0,
        )
        panel_label(ax, letter)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(stem + ".png", dpi=300)
    savefig_true_size_svg(fig, stem + ".svg")
    print(f"wrote {stem}.svg")


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", choices=["features", "fit", "wandb", "plot", "all"], default="all"
    )
    ap.add_argument("--wandb", action="store_true", help="fetch the transformer rows")
    ap.add_argument(
        "--limit", type=int, default=None, help="SMOKE: first N triples only"
    )
    args = ap.parse_args()
    global LIMIT
    LIMIT = args.limit
    os.makedirs(IMAGES, exist_ok=True)

    if args.stage in ("features", "all"):
        build_features()
    if args.stage in ("fit", "all"):
        b010 = _b010()
        rows: list[RungRow] = []
        paired: list[PairedRow] = []
        coverage: dict[str, TermCoverage] = {}
        hgb: dict[str, Any] = {}
        alphas: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            r, p, c, h, a = fit_arm(arm, b010)
            rows += r
            paired += p
            coverage[arm] = c
            hgb[arm] = h
            alphas[arm] = a
        ladder = pd.DataFrame([r.model_dump() for r in rows])
        ladder.to_csv(osp.join(RESULTS, "lower_order_baselines_030.csv"), index=False)
        pd.DataFrame([p.model_dump() for p in paired]).to_csv(
            osp.join(RESULTS, "lower_order_baselines_030_paired.csv"), index=False
        )
        refs = [r for arm in ARMS for r in wandb_references(arm)] if args.wandb else []
        test = ladder[ladder["split"] == "test"]
        summary = LadderSummary(
            policy_name=POLICY_NAME,
            policy_id=LabelPolicy(name=POLICY_NAME).policy_id,
            features_path=FEATURES,
            entries_path=osp.join(CLOSURE, "entries.parquet"),
            coverage=coverage,
            screen_floor={
                arm: {
                    s: float(
                        ladder[
                            (ladder["arm"] == arm)
                            & (ladder["rung"] == "N0_screen_mean")
                            & (ladder["split"] == s)
                        ]["pearson"].iloc[0]
                    )
                    for s in ("val", "test")
                }
                for arm in ARMS
            },
            test_pearson={
                arm: {
                    r["rung"]: float(r["pearson"])
                    for _, r in test[test["arm"] == arm].iterrows()
                }
                for arm in ARMS
            },
            paired=paired,
            wandb=refs,
            hgb_selected=hgb,
            alpha_selected=alphas,
        )
        with open(osp.join(RESULTS, "lower_order_baselines_030_summary.json"), "w") as f:
            f.write(summary.model_dump_json(indent=2))
        write_tex(
            ladder,
            pd.DataFrame([p.model_dump() for p in paired]),
            osp.join(RESULTS, "t11-lower-order-baselines-030.tex"),
        )
    if args.stage == "wandb":
        # refresh only the transformer references of an existing summary (the fits are
        # deterministic and already on file)
        path = osp.join(RESULTS, "lower_order_baselines_030_summary.json")
        with open(path) as f:
            data = json.load(f)
        data["wandb"] = []  # the old references are replaced, whatever their shape
        summary = LadderSummary(**data)
        summary.wandb = [r for arm in ARMS for r in wandb_references(arm)]
        with open(path, "w") as f:
            f.write(summary.model_dump_json(indent=2))
        write_tex(
            pd.read_csv(osp.join(RESULTS, "lower_order_baselines_030.csv")),
            pd.read_csv(osp.join(RESULTS, "lower_order_baselines_030_paired.csv")),
            osp.join(RESULTS, "t11-lower-order-baselines-030.tex"),
        )
    if args.stage in ("plot", "all"):
        ladder = pd.read_csv(osp.join(RESULTS, "lower_order_baselines_030.csv"))
        with open(osp.join(RESULTS, "lower_order_baselines_030_summary.json")) as f:
            refs = [WandbReference(**r) for r in json.load(f)["wandb"]]
        plot_ladder(
            ladder, refs, osp.join(IMAGES, f"lower_order_baselines_030_ladder_{timestamp()}")
        )
    print("finished")


if __name__ == "__main__":
    main()
