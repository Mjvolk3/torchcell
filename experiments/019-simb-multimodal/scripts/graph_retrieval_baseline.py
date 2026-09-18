# experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py
# [[experiments.019-simb-multimodal.scripts.graph_retrieval_baseline]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/graph_retrieval_baseline
"""B4: the cell graph as a RETRIEVAL KEY, on the partitions the trained arms use.

The trained CGT is handed nine gene graphs and consumes them as an attention mask. This
baseline consumes the same graphs the other way round: a strain is represented by the
ADJACENCY ROW of its deleted gene (the mean of the rows for a double deletion), the k
training strains with the most similar rows are retrieved by cosine similarity (common
neighbors over the geometric mean of the degrees), and the prediction is the
similarity-weighted mean of their training residuals added to the per-gene training mean.
Nothing is fit. The same neighbor average is run with the ProtT5 embedding as the key, the
strain's own measured profile as the key (the oracle of this function class), a fixed-seed
random embedding, and for every graph a degree-preserving configuration-model rewiring, so
the graph's contribution is read against both its own degree sequence and the best
sequence key.

Scored on the v13 expression partitions (fig3_core, split seeds 0 to 3) and the v14
proteome partitions (fig3_proteome, split seeds 0 to 3), with the per-feature Pearson of
expression_baselines_split.py, on the genes that are cell-graph nodes (6,127 of the 6,169
expression keys; all 1,850 proteins). k is chosen on validation over all strains and the
chosen k is applied to test. Every score is reported on three strain subsets of each
evaluation split: all strains, gene-disjoint strains (none of the deleted genes is deleted
in any training strain) and their complement (a deleted gene is shared with a training
strain), the latter further split into exact-genotype duplicates and partial overlaps. Two
further metrics accompany the standard one: the per-feature Pearson restricted to the 10%
(and 5%) of genes with the highest training-strain sd, the strata of
variance_stratified_pearson.py, and a per-strain Pearson over the 10% (and 5%) of genes
with the largest |target| for that strain, averaged over strains. The CGT prediction dumps
scored by gh_eval_ckpt_predictions.slurm are read through variance_stratified_pearson's
loader and scored on the same strains, genes, subsets and metrics, so the trained model,
the graph key and the ProtT5 key are compared under one rule. A split seed without a dump
leaves the CGT entry empty.

Run from the repo root (CPU; the launcher is gh_graph_retrieval_baseline.slurm):
    python experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py
    python experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py --rounds v13 --seeds 0 --graphs physical --out-dir /tmp/b4 --image-dir /tmp/b4
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator

load_dotenv()
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from expression_baselines import _embedding_matrix  # noqa: E402
from expression_baselines_split import (  # noqa: E402
    _load_records,
    _load_split,
    per_feature_pearson,
)
from variance_stratified_pearson import RUNS, _load_dump, per_gene_pearson  # noqa: E402

from torchcell.graph import SCerevisiaeGraph, build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils import (  # noqa: E402
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    panel_label,
    savefig_true_size_svg,
)
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

# The nine graphs of cgt_expr_006.yaml (inherited by v13 and v14), in the config's order.
GRAPHS: list[str] = [
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
UNION = "all9_union"
ROUNDS: dict[str, dict[str, str]] = {
    "v13": {
        "tag": "fig3_core",
        "label": "expression_log2_ratio",
        "name": "expression (Kemmeren)",
        "cgt_family": "V_ref",
    },
    "v14": {
        "tag": "fig3_proteome",
        "label": "protein_abundance",
        "name": "proteome (Messner)",
        "cgt_family": "P_ref",
    },
}
# expression_baselines' KNN_GRID plus 50: on validation the dense graph keys select 25
# or 50, so the committed grid's ceiling of 25 would truncate the sweep.
K_GRID: tuple[int, ...] = (1, 3, 5, 10, 25, 50)
REWIRE_SEED = 7
RANDOM_EMBEDDING_SEED = 0
RANDOM_EMBEDDING_DIM = 1024
PROT_T5 = "prot_T5_all"
# A subset with fewer strains than this is counted but not scored: per-feature Pearson
# needs three finite pairs per gene, and below five strains the number is noise.
MIN_STRAINS = 5
SUBSETS: list[str] = [
    "all",
    "gene_disjoint",
    "shared",
    "exact_duplicate",
    "double_partial",
    "single_partial",
]
# per_strain_top* is the Pearson between prediction and target over the genes with the
# largest |target| of that strain; prediction and target both carry the per-gene training
# mean, so the train-mean reference (B0) already scores high on it. The _residual
# variant subtracts that mean from both before selecting and scoring, and reads the
# strain-specific part alone.
METRICS: list[str] = [
    "pearson_per_feature",
    "top10_train_variance",
    "top5_train_variance",
    "per_strain_top10",
    "per_strain_top5",
    "per_strain_top10_residual",
    "per_strain_top5_residual",
]
# Keys that are not graph rows: no rewired control, and excluded from the paired tests.
NON_GRAPH_KEYS: tuple[str, ...] = (
    "prot_T5",
    "random_embedding",
    "oracle",
    "train_mean",
)
KEY_LABEL: dict[str, str] = {
    "physical": "physical",
    "regulatory": "regulatory",
    "tflink": "tflink",
    "string12_0_neighborhood": "neighborhood",
    "string12_0_fusion": "fusion",
    "string12_0_cooccurence": "cooccurence",
    "string12_0_coexpression": "coexpression",
    "string12_0_experimental": "experimental",
    "string12_0_database": "database",
    UNION: "union of 9",
    "prot_T5": "ProtT5",
    "random_embedding": "random",
    "oracle": "own profile",
    "train_mean": "train mean (B0)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", nargs="+", default=list(ROUNDS), choices=list(ROUNDS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument(
        "--graphs",
        nargs="+",
        default=GRAPHS,
        choices=GRAPHS,
        help="graphs to use as keys; the union key is over exactly these",
    )
    p.add_argument("--k-grid", nargs="+", type=int, default=list(K_GRID))
    p.add_argument(
        "--out-dir",
        default=None,
        help="where graph_retrieval_baseline.json lands (default: the experiment "
        "results directory)",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help="where the figure lands (default: $ASSET_IMAGES_DIR/019-simb-multimodal)",
    )
    p.add_argument(
        "--report-md",
        default=None,
        help="optional path for a markdown rendering of the tables",
    )
    p.add_argument(
        "--figure-only",
        action="store_true",
        help="redraw the figure (and the markdown tables, if --report-md is given) "
        "from the existing results JSON without recomputing anything",
    )
    return p.parse_args()


# ---- graphs as keys ------------------------------------------------------------------


def _undirected_simple(g: nx.Graph) -> nx.Graph:
    """The graph as a simple undirected graph with no self loops, the form the row uses."""
    h = nx.Graph(g.to_undirected() if g.is_directed() else g)
    h.remove_edges_from(nx.selfloop_edges(h))
    return h


def _adjacency(g: nx.Graph, gene_pos: dict[str, int]) -> sp.csr_matrix:
    """Binary symmetric adjacency over the genome gene set, in ``gene_pos`` order."""
    n = len(gene_pos)
    rows: list[int] = []
    cols: list[int] = []
    for u, v in g.edges():
        if u in gene_pos and v in gene_pos and u != v:
            rows += [gene_pos[u], gene_pos[v]]
            cols += [gene_pos[v], gene_pos[u]]
    a = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    ).tocsr()
    a.data[:] = 1.0
    return a


def _rewired(g: nx.Graph, seed: int) -> nx.Graph:
    """Degree-preserving control: the configuration model on ``g``'s degree sequence.

    Multi-edges collapse and self loops are dropped, as in the scratchpad measurement, so
    the control carries slightly fewer edges than the graph; its degree sequence is the
    graph's up to that collapse. The node labels are the graph's, so the rows index the
    same genes.
    """
    nodes = list(g.nodes())
    deg = [int(d) for _, d in g.degree(nodes)]
    h = nx.Graph(nx.configuration_model(deg, seed=seed))
    h.remove_edges_from(nx.selfloop_edges(h))
    return nx.relabel_nodes(h, dict(enumerate(nodes)))


def _row_keys(
    perts: list[list[str]], a: sp.csr_matrix, gene_pos: dict[str, int], ok: np.ndarray
) -> np.ndarray:
    """Mean adjacency row over a strain's deleted genes; zero rows where ``ok`` is False."""
    out = np.zeros((len(perts), a.shape[1]), dtype=np.float32)
    for i, genes in enumerate(perts):
        if ok[i]:
            out[i] = np.asarray(a[[gene_pos[g] for g in genes]].mean(axis=0)).ravel()
    return out


def _vector_keys(
    perts: list[list[str]], emb: dict[str, np.ndarray], dim: int, ok: np.ndarray
) -> np.ndarray:
    """Mean embedding over a strain's deleted genes, as _pert_matrix does for B3."""
    out = np.zeros((len(perts), dim), dtype=np.float64)
    for i, genes in enumerate(perts):
        if ok[i]:
            out[i] = np.mean([emb[g] for g in genes], axis=0)
    return out


def _zscore(keys: dict[str, np.ndarray], ok_tr: np.ndarray) -> dict[str, np.ndarray]:
    m = keys["train"][ok_tr].mean(axis=0, keepdims=True)
    s = keys["train"][ok_tr].std(axis=0, keepdims=True) + 1e-8
    return {split: (v - m) / s for split, v in keys.items()}


# ---- the neighbor average -------------------------------------------------------------


def _knn(
    key_tr: np.ndarray, r_tr: np.ndarray, key_q: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted mean residual of the k most cosine-similar training strains.

    Weights are the cosine similarities clipped at zero; a query with no positive
    similarity to any training strain (an isolated gene on this graph) predicts the
    training mean (residual 0) and is flagged as uncovered. A residual the neighbor did
    not measure (NaN, proteome) drops out of that gene's weighted mean.
    """
    a = key_tr / (np.linalg.norm(key_tr, axis=1, keepdims=True) + 1e-12)
    b = key_q / (np.linalg.norm(key_q, axis=1, keepdims=True) + 1e-12)
    sim = b @ a.T
    finite = np.isfinite(r_tr)
    r0 = np.where(finite, r_tr, 0.0)
    out = np.zeros((len(key_q), r_tr.shape[1]), dtype=np.float64)
    covered = np.zeros(len(key_q), dtype=bool)
    for i in range(len(key_q)):
        idx = np.argpartition(-sim[i], k)[:k]
        w = np.maximum(sim[i, idx], 0.0)
        if w.sum() <= 0:
            continue
        num = w @ r0[idx]
        den = w @ finite[idx].astype(np.float64)
        out[i] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        covered[i] = True
    return out, covered


# ---- metrics ---------------------------------------------------------------------------


def _per_strain_top(pred: np.ndarray, true: np.ndarray, frac: float) -> float:
    """Pearson over the ``frac`` genes with the largest |target| per strain, mean over strains."""
    vals: list[float] = []
    for i in range(len(true)):
        idx = np.flatnonzero(np.isfinite(true[i]) & np.isfinite(pred[i]))
        m = max(int(round(frac * len(idx))), 3)
        sel = idx[np.argsort(-np.abs(true[i, idx]))[:m]]
        p, t = pred[i, sel], true[i, sel]
        if p.std() < 1e-8 or t.std() < 1e-8:
            continue
        vals.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def _score(
    pred: np.ndarray,
    true: np.ndarray,
    mu: np.ndarray,
    pct: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    n = int(mask.sum())
    if n < MIN_STRAINS:
        return {"n": n, **{m: None for m in METRICS}}
    p, t = pred[mask], true[mask]
    r_gene = per_gene_pearson(p, t)

    def _mean(r: np.ndarray) -> float:
        # A constant prediction (the train mean) has no per-gene Pearson: NaN, kept out
        # of the JSON as null rather than as a non-standard NaN token.
        return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")

    scores = {
        "pearson_per_feature": per_feature_pearson(p, t),
        "top10_train_variance": _mean(r_gene[pct > 90]),
        "top5_train_variance": _mean(r_gene[pct > 95]),
        "per_strain_top10": _per_strain_top(p, t, 0.10),
        "per_strain_top5": _per_strain_top(p, t, 0.05),
        "per_strain_top10_residual": _per_strain_top(p - mu, t - mu, 0.10),
        "per_strain_top5_residual": _per_strain_top(p - mu, t - mu, 0.05),
    }
    return {"n": n, **{k: (v if np.isfinite(v) else None) for k, v in scores.items()}}


def _subset_masks(
    perts: list[list[str]], perts_tr: list[list[str]], ok: np.ndarray
) -> dict[str, np.ndarray]:
    """Strain subsets by deleted-gene overlap with the training strains.

    gene_disjoint: no deleted gene is deleted in any training strain. shared: at least
    one is. exact_duplicate: the deleted gene SET equals a training strain's.
    double_partial: a double deletion sharing a gene with a training strain but not its
    whole genotype (a Sameith double whose single is in train). single_partial: a single
    deletion whose gene is deleted in training only inside another genotype.
    """
    train_sets = {frozenset(p) for p in perts_tr}
    train_genes: set[str] = set().union(*train_sets) if train_sets else set()
    sets = [frozenset(p) for p in perts]
    exact = np.array([s in train_sets for s in sets]) & ok
    shared = np.array([any(g in train_genes for g in s) for s in sets]) & ok
    double = np.array([len(s) == 2 for s in sets])
    return {
        "all": ok.copy(),
        "gene_disjoint": ok & ~shared,
        "shared": shared,
        "exact_duplicate": exact,
        "double_partial": shared & ~exact & double,
        "single_partial": shared & ~exact & ~double,
    }


# ---- one round, one split seed ----------------------------------------------------------


def _run_seed(
    rnd: str,
    seed: int,
    args: argparse.Namespace,
    data_root: str,
    gene_pos: dict[str, int],
    gene_set: set[str],
    adj: dict[str, sp.csr_matrix],
    prot_t5: dict[str, np.ndarray],
    random_emb: np.ndarray,
) -> dict[str, Any]:
    rd = ROUNDS[rnd]
    base = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", rd["tag"]
    )
    split = _load_split(osp.join(base, "data_module_cache"), seed, rd["label"])
    perts: dict[str, list[list[str]]] = {}
    y: dict[str, np.ndarray] = {}
    keys_raw: list[str] | None = None
    for s in ("train", "val", "test"):
        p, mat, k = _load_records(base, split[s], rd["label"])
        if keys_raw is None:
            keys_raw = k
        elif k != keys_raw:
            raise ValueError("gene key order differs between splits")
        perts[s], y[s] = p, mat.astype(np.float64)
    assert keys_raw is not None
    # The genes the per-gene head predicts: the LMDB keys that are cell-graph nodes.
    in_graph = np.array([k in gene_set for k in keys_raw])
    keys = [k for k, ok in zip(keys_raw, in_graph) if ok]
    y = {s: v[:, in_graph] for s, v in y.items()}
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(y["train"], axis=0, keepdims=True)
    if not np.isfinite(mu).all():
        raise ValueError("a gene has no finite training value; cannot center it")
    r = {s: v - mu for s, v in y.items()}
    sd_tr = np.nanstd(y["train"], axis=0, ddof=1)
    pct = 100.0 * (np.argsort(np.argsort(sd_tr)) + 0.5) / len(sd_tr)
    # One strain mask for every key: each deleted gene is a genome gene with a ProtT5
    # vector, so the graph row, the embedding and the control all exist for the strain.
    ok = {
        s: np.array([all(g in gene_pos and g in prot_t5 for g in p) for p in perts[s]])
        for s in perts
    }
    masks = {s: _subset_masks(perts[s], perts["train"], ok[s]) for s in ("val", "test")}
    counts = {
        s: {
            **{name: int(m.sum()) for name, m in masks[s].items()},
            "records": int(len(perts[s])),
            "excluded_no_key": int((~ok[s]).sum()),
            "double_deletion": int(sum(len(p) == 2 for p in perts[s])),
            "shared_double": int(
                (masks[s]["shared"] & np.array([len(p) == 2 for p in perts[s]])).sum()
            ),
        }
        for s in ("val", "test")
    }
    print(
        f"\n{rnd} split seed {seed}: train={len(perts['train'])} "
        f"val={len(perts['val'])} test={len(perts['test'])}; genes={len(keys)} "
        f"of {len(keys_raw)}",
        flush=True,
    )
    for s in ("val", "test"):
        c = counts[s]
        print(
            f"  {s}: scored {c['all']} (excluded {c['excluded_no_key']}); "
            f"gene-disjoint {c['gene_disjoint']}, shared {c['shared']} "
            f"(exact {c['exact_duplicate']}, double partial {c['double_partial']}, "
            f"single partial {c['single_partial']}); doubles {c['double_deletion']}",
            flush=True,
        )

    # ---- keys --------------------------------------------------------------------------
    key_specs: list[tuple[str, str]] = []
    for g in args.graphs:
        key_specs += [(g, "row"), (f"{g}_rewired", "row")]
    key_specs += [(UNION, "row"), (f"{UNION}_rewired", "row")]
    key_specs += [("prot_T5", "vec"), ("random_embedding", "vec"), ("oracle", "own")]
    # An all-zero key has no positive similarity to anything, so _knn returns the
    # training mean for every strain: the B0 reference under the same scoring.
    key_specs += [("train_mean", "zero")]
    dim_t5 = len(next(iter(prot_t5.values())))
    rnd_emb = {g: random_emb[i] for g, i in gene_pos.items()}
    out_keys: dict[str, Any] = {}
    r_tr = r["train"][ok["train"]]
    for name, kind in key_specs:
        if kind == "row":
            keys_s = {s: _row_keys(perts[s], adj[name], gene_pos, ok[s]) for s in perts}
        elif kind == "vec":
            emb = prot_t5 if name == "prot_T5" else rnd_emb
            dim = dim_t5 if name == "prot_T5" else RANDOM_EMBEDDING_DIM
            keys_s = _zscore(
                {s: _vector_keys(perts[s], emb, dim, ok[s]) for s in perts}, ok["train"]
            )
        elif kind == "own":
            keys_s = {s: np.nan_to_num(r[s], nan=0.0) for s in perts}
        else:
            keys_s = {s: np.zeros((len(perts[s]), 1)) for s in perts}
        cells: dict[str, list[dict[str, Any]]] = {"val": [], "test": []}
        preds: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            "val": {},
            "test": {},
        }
        for s in ("val", "test"):
            for k in args.k_grid:
                res, cov = _knn(keys_s["train"][ok["train"]], r_tr, keys_s[s], k)
                pred = res + mu
                preds[s][k] = (pred, cov)
                cells[s].append(
                    {
                        "k": k,
                        "pearson_per_feature": per_feature_pearson(
                            pred[ok[s]], y[s][ok[s]]
                        ),
                        "coverage": float(cov[ok[s]].mean()),
                    }
                )
        k_sel = max(cells["val"], key=lambda c: float(c["pearson_per_feature"]))["k"]
        entry: dict[str, Any] = {"k": int(k_sel), "cells": cells}
        for s in ("val", "test"):
            pred, cov = preds[s][k_sel]
            entry[s] = {
                sub: {
                    **_score(pred, y[s], mu, pct, masks[s][sub]),
                    "coverage": float(cov[masks[s][sub]].mean())
                    if masks[s][sub].any()
                    else None,
                }
                for sub in SUBSETS
            }
        out_keys[name] = entry
        v, t = entry["val"], entry["test"]
        print(
            f"  {name:34s} k={k_sel:3d} val {v['all']['pearson_per_feature']:+.4f} "
            f"test {t['all']['pearson_per_feature']:+.4f} | gene-disjoint "
            f"{_fmt(v['gene_disjoint']['pearson_per_feature'])} "
            f"{_fmt(t['gene_disjoint']['pearson_per_feature'])} | shared "
            f"{_fmt(v['shared']['pearson_per_feature'])} "
            f"{_fmt(t['shared']['pearson_per_feature'])} "
            f"(val coverage {v['all']['coverage']:.2f})",
            flush=True,
        )

    # ---- the CGT dumps, same strains, genes, subsets and metrics --------------------------
    out_cgt: dict[str, Any] = {}
    for spec in RUNS:
        if spec["round"] != rnd or spec["split_seed"] != seed:
            continue
        entry = {"run": spec["run"], "group": spec["group"], "seed": spec["seed"]}
        for s in ("val", "test"):
            dump = osp.join(data_root, f"{s}-predictions", f"{spec['group']}.json")
            pred, tgt, rec_idx = _load_dump(dump, keys, gene_set)
            if rec_idx != list(split[s]):
                raise ValueError(f"{dump}: record order differs from the split's {s}")
            if not np.allclose(np.nan_to_num(tgt), np.nan_to_num(y[s]), atol=1e-3):
                raise ValueError(f"{dump}: targets differ from the LMDB {s} rows")
            entry[s] = {
                sub: _score(pred, y[s], mu, pct, masks[s][sub]) for sub in SUBSETS
            }
            # Every record, no key mask: the number variance_stratified_pearson.py reports.
            entry[s]["all_records_unmasked"] = {
                "n": int(len(rec_idx)),
                "pearson_per_feature": per_feature_pearson(pred, y[s]),
            }
        out_cgt[spec["arm"]] = entry
        v, t = entry["val"], entry["test"]
        print(
            f"  CGT {spec['arm']:30s}       val {v['all']['pearson_per_feature']:+.4f} "
            f"test {t['all']['pearson_per_feature']:+.4f} | gene-disjoint "
            f"{_fmt(v['gene_disjoint']['pearson_per_feature'])} "
            f"{_fmt(t['gene_disjoint']['pearson_per_feature'])} | shared "
            f"{_fmt(v['shared']['pearson_per_feature'])} "
            f"{_fmt(t['shared']['pearson_per_feature'])} "
            f"(unmasked val {v['all_records_unmasked']['pearson_per_feature']:+.4f})",
            flush=True,
        )

    return {
        "split_seed": seed,
        "n_train": int(len(perts["train"])),
        "n_train_scored": int(ok["train"].sum()),
        "n_genes": int(len(keys)),
        "n_genes_raw": int(len(keys_raw)),
        "counts": counts,
        "keys": out_keys,
        "cgt": out_cgt,
    }


def _fmt(x: float | None) -> str:
    return "   n/a " if x is None else f"{x:+.4f}"


# ---- aggregation over seeds ------------------------------------------------------------


def _paired(diffs: list[float]) -> dict[str, Any]:
    d = np.array(diffs, dtype=np.float64)
    return {
        "mean": float(d.mean()) if len(d) else None,
        "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
        "n": int(len(d)),
        "n_positive": int((d > 0).sum()),
    }


def _aggregate(per_seed: dict[int, dict[str, Any]], family: str) -> dict[str, Any]:
    """Mean and sd over seeds per key, and paired within-seed differences."""
    seeds = sorted(per_seed)
    key_names = list(per_seed[seeds[0]]["keys"])
    summary: dict[str, Any] = {}
    for name in key_names:
        summary[name] = {
            "k_by_seed": {str(s): per_seed[s]["keys"][name]["k"] for s in seeds}
        }
        for split in ("val", "test"):
            for sub in SUBSETS:
                for metric in METRICS:
                    vals = [
                        per_seed[s]["keys"][name][split][sub][metric] for s in seeds
                    ]
                    vals = [v for v in vals if v is not None]
                    summary[name][f"{split}/{sub}/{metric}"] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "n_seeds": len(vals),
                    }
    cgt: dict[str, Any] = {}
    arms = sorted({a for s in seeds for a in per_seed[s]["cgt"]})
    for arm in arms:
        cgt[arm] = {}
        for split in ("val", "test"):
            for sub in SUBSETS:
                for metric in METRICS:
                    vals = [
                        per_seed[s]["cgt"][arm][split][sub][metric]
                        for s in seeds
                        if arm in per_seed[s]["cgt"]
                    ]
                    vals = [v for v in vals if v is not None]
                    cgt[arm][f"{split}/{sub}/{metric}"] = {
                        "mean": float(np.mean(vals)) if vals else None,
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        "n_seeds": len(vals),
                    }
    # The CGT family (the reference arm, one dump per split seed where it exists).
    family_by_seed = {
        s: next((a for a in per_seed[s]["cgt"] if a.startswith(family)), None)
        for s in seeds
    }
    paired: dict[str, Any] = {}
    graph_keys = [
        n for n in key_names if not n.endswith("_rewired") and n not in NON_GRAPH_KEYS
    ]
    for name in graph_keys:
        paired[name] = {}
        for split in ("val", "test"):
            for sub in ("all", "gene_disjoint", "shared"):
                for ref_label, ref in (
                    ("minus_prot_T5", "prot_T5"),
                    ("minus_rewired", f"{name}_rewired"),
                ):
                    diffs = []
                    for s in seeds:
                        a = per_seed[s]["keys"][name][split][sub]["pearson_per_feature"]
                        b = per_seed[s]["keys"][ref][split][sub]["pearson_per_feature"]
                        if a is not None and b is not None:
                            diffs.append(a - b)
                    paired[name][f"{split}/{sub}/{ref_label}"] = _paired(diffs)
                diffs = []
                for s in seeds:
                    arm = family_by_seed[s]
                    if arm is None:
                        continue
                    a = per_seed[s]["keys"][name][split][sub]["pearson_per_feature"]
                    b = per_seed[s]["cgt"][arm][split][sub]["pearson_per_feature"]
                    if a is not None and b is not None:
                        diffs.append(a - b)
                paired[name][f"{split}/{sub}/minus_cgt_{family}"] = _paired(diffs)
    return {"keys": summary, "cgt": cgt, "paired": paired, "cgt_family": family}


# ---- markdown tables --------------------------------------------------------------------


def _ms(cell: dict[str, Any]) -> str:
    if cell["mean"] is None:
        return "n/a"
    if cell["sd"] is None:
        return f"{cell['mean']:+.3f} (n={cell['n_seeds']})"
    return f"{cell['mean']:+.3f} +/- {cell['sd']:.3f}"


def _markdown(out: dict[str, Any]) -> str:
    lines: list[str] = [
        "# B4 graph retrieval baseline: tables",
        "",
        f"Generated by `{out['generated_by']}`; mean +/- sd over split seeds "
        f"{out['seeds']}; k grid {out['k_grid']}; k chosen on validation, all strains.",
        "",
    ]
    for rnd, res in out["rounds"].items():
        agg = res["summary"]
        seeds = sorted(int(s) for s in res["per_seed"])
        lines += [f"## {rnd}: {ROUNDS[rnd]['name']}", ""]
        lines += ["### Strain subsets per split seed", ""]
        lines += [
            "| seed | split | records | scored | gene-disjoint | shared | exact dup | double partial | single partial | doubles | shared doubles |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
        ]
        for s in seeds:
            for split in ("val", "test"):
                c = res["per_seed"][str(s)]["counts"][split]
                lines.append(
                    f"| {s} | {split} | {c['records']} | {c['all']} | {c['gene_disjoint']} | "
                    f"{c['shared']} | {c['exact_duplicate']} | {c['double_partial']} | "
                    f"{c['single_partial']} | {c['double_deletion']} | {c['shared_double']} |"
                )
        lines.append("")
        for split in ("val", "test"):
            lines += [f"### {split}: per-feature Pearson by strain subset", ""]
            lines += [
                "| key | k per seed | all | gene-disjoint | shared | exact dup | double partial |",
                "|---|---|--:|--:|--:|--:|--:|",
            ]
            for name, sm in agg["keys"].items():
                ks = ",".join(str(v) for v in sm["k_by_seed"].values())
                lines.append(
                    f"| {name} | {ks} | "
                    + " | ".join(
                        _ms(sm[f"{split}/{sub}/pearson_per_feature"])
                        for sub in (
                            "all",
                            "gene_disjoint",
                            "shared",
                            "exact_duplicate",
                            "double_partial",
                        )
                    )
                    + " |"
                )
            for arm, sm in agg["cgt"].items():
                lines.append(
                    f"| CGT {arm} | -- | "
                    + " | ".join(
                        _ms(sm[f"{split}/{sub}/pearson_per_feature"])
                        for sub in (
                            "all",
                            "gene_disjoint",
                            "shared",
                            "exact_duplicate",
                            "double_partial",
                        )
                    )
                    + " |"
                )
            lines.append("")
        for split in ("val", "test"):
            lines += [
                f"### {split}: top-variance and per-strain top-|target| metrics, all strains",
                "",
            ]
            lines += [
                "| key | per-feature | top 10% train-sd genes | top 5% | per-strain top 10% | per-strain top 5% | per-strain top 10% residual | top 5% residual |",
                "|---|--:|--:|--:|--:|--:|--:|--:|",
            ]
            rows = list(agg["keys"].items()) + [
                (f"CGT {a}", sm) for a, sm in agg["cgt"].items()
            ]
            for name, sm in rows:
                lines.append(
                    f"| {name} | "
                    + " | ".join(_ms(sm[f"{split}/all/{m}"]) for m in METRICS)
                    + " |"
                )
            lines.append("")
        lines += ["### Paired within-seed differences, per-feature Pearson", ""]
        lines += [
            "| key | split | subset | minus ProtT5 | minus rewired | minus CGT "
            + agg["cgt_family"]
            + " |",
            "|---|---|---|--:|--:|--:|",
        ]
        for name, pr in agg["paired"].items():
            for split in ("val", "test"):
                for sub in ("all", "gene_disjoint", "shared"):
                    cells = []
                    for ref in (
                        "minus_prot_T5",
                        "minus_rewired",
                        f"minus_cgt_{agg['cgt_family']}",
                    ):
                        p = pr[f"{split}/{sub}/{ref}"]
                        if p["n"] == 0:
                            cells.append("n/a")
                        elif p["sd"] is None:
                            cells.append(
                                f"{p['mean']:+.3f} (n={p['n']}, {p['n_positive']}+)"
                            )
                        else:
                            cells.append(
                                f"{p['mean']:+.3f} +/- {p['sd']:.3f} (n={p['n']}, {p['n_positive']}+)"
                            )
                    lines.append(
                        f"| {name} | {split} | {sub} | " + " | ".join(cells) + " |"
                    )
        lines.append("")
    return "\n".join(lines)


# ---- figure ------------------------------------------------------------------------------


def _bar_cell(sm: dict[str, Any], field: str) -> tuple[float, float]:
    c = sm[field]
    return (
        float("nan") if c["mean"] is None else c["mean"],
        0.0 if c["sd"] is None else c["sd"],
    )


def _figure(out: dict[str, Any], img_dir: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "svg.fonttype": "none",
            "axes.linewidth": 0.5,
        }
    )
    rounds = list(out["rounds"])
    fig, axes = plt.subplots(
        2,
        len(rounds),
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(105)),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.94, bottom=0.12, wspace=0.18, hspace=0.62
    )
    letters = iter("abcdefgh")
    for col, rnd in enumerate(rounds):
        agg = out["rounds"][rnd]["summary"]
        graph_keys = [
            n
            for n in agg["keys"]
            if not n.endswith("_rewired") and n not in NON_GRAPH_KEYS
        ]
        control_of = {n: f"{n}_rewired" for n in graph_keys}
        control_of["prot_T5"] = "random_embedding"
        names = graph_keys + ["prot_T5"]
        # (top) test score, all strains: key against its control, per-seed points.
        ax = axes[0][col]
        x = np.arange(len(names))
        w = 0.38
        for j, name in enumerate(names):
            for off, key, color in (
                (-w / 2, name, PLOT_PALETTE[0]),
                (w / 2, control_of[name], PLOT_PALETTE[5]),
            ):
                m, sd = _bar_cell(agg["keys"][key], "test/all/pearson_per_feature")
                ax.bar(
                    x[j] + off,
                    m,
                    width=w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.errorbar(
                    x[j] + off,
                    m,
                    yerr=sd,
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.5,
                )
        for arm, sm in agg["cgt"].items():
            if not arm.startswith(agg["cgt_family"]):
                continue
            m, sd = _bar_cell(sm, "test/all/pearson_per_feature")
            ax.axhline(
                m,
                color=PLOT_PALETTE[1],
                linewidth=0.8,
                linestyle="--",
                label=f"CGT {arm}, test",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([KEY_LABEL[n] for n in names], rotation=45, ha="right")
        ax.set_ylabel("per-feature Pearson, test")
        ax.set_title(
            f"{ROUNDS[rnd]['name']}: retrieval key vs its control, all strains"
        )
        _style_axis(ax)
        ax.bar(
            [np.nan],
            [np.nan],
            color=PLOT_PALETTE[0],
            edgecolor="black",
            linewidth=0.5,
            label="key",
        )
        ax.bar(
            [np.nan],
            [np.nan],
            color=PLOT_PALETTE[5],
            edgecolor="black",
            linewidth=0.5,
            label="control (rewired / random)",
        )
        ax.legend(frameon=True, edgecolor="black", fancybox=False, loc="upper left")
        panel_label(ax, next(letters))
        # (bottom) by strain subset: union key, ProtT5 key, CGT; validation dark, test light.
        ax = axes[1][col]
        series = [(UNION, "union of 9 kNN", 0), ("prot_T5", "ProtT5 kNN", 2)]
        cgt_arms = [a for a in agg["cgt"] if a.startswith(agg["cgt_family"])]
        # A subset empty on every seed (the proteome has no shared-gene strain) is
        # left off the axis rather than drawn as an empty group.
        subsets = [
            sub
            for sub in ("all", "gene_disjoint", "shared")
            if agg["keys"][UNION][f"val/{sub}/pearson_per_feature"]["n_seeds"] > 0
        ]
        n_series = len(series) + (1 if cgt_arms else 0)
        gw = 0.8 / n_series
        x = np.arange(len(subsets))
        for i, (key, label, ci) in enumerate(series):
            for jj, (split, face) in enumerate(
                (("val", PLOT_PALETTE[ci]), ("test", PLOT_PALETTE_FILL[ci]))
            ):
                ms = [
                    _bar_cell(agg["keys"][key], f"{split}/{sub}/pearson_per_feature")
                    for sub in subsets
                ]
                pos = x - 0.4 + gw * (i + 0.25 + 0.5 * jj)
                ax.bar(
                    pos,
                    [m for m, _ in ms],
                    width=gw / 2,
                    color=face,
                    edgecolor="black",
                    linewidth=0.5,
                    label=label if jj == 0 else None,
                )
                ax.errorbar(
                    pos,
                    [m for m, _ in ms],
                    yerr=[s for _, s in ms],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.5,
                )
        if cgt_arms:
            i = len(series)
            for jj, (split, face) in enumerate(
                (("val", PLOT_PALETTE[1]), ("test", PLOT_PALETTE_FILL[1]))
            ):
                vals = []
                for sub in subsets:
                    per = [
                        _bar_cell(agg["cgt"][a], f"{split}/{sub}/pearson_per_feature")[
                            0
                        ]
                        for a in cgt_arms
                    ]
                    per = [v for v in per if np.isfinite(v)]
                    vals.append(
                        (
                            np.mean(per) if per else np.nan,
                            np.std(per, ddof=1) if len(per) > 1 else 0.0,
                        )
                    )
                pos = x - 0.4 + gw * (i + 0.25 + 0.5 * jj)
                ax.bar(
                    pos,
                    [m for m, _ in vals],
                    width=gw / 2,
                    color=face,
                    edgecolor="black",
                    linewidth=0.5,
                    label=f"CGT {agg['cgt_family']}" if jj == 0 else None,
                )
                ax.errorbar(
                    pos,
                    [m for m, _ in vals],
                    yerr=[s for _, s in vals],
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.5,
                    capsize=1.5,
                )
        ax.bar(
            [np.nan],
            [np.nan],
            color=PLOT_PALETTE[5],
            edgecolor="black",
            linewidth=0.5,
            label="validation",
        )
        ax.bar(
            [np.nan],
            [np.nan],
            color=PLOT_PALETTE_FILL[5],
            edgecolor="black",
            linewidth=0.5,
            label="test",
        )
        ax.set_xticks(x)
        subset_label = {
            "all": "all strains",
            "gene_disjoint": "gene-disjoint",
            "shared": "shared gene",
        }
        ax.set_xticklabels([subset_label[sub] for sub in subsets])
        ax.set_ylabel("per-feature Pearson")
        ax.set_title(f"{ROUNDS[rnd]['name']}: by overlap with training genotypes")
        _style_axis(ax)
        ax.legend(
            frameon=True, edgecolor="black", fancybox=False, loc="upper left", ncol=2
        )
        panel_label(ax, next(letters))
    stem = "graph_retrieval_baseline"
    fig.savefig(osp.join(img_dir, f"{stem}.png"), dpi=300)
    savefig_true_size_svg(fig, osp.join(img_dir, f"{stem}.svg"))
    plt.close(fig)


def _style_axis(ax: Any) -> None:
    ax.set_ylim(-0.1, 0.7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", which="both", linewidth=0.3, alpha=0.4)
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)


# ---- main --------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_root = os.environ["DATA_ROOT"]
    out_dir = args.out_dir or experiment_results_dir("019-simb-multimodal", __file__)
    img_dir = args.image_dir or osp.join(
        os.environ["ASSET_IMAGES_DIR"], "019-simb-multimodal"
    )
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    dst = osp.join(out_dir, "graph_retrieval_baseline.json")
    if args.figure_only:
        with open(dst) as fh:
            out = json.load(fh)
        if args.report_md:
            with open(args.report_md, "w") as fh:
                fh.write(_markdown(out))
            print(f"wrote {args.report_md}")
        _figure(out, img_dir)
        print(f"wrote {osp.join(img_dir, 'graph_retrieval_baseline.svg')}")
        return

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    gene_set = set(genome.gene_set)
    genes = sorted(gene_set)
    gene_pos = {g: i for i, g in enumerate(genes)}
    print(f"{len(genes)} genes in the genome gene set", flush=True)
    prot_t5 = _embedding_matrix((PROT_T5,), data_root, genome, graph)
    random_emb = np.random.default_rng(RANDOM_EMBEDDING_SEED).standard_normal(
        (len(genes), RANDOM_EMBEDDING_DIM)
    )

    multigraph = build_gene_multigraph(graph=graph, graph_names=args.graphs)
    adj: dict[str, sp.csr_matrix] = {}
    graph_stats: dict[str, Any] = {}
    union = nx.Graph()
    for name in args.graphs:
        g = _undirected_simple(multigraph[name].graph)
        union.add_edges_from(g.edges())
        adj[name] = _adjacency(g, gene_pos)
        adj[f"{name}_rewired"] = _adjacency(_rewired(g, REWIRE_SEED), gene_pos)
        graph_stats[name] = {
            "edges": int(adj[name].nnz // 2),
            "edges_rewired": int(adj[f"{name}_rewired"].nnz // 2),
            "genes_with_neighbor": int((adj[name].sum(axis=1) > 0).sum()),
        }
        print(f"  {name}: {graph_stats[name]}", flush=True)
    union.remove_edges_from(nx.selfloop_edges(union))
    adj[UNION] = _adjacency(union, gene_pos)
    adj[f"{UNION}_rewired"] = _adjacency(_rewired(union, REWIRE_SEED), gene_pos)
    graph_stats[UNION] = {
        "edges": int(adj[UNION].nnz // 2),
        "edges_rewired": int(adj[f"{UNION}_rewired"].nnz // 2),
        "genes_with_neighbor": int((adj[UNION].sum(axis=1) > 0).sum()),
    }
    print(f"  {UNION}: {graph_stats[UNION]}", flush=True)

    out: dict[str, Any] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/graph_retrieval_baseline.py",
        "seeds": args.seeds,
        "k_grid": args.k_grid,
        "graphs": args.graphs,
        "rewire_seed": REWIRE_SEED,
        "random_embedding": {
            "seed": RANDOM_EMBEDDING_SEED,
            "dim": RANDOM_EMBEDDING_DIM,
        },
        "k_selection": "validation, all strains, per-feature Pearson",
        "min_strains_to_score": MIN_STRAINS,
        "metrics": METRICS,
        "subsets": SUBSETS,
        "graph_stats": graph_stats,
        "rounds": {},
    }
    for rnd in args.rounds:
        per_seed: dict[int, dict[str, Any]] = {}
        for seed in args.seeds:
            per_seed[seed] = _run_seed(
                rnd, seed, args, data_root, gene_pos, gene_set, adj, prot_t5, random_emb
            )
        out["rounds"][rnd] = {
            "tag": ROUNDS[rnd]["tag"],
            "label": ROUNDS[rnd]["label"],
            "per_seed": {str(s): v for s, v in per_seed.items()},
            "summary": _aggregate(per_seed, ROUNDS[rnd]["cgt_family"]),
        }

    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dst}")
    if args.report_md:
        with open(args.report_md, "w") as fh:
            fh.write(_markdown(out))
        print(f"wrote {args.report_md}")
    _figure(out, img_dir)
    print(f"wrote {osp.join(img_dir, 'graph_retrieval_baseline.svg')}")


if __name__ == "__main__":
    main()
