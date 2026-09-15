# experiments/019-simb-multimodal/scripts/expression_baselines_split.py
# [[experiments.019-simb-multimodal.scripts.expression_baselines_split]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/expression_baselines_split
"""The linear baselines on the SAME partition the trained arms use, one split seed at a time.

expression_baselines.py scored B0 to B3 on the oracle-family split, a private permutation
of the raw Kemmeren LMDB, so its numbers and the trained arms' were only approximately on
the same strains. This script reads the partition the CellDataModule wrote for a split
seed (``data_module_cache/index_details_seed_<k>.json``, the expression record indices per
split) and the phenotype values from the SAME processed dataset the model trains on, so
baseline and model see one train set, one 155-strain val set and one test set per seed.
Companion to the v13 split round (conf/cgt_expr_v13_split.yaml), which trains H_ref and
H_concat on split seeds 0 to 3 and on split 0 with the test records folded into train.

The four baselines are those of Ahlmann-Eltze, Huber and Anders 2025
(doi:10.1038/s41592-025-02772-6), unchanged from expression_baselines.py:

  B0  per-gene training mean            B2  Y ~ G W P^T + b, ridge, rank-swept
  B1  no change (log2 ratio 0)          B3  mean profile of the k nearest deleted genes

Selection mirrors the model's: the model reports the epoch that maximizes val Pearson, so
B2's rank and ridge and B3's k are the val-maximizing cells, and test is reported at those
cells. Every cell's val and test score is stored as well.

Two things differ from the oracle-family version and are deliberate. Strains with two
deletions (the 72 Sameith doubles) are kept, since the model's val set contains them; their
perturbation representation is the mean of the two genes' embeddings. And the per-feature
Pearson drops (near-)constant columns exactly as the training metric does.

Run from the repo root:
    python experiments/019-simb-multimodal/scripts/expression_baselines_split.py --split-seed 1
    python experiments/019-simb-multimodal/scripts/expression_baselines_split.py --split-seed 0 --fold-test-into-train
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import lmdb
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from expression_baselines import (  # noqa: E402
    EMBEDDINGS,
    KNN_GRID,
    RANK_GRID,
    RIDGE_GRID,
    _bilinear,
    _embedding_matrix,
)

from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402
from torchcell.utils.paths import experiment_results_dir  # noqa: E402

DATASET_TAG = "fig3_core"
EXPRESSION_LABEL = "expression_log2_ratio"


def per_feature_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Per-gene Pearson across strains over each gene's FINITE pairs, averaged over genes.

    The expression matrices are complete and this equals expression_baselines'
    function on them. The proteome matrix is not: a strain carries NaN for a protein it
    did not quantify (1,441 to 1,850 of 1,850 per strain), and the training metric scores
    each protein over its finite pairs, so this does too. Genes with fewer than 3 pairs
    or a (near-)constant column are dropped, as in the training metric.
    """
    ok_pair = np.isfinite(pred) & np.isfinite(true)
    n = ok_pair.sum(axis=0)
    p0 = np.where(ok_pair, pred, 0.0)
    t0 = np.where(ok_pair, true, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = p0.sum(axis=0) / np.maximum(n, 1)
        tm = t0.sum(axis=0) / np.maximum(n, 1)
    p = np.where(ok_pair, p0 - pm, 0.0)
    t = np.where(ok_pair, t0 - tm, 0.0)
    num = (p * t).sum(axis=0)
    den = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    ok = (den > 1e-8) & (n >= 3)
    if not ok.any():
        return 0.0
    return float((num[ok] / den[ok]).mean())


def nmse(pred: np.ndarray, true: np.ndarray, mu_fit: np.ndarray) -> float:
    """MSE over finite target entries, divided by their variance about the fit mean."""
    ok = np.isfinite(true) & np.isfinite(pred)
    num = float(((pred - true) ** 2)[ok].mean())
    den = float(((true - mu_fit) ** 2)[ok].mean())
    return num / den


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-seed", type=int, required=True)
    p.add_argument(
        "--fold-test-into-train",
        action="store_true",
        help="train on train+test against the same val set (the 90/10 arm); no test score",
    )
    # The proteome round (v14) reads the same baselines on the fig3_proteome build, whose
    # label is the log2 ratio of protein abundance to the HIS3 reference. The two
    # arguments travel together; results land in baselines_split_<tag>/ for any tag
    # other than fig3_core.
    p.add_argument("--dataset-tag", default=DATASET_TAG)
    p.add_argument(
        "--label",
        default=EXPRESSION_LABEL,
        help="phenotype label to predict (expression_log2_ratio or protein_abundance)",
    )
    return p.parse_args()


def _load_split(cache_dir: str, seed: int, label: str) -> dict[str, list[int]]:
    path = osp.join(cache_dir, f"index_details_seed_{seed}.json")
    if not osp.exists(path):
        raise FileNotFoundError(
            f"{path} missing: run make_split_indices.py --seeds {seed} first so the "
            "baseline reads the same partition the GPU round trains on"
        )
    with open(path) as f:
        details = json.load(f)
    return {
        split: list(details[split]["phenotype_label_index"][label]["indices"])
        for split in ("train", "val", "test")
    }


def _load_records(
    base: str, indices: list[int], label: str
) -> tuple[list[list[str]], np.ndarray, list[str]]:
    """Expression rows for the given record indices, in that order.

    A record is the deduplicated group the dataset stores under its index; the round's
    expression records each carry exactly one expression experiment (checked here, so a
    group that ever carries two fails loudly rather than being averaged silently).
    """
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    perts: list[list[str]] = []
    rows: list[np.ndarray] = []
    keys: list[str] | None = None
    with env.begin() as txn:
        for idx in indices:
            raw = txn.get(f"{idx}".encode())
            if raw is None:
                raise KeyError(f"record {idx} not in {base}")
            recs = json.loads(raw.decode())
            if isinstance(recs, dict):
                recs = [recs]
            expr = [
                r for r in recs if r["experiment"]["phenotype"]["label_name"] == label
            ]
            if len(expr) != 1:
                raise ValueError(f"record {idx} carries {len(expr)} {label} experiments")
            d = expr[0]["experiment"]["phenotype"][label]
            if keys is None:
                keys = sorted(d)
            if set(d) != set(keys):
                raise ValueError(f"record {idx} has a different gene key set")
            perts.append(
                [
                    p["systematic_gene_name"]
                    for p in expr[0]["experiment"]["genotype"]["perturbations"]
                ]
            )
            rows.append(np.array([d[k] for k in keys], dtype=np.float32))
    env.close()
    assert keys is not None
    return perts, np.stack(rows), keys


def _pert_matrix(
    perts: list[list[str]], emb: dict[str, np.ndarray], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean embedding over the deleted genes that have one; ``ok`` marks strains with >= 1."""
    out = np.zeros((len(perts), dim), dtype=np.float64)
    ok = np.zeros(len(perts), dtype=bool)
    for i, genes in enumerate(perts):
        vecs = [emb[g] for g in genes if g in emb]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
            ok[i] = True
    return out, ok


def main() -> None:
    args = parse_args()
    data_root = os.environ["DATA_ROOT"]
    base = osp.join(
        data_root, "data/torchcell/experiments/019-simb-multimodal", args.dataset_tag
    )
    split = _load_split(
        osp.join(base, "data_module_cache"), args.split_seed, args.label
    )
    if args.fold_test_into_train:
        split = {
            "train": split["train"] + split["test"],
            "val": split["val"],
            "test": [],
        }
    perts: dict[str, list[list[str]]] = {}
    y: dict[str, np.ndarray] = {}
    keys: list[str] | None = None
    for name, idx in split.items():
        if not idx:
            continue
        p, mat, k = _load_records(base, idx, args.label)
        if keys is None:
            keys = k
        elif k != keys:
            raise ValueError("gene key order differs between splits")
        perts[name], y[name] = p, mat.astype(np.float64)
    n_gene = y["train"].shape[1]
    eval_splits = [s for s in ("val", "test") if s in y]
    print(
        f"split seed {args.split_seed}"
        f"{' (test folded into train)' if args.fold_test_into_train else ''}: "
        + ", ".join(f"{s}={len(y[s])}" for s in y)
        + f"; genes={n_gene}"
    )

    # Per-gene TRAIN mean over finite entries; residuals keep NaN where the target is
    # unmeasured. The fits below see the residuals with NaN replaced by 0 (the per-gene
    # mean, the least-informative imputation) and every score runs on finite entries only.
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(y["train"], axis=0, keepdims=True)
    if not np.isfinite(mu).all():
        raise ValueError("a gene has no finite training value; cannot center it")
    r = {s: y[s] - mu for s in y}
    r_dense = {s: np.nan_to_num(r[s], nan=0.0) for s in y}
    frac_missing = {s: float(np.isnan(y[s]).mean()) for s in y}
    out: dict[str, object] = {
        "generated_by": "experiments/019-simb-multimodal/scripts/expression_baselines_split.py",
        "dataset_tag": args.dataset_tag,
        "label": args.label,
        "split": {
            "kind": "CellDataModule index_details_seed",
            "split_seed": args.split_seed,
            "fold_test_into_train": args.fold_test_into_train,
            "n_train": int(len(y["train"])),
            **{f"n_{s}": int(len(y[s])) for s in eval_splits},
            "n_gene": int(n_gene),
            "n_double_deletion": {
                s: int(sum(len(p) == 2 for p in perts[s])) for s in perts
            },
            "fraction_missing": frac_missing,
        },
    }

    for s in eval_splits:
        pred_b0 = np.repeat(mu, len(y[s]), axis=0)
        b0 = {
            "pearson_per_feature": per_feature_pearson(pred_b0, y[s]),
            "nmse": nmse(pred_b0, y[s], mu),
        }
        b1 = {
            "pearson_per_feature": per_feature_pearson(np.zeros_like(y[s]), y[s]),
            "nmse": nmse(np.zeros_like(y[s]), y[s], mu),
        }
        out[f"B0_per_gene_mean_{s}"] = b0
        out[f"B1_no_change_{s}"] = b1
        print(
            f"  {s}: B0 pf={b0['pearson_per_feature']:+.4f} nmse={b0['nmse']:.4f}; "
            f"B1 pf={b1['pearson_per_feature']:+.4f} nmse={b1['nmse']:.4f}"
        )
        if abs(b0["pearson_per_feature"]) > 1e-9:
            raise ValueError(
                f"B0 pearson_per_feature on {s} is {b0['pearson_per_feature']}"
            )
        if not 0.85 < b0["nmse"] < 1.2:
            raise ValueError(f"B0 nmse on {s} is {b0['nmse']}, expected ~1.0")

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

    b2_by_emb: dict[str, object] = {}
    b3_by_emb: dict[str, object] = {}
    for emb_name in EMBEDDINGS:
        emb = _embedding_matrix(EMBEDDINGS[emb_name], data_root, genome, graph)
        dim = len(next(iter(emb.values())))
        p: dict[str, np.ndarray] = {}
        ok: dict[str, np.ndarray] = {}
        for s in y:
            p[s], ok[s] = _pert_matrix(perts[s], emb, dim)
        if ok["train"].sum() < 50 or any(ok[s].sum() < 20 for s in eval_splits):
            print(f"  {emb_name}: too few strains covered; skipped")
            continue
        pm = p["train"][ok["train"]].mean(axis=0, keepdims=True)
        ps = p["train"][ok["train"]].std(axis=0, keepdims=True) + 1e-8
        for s in y:
            p[s] = (p[s] - pm) / ps
        r_tr, p_tr = r_dense["train"][ok["train"]], p["train"][ok["train"]]

        # ---- B2: every (rank, ridge) cell scored on every eval split -------------------
        cells: list[dict[str, object]] = []
        for k_rank in RANK_GRID:
            if k_rank > min(r_tr.shape) - 1:
                continue
            for ridge in RIDGE_GRID:
                cell: dict[str, object] = {"k_gene": k_rank, "ridge": ridge}
                for s in eval_splits:
                    r_hat = _bilinear(r_tr, r_dense[s], p_tr, p[s], k_rank, ridge)
                    cell[f"{s}_pearson_per_feature"] = per_feature_pearson(
                        r_hat[ok[s]], r[s][ok[s]]
                    )
                    cell[f"{s}_nmse"] = nmse(r_hat[ok[s]] + mu, y[s][ok[s]], mu)
                cells.append(cell)
        best2 = max(cells, key=lambda c: float(c["val_pearson_per_feature"]))
        b2_by_emb[emb_name] = {
            "emb_dim": int(dim),
            "selected_on_val": best2,
            "cells": cells,
            "n_scored": {s: int(ok[s].sum()) for s in eval_splits},
        }

        # ---- B3: k nearest TRAIN deleted genes by cosine, mean train residual -----------
        e_tr = p_tr / (np.linalg.norm(p_tr, axis=1, keepdims=True) + 1e-12)
        cells3: list[dict[str, object]] = []
        for k in KNN_GRID:
            cell = {"k": k}
            for s in eval_splits:
                e_s = p[s] / (np.linalg.norm(p[s], axis=1, keepdims=True) + 1e-12)
                nn = np.argsort(-(e_s @ e_tr.T), axis=1)[:, :k]
                # Mean over the neighbours that measured the gene; a gene no neighbour
                # measured is the per-gene mean (residual 0), the same imputation as B2's.
                r_nn = r["train"][ok["train"]][nn]
                with np.errstate(invalid="ignore"):
                    preds = np.nan_to_num(np.nanmean(r_nn, axis=1), nan=0.0)
                cell[f"{s}_pearson_per_feature"] = per_feature_pearson(
                    preds[ok[s]], r[s][ok[s]]
                )
                cell[f"{s}_nmse"] = nmse(preds[ok[s]] + mu, y[s][ok[s]], mu)
            cells3.append(cell)
        best3 = max(cells3, key=lambda c: float(c["val_pearson_per_feature"]))
        b3_by_emb[emb_name] = {
            "selected_on_val": best3,
            "cells": cells3,
            "n_scored": {s: int(ok[s].sum()) for s in eval_splits},
        }
        line = f"  {emb_name:<28} B2 val {best2['val_pearson_per_feature']:+.4f}"
        if "test" in eval_splits:
            line += f" test {best2['test_pearson_per_feature']:+.4f}"
        line += f" (k={best2['k_gene']}, ridge={best2['ridge']}) | B3 val {best3['val_pearson_per_feature']:+.4f}"
        if "test" in eval_splits:
            line += f" test {best3['test_pearson_per_feature']:+.4f}"
        line += f" (k={best3['k']})"
        print(line)

    def _top(d: dict) -> str:
        return max(
            d, key=lambda k: float(d[k]["selected_on_val"]["val_pearson_per_feature"])
        )

    out["B2_bilinear"] = {"by_embedding": b2_by_emb, "best_embedding": _top(b2_by_emb)}
    out["B3_neighbor_average"] = {
        "by_embedding": b3_by_emb,
        "best_embedding": _top(b3_by_emb),
    }

    tag = f"seed{args.split_seed}" + ("_fold90" if args.fold_test_into_train else "")
    dst_dir = osp.join(
        experiment_results_dir("019-simb-multimodal", __file__),
        "expression_baselines_split"
        if args.dataset_tag == DATASET_TAG
        else f"baselines_split_{args.dataset_tag}",
    )
    os.makedirs(dst_dir, exist_ok=True)
    dst = osp.join(dst_dir, f"{tag}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
