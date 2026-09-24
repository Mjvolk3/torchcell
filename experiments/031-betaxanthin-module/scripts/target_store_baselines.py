# experiments/031-betaxanthin-module/scripts/target_store_baselines.py
# [[experiments.031-betaxanthin-module.scripts.target_store_baselines]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_store_baselines

r"""CPU baselines over the metabolic-store representations, on the trainer's pinned split.

The transformer arms of 031 and 032 read a per-gene deletion-flux vector through a
zero-initialized affine beside the backbone. Whether that vector carries the target on its
own, without the backbone, is a CPU question: represent each deleted gene by the store row
and fit the same ladder 027's ``linear_baselines_bx.py`` fits over the protein-language
embedding (ridge or rank-swept bilinear, cosine nearest neighbors), plus a random forest,
which is the Flux Cone Learning form (a forest over flux features) on one point per
deletion instead of a sample of the cone.

Representations (each a row per deleted gene, ``log(1+|v|)`` for flux columns, standardized
on the training genes; a gene the store does not cover reads the store's wild-type row and
is counted, since a deletion outside the model is a deletion the model says does nothing):

* the parsimonious store the store-affine arms read (betaxanthin: the growth vector, the
  production vector and the total demand of the GLPK build, 8,355 columns; amino acids:
  the HiGHS build on SM, 4,131), and for betaxanthin the growth block alone;
* the HiGHS parsimonious rebuild (betaxanthin) and the linear MOMA store
  (``target_physics_store.py``);
* the six gene-rule counts (``isozyme_affine_store.py``) and the six flux statistics of
  the parsimonious build (growth, production, ratios, reactions knocked out, distance);
* ``prot_T5_all``, the reference the 027 ladder used, and its concatenation with the
  parsimonious store.

Split, targets, standardization, selection (validation per-feature Pearson) and the pinned
test statistic are the 027 ladder's, imported from it, so a number here sits in the same
column as the 027 baselines and the transformer arms. Writes
``results/store_baselines_<target>.json`` in the target's study. Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_store_baselines.py \
        --target betaxanthin --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "scripts"))
import train_bx  # noqa: E402
from cbm_scoring import assess  # noqa: E402
from linear_baselines_bx import (  # noqa: E402
    KNN_GRID,
    RANK_GRID,
    RIDGE_GRID,
    TARGETS,
    bilinear,
    embedding_matrix,
    nmse,
    open_dataset,
    per_feature_pearson,
    read_targets,
    record_genes,
)

from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

STUDY = {"betaxanthin": "031-betaxanthin-module", "amino_acids": "032-amino-acid-module"}
#: (store file stem, column slice or None, flux-encoded or not) per representation.
REPRESENTATIONS: dict[str, dict[str, Any]] = {
    "betaxanthin": {
        "pfba_glpk_full": {"store": "YPD_bx", "block": "v_deletion", "flux": True},
        "pfba_glpk_growth": {
            "store": "YPD_bx", "block": "v_deletion", "flux": True, "columns": 4177
        },
        "pfba_highs": {"store": "YPD_bx_pfba", "block": "v_deletion", "flux": True},
        "moma": {"store": "YPD_bx_moma", "block": "v_deletion", "flux": True},
        "gpr_counts": {"store": "YPD_bx_gpronly", "block": "v_deletion", "flux": True},
        "flux_stats": {"store": "YPD_bx", "block": "gene_flux_stats", "flux": False},
        # The metabolite side (target_driving_force_store.py): concentrations in mM of
        # the max-min driving-force solution, with the second law over every priced
        # reaction (mdf) or over those under 100 kJ/mol (mdfcap100); and the sampled cone
        # (target_sampled_flux_store.py), the mean over draws.
        "mdf_conc": {"store": "YPD_bx_mdf", "block": "v_deletion", "flux": True},
        "mdfcap100_conc": {"store": "YPD_bx_mdfcap100", "block": "v_deletion", "flux": True},
        "sampled_mean": {"store": "YPD_bx_sampledmean", "block": "v_deletion", "flux": True},
    },
    "amino_acids": {
        "pfba_highs": {"store": "SM_gem", "block": "v_deletion", "flux": True},
        "moma": {"store": "SM_gem_moma", "block": "v_deletion", "flux": True},
        "gpr_counts": {"store": "SM_gem_gpronly", "block": "v_deletion", "flux": True},
        "flux_stats": {"store": "SM_gem", "block": "gene_flux_stats", "flux": False},
        "mdf_conc": {"store": "SM_gem_mdf", "block": "v_deletion", "flux": True},
        "mdfcap100_conc": {"store": "SM_gem_mdfcap100", "block": "v_deletion", "flux": True},
        "sampled_mean": {"store": "SM_gem_sampledmean", "block": "v_deletion", "flux": True},
    },
}
PRIMARY_STORE = {"betaxanthin": "pfba_glpk_full", "amino_acids": "pfba_highs"}
FOREST_LEAF_GRID = [1, 5, 20]
FOREST_TREES = 300
TOP_K = 50
TOP_FRAC = 0.1


def load_store(spec: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Gene -> encoded row, the encoded wild-type row, the column count."""
    z = np.load(osp.join(TABLE_DIR, f"ess_solve_encodings_{spec['store']}.npz"))
    block = z[spec["block"]].astype(np.float64)
    if spec["block"] == "v_deletion":
        wild = z["v_wild_type"].astype(np.float64)
    else:
        # The statistics of an unchanged genotype: growth and production equal the wild
        # type's, every ratio 1, every count, flux magnitude and distance of the deleted
        # gene 0 (the two stores name different statistics; the rule covers both).
        names = [str(s) for s in z["stat_names"]]
        wild = np.zeros(block.shape[1])
        p = json.loads(str(z["provenance"]))
        for i, name in enumerate(names):
            if name.endswith("ratio") or name == "btx_ratio_to_wild_type":
                wild[i] = 1.0
            elif name == "growth":
                wild[i] = p["wild_type_growth"]
            elif name == "btx_at_own_growth_floor":
                wild[i] = p["wild_type_btx_at_own_growth_floor"]
    if "columns" in spec:
        block, wild = block[:, : spec["columns"]], wild[: spec["columns"]]
    if spec["flux"]:
        block, wild = np.log1p(np.abs(block)), np.log1p(np.abs(wild))
    genes = [str(g) for g in z["genes"]]
    return {g: block[i] for i, g in enumerate(genes)}, wild, int(block.shape[1])


#: The arm whose dump the pinned targets are checked against, per target.
REFERENCE_ARM = {
    "betaxanthin": "w140_bxdelstore_h12l1p1_e200_s2700",
    "amino_acids": "w141_aadel_h12l1p1_e200_s2700",
}


def arm_targets(target: str) -> dict[str, Any]:
    """{gene -> pinned target} from the reference arm's dump, resolved through its result."""
    result_path = osp.join(
        EXPERIMENT_ROOT, STUDY[target], "results", f"{REFERENCE_ARM[target]}.json"
    )
    with open(result_path) as fh:
        result = json.load(fh)
    with open(result["test"]["dump_path"]) as fh:
        dump = json.load(fh)
    assert dump["cell"] == result["cell"], result_path
    return {
        rec["genes"][0]: rec["target"]
        for rec in dump["predictions"]
        if rec["pinned"] and len(rec["genes"]) == 1 and rec["target"] is not None
    }


def top50_enrichment(pred: np.ndarray, true: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(pred) & np.isfinite(true)
    p, t = pred[finite], true[finite]
    n_high = max(1, int(round(TOP_FRAC * t.size)))
    high = set(np.argsort(-t)[:n_high].tolist())
    hits = sum(1 for i in np.argsort(-p)[:TOP_K].tolist() if i in high)
    return float((hits / TOP_K) / (n_high / t.size)), int(hits)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True, choices=sorted(STUDY))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from sklearn.ensemble import RandomForestRegressor

    started = time.time()
    spec = TARGETS[args.target]
    columns = spec["columns"]
    rng = np.random.default_rng(args.seed)

    cfg = train_bx.load_config()
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    dataset = open_dataset(spec, genome, graph)
    pinned, pinned_report = train_bx.resolve_pinned_test(dataset, cfg)
    dm = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(DATA_ROOT, spec["dataset_subdir"], "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=cfg.model.batch_size,
        random_seed=cfg.data.split_seed,
        num_workers=0,
        pin_memory=False,
        prefetch=False,
        follow_batch=["perturbation_indices", "deletion_indices", "phenotype_values"],
        pinned_test_indices=pinned,
    )
    dm.setup()
    base = osp.join(DATA_ROOT, spec["dataset_subdir"])
    splits = {
        name: [int(i) for i in getattr(dm, f"{name}_dataset").indices]
        for name in ("train", "val", "test")
    }
    idx_to_genes = record_genes(dataset)
    with open(osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)) as fh:
        pinned_genes = set(json.load(fh)["split"]["test"])

    y: dict[str, np.ndarray] = {}
    perts: dict[str, list[list[str]]] = {}
    keep_pinned: dict[str, np.ndarray] = {}
    for name, indices in splits.items():
        values = read_targets(base, indices, spec["label"], columns)
        genes = [idx_to_genes.get(i, []) for i in indices]
        has = np.isfinite(values).any(axis=1) & np.array([len(g) > 0 for g in genes])
        y[name] = values[has]
        perts[name] = [g for g, k in zip(genes, has, strict=True) if k]
        keep_pinned[name] = np.array(
            [any(g in pinned_genes for g in gs) for gs in perts[name]]
        )
        print(
            f"  {name}: {int(has.sum())} genotypes with a measurement, "
            f"{int(keep_pinned[name].sum())} pinned",
            flush=True,
        )
    obs = {name: np.isfinite(y[name]) for name in y}
    # The identity check the 027 ladder makes, against THIS study's reference arm: its
    # dump is resolved through the result file, as the readouts resolve it. (The 027
    # wave-7 directory now also holds dumps with log-scaled targets, which the ladder's
    # name-order pick lands on for the amino acids.)
    theirs = arm_targets(args.target)
    mine = {
        perts["test"][i][0]: y["test"][i]
        for i in range(len(perts["test"]))
        if keep_pinned["test"][i] and len(perts["test"][i]) == 1
    }
    shared = sorted(set(mine) & set(theirs))
    assert len(shared) >= 500, f"only {len(shared)} pinned genes match the dump"
    for gene in shared:
        theirs_vec = np.array(
            [np.nan if v is None else float(v) for v in np.atleast_1d(theirs[gene])]
        )
        assert np.allclose(mine[gene], theirs_vec, equal_nan=True, atol=1e-5), gene
    print(f"  identity check: {len(shared)} pinned targets match the dump", flush=True)

    mu = np.array(
        [
            y["train"][obs["train"][:, f], f].mean() if obs["train"][:, f].any() else 0.0
            for f in range(len(columns))
        ]
    )[None, :]
    resid_dense = {name: np.where(obs[name], y[name] - mu, 0.0) for name in y}

    def score(pred: np.ndarray, name: str, pinned_only: bool, covered: np.ndarray,
              full_stats: bool = False) -> dict[str, Any]:
        sel = (keep_pinned[name] if pinned_only else np.ones(len(y[name]), bool)) & covered
        n_stat = 2000 if full_stats else 0
        block: dict[str, Any] = {
            "per_feature_pearson": per_feature_pearson(pred[sel], y[name][sel], obs[name][sel]),
            "nmse": nmse(pred[sel], y[name][sel], mu, obs[name][sel]),
            "n": int(sel.sum()),
        }
        rhos = []
        per_feature: dict[str, Any] = {}
        for f, column in enumerate(columns):
            m = sel & obs[name][:, f]
            cell = assess(pred[m, f], y[name][m, f], rng, n_perm=n_stat, n_boot=n_stat)
            per_feature[column] = cell
            rhos.append(0.0 if cell["spearman"] is None else float(cell["spearman"]))
        block["mean_spearman"] = float(np.mean(rhos))
        block["per_feature"] = per_feature
        if args.target == "betaxanthin":
            enrichment, hits = top50_enrichment(pred[sel, 0], y[name][sel, 0])
            block["top50_enrichment"] = enrichment
            block["top50_hits"] = hits
        return block

    # ---- representations --------------------------------------------------------------
    reps: dict[str, dict[str, Any]] = {}
    not_built: list[str] = []
    for rep_name, rep_spec in REPRESENTATIONS[args.target].items():
        if not osp.isfile(osp.join(TABLE_DIR, f"ess_solve_encodings_{rep_spec['store']}.npz")):
            # A store that has not been built yet is reported as such, not filled.
            not_built.append(rep_name)
            print(f"  {rep_name}: store {rep_spec['store']} not built; skipped", flush=True)
            continue
        rows, wild, dim = load_store(rep_spec)
        p: dict[str, np.ndarray] = {}
        filled: dict[str, int] = {}
        for name in y:
            mat = np.zeros((len(perts[name]), dim))
            n_fill = 0
            for i, genes in enumerate(perts[name]):
                vecs = [rows[g] for g in genes if g in rows]
                if vecs:
                    mat[i] = np.mean(vecs, axis=0)
                else:
                    mat[i] = wild
                    n_fill += 1
            p[name], filled[name] = mat, n_fill
        reps[rep_name] = {
            "p": p,
            "ok": {name: np.ones(len(perts[name]), bool) for name in y},
            "dim": dim,
            "filled_wild_type": filled,
            "store": rep_spec["store"],
        }
    emb = embedding_matrix(("prot_T5_all",), DATA_ROOT, genome, graph)
    dim = len(next(iter(emb.values())))
    p_emb: dict[str, np.ndarray] = {}
    ok_emb: dict[str, np.ndarray] = {}
    for name in y:
        mat = np.zeros((len(perts[name]), dim))
        good = np.zeros(len(perts[name]), bool)
        for i, genes in enumerate(perts[name]):
            vecs = [emb[g] for g in genes if g in emb]
            if vecs:
                mat[i], good[i] = np.mean(vecs, axis=0), True
        p_emb[name], ok_emb[name] = mat, good
    reps["prot_T5_all"] = {
        "p": p_emb, "ok": ok_emb, "dim": dim, "filled_wild_type": {}, "store": None
    }
    primary = reps[PRIMARY_STORE[args.target]]
    reps[f"{PRIMARY_STORE[args.target]}+prot_T5_all"] = {
        "p": {n: np.concatenate([primary["p"][n], p_emb[n]], axis=1) for n in y},
        "ok": ok_emb,
        "dim": primary["dim"] + dim,
        "filled_wild_type": primary["filled_wild_type"],
        "store": primary["store"],
    }

    ranks = [k for k in RANK_GRID if k <= len(columns)]
    payload: dict[str, Any] = {
        "generated_by": "experiments/031-betaxanthin-module/scripts/target_store_baselines.py",
        "target": args.target,
        "columns": columns,
        "pinned_split": pinned_report,
        "n": {name: int(len(y[name])) for name in y},
        "n_pinned": {name: int(keep_pinned[name].sum()) for name in y},
        "selection": "validation per-feature Pearson, as the 027 ladder",
        "statistic": (
            "pinned test mean Spearman over columns; betaxanthin also the top-50 "
            "enrichment of true top-10-percent producers among the 50 highest predictions"
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "not_built": not_built,
        "representations": {},
    }

    for rep_name, rep in reps.items():
        p, ok = rep["p"], rep["ok"]
        pm = p["train"][ok["train"]].mean(axis=0, keepdims=True)
        ps = p["train"][ok["train"]].std(axis=0, keepdims=True) + 1e-8
        p = {n: (p[n] - pm) / ps for n in p}
        r_tr, p_tr = resid_dense["train"][ok["train"]], p["train"][ok["train"]]
        out: dict[str, Any] = {
            "dim": rep["dim"],
            "store": rep["store"],
            "filled_wild_type": rep["filled_wild_type"],
            "n_scored": {s: int(ok[s].sum()) for s in ("val", "test")},
        }

        def select(cells: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
            # The validation criterion is rounded to 1e-6 and ties go to the LATER
            # grid point (the stronger ridge, the wider neighborhood, the larger leaf):
            # the validation target is 98 percent one value on betaxanthin, so an
            # unrounded argmax flipped between runs on 1e-13 float noise (2026-09-23).
            best = max(
                enumerate(cells),
                key=lambda ic: (round(float(ic[1]["val"]["per_feature_pearson"]), 6), ic[0]),
            )[1]
            return {
                "selected_on_val": {
                    **{k: best[k] for k in keys},
                    "val": best["val"],
                    "test_pinned": score(best["_pred"], "test", True, ok["test"], True),
                },
                "cells": [
                    {
                        **{k: c[k] for k in keys},
                        "val_per_feature_pearson": c["val"]["per_feature_pearson"],
                        "test_pinned_mean_spearman": c["test_pinned"]["mean_spearman"],
                    }
                    for c in cells
                ],
            }

        def evaluate(pred_fn: Any, params: dict[str, Any]) -> dict[str, Any]:
            cell: dict[str, Any] = dict(params)
            for name in ("val", "test"):
                pred = np.zeros_like(y[name])
                pred[ok[name]] = pred_fn(p[name][ok[name]])
                # Predictions within 1e-9 are one rank: the no-effect deletions read
                # rows that differ from the wild type's by float noise, and their
                # unrounded predictions reordered inside the tie between runs.
                pred = np.round(pred + mu, 9)
                cell[name] = score(pred, name, False, ok[name])
                if name == "test":
                    cell["test_pinned"] = score(pred, name, True, ok[name])
                    cell["_pred"] = pred
            return cell

        # ridge / bilinear
        cells = [
            evaluate(lambda q, k=k, r=r: bilinear(r_tr, p_tr, q, k, r), {"k_gene": k, "ridge": r})
            for k in ranks
            if k <= min(r_tr.shape)
            for r in RIDGE_GRID
        ]
        out["ridge"] = select(cells, ["k_gene", "ridge"])
        # cosine nearest neighbors
        e_tr = p_tr / (np.linalg.norm(p_tr, axis=1, keepdims=True) + 1e-12)

        def knn(q: np.ndarray, k: int) -> np.ndarray:
            # Every training row tied with the k-th nearest (cosine within 1e-9) is
            # averaged in: a query on the wild-type row has about two thousand
            # identical training rows, and taking k of them by argsort order made the
            # prediction an arbitrary subset that changed between runs.
            e_q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims = e_q @ e_tr.T
            kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1 : k]
            weights = (sims >= kth - 1e-9).astype(float)
            return (weights @ r_tr) / weights.sum(axis=1, keepdims=True)

        cells = [evaluate(lambda q, k=k: knn(q, k), {"k": k}) for k in KNN_GRID]
        out["knn"] = select(cells, ["k"])
        # random forest, the Flux Cone Learning form on one point per deletion
        cells = []
        for leaf in FOREST_LEAF_GRID:
            forest = RandomForestRegressor(
                n_estimators=FOREST_TREES,
                max_features="sqrt",
                min_samples_leaf=leaf,
                n_jobs=args.threads,
                random_state=args.seed,
            ).fit(p_tr, r_tr if r_tr.shape[1] > 1 else r_tr[:, 0])
            cells.append(
                evaluate(
                    lambda q, f=forest: np.atleast_2d(f.predict(q)).reshape(len(q), -1),
                    {"min_samples_leaf": leaf, "n_estimators": FOREST_TREES},
                )
            )
        out["forest"] = select(cells, ["min_samples_leaf", "n_estimators"])
        payload["representations"][rep_name] = out
        line = f"  {rep_name:<28s} dim {rep['dim']:>5d}"
        for model in ("ridge", "knn", "forest"):
            tp = out[model]["selected_on_val"]["test_pinned"]
            line += f" | {model} rho {tp['mean_spearman']:+.4f}"
            if "top50_enrichment" in tp:
                line += f" top50 {tp['top50_enrichment']:.2f}"
        print(line, flush=True)

    payload["seconds"] = time.time() - started
    results_dir = osp.join(EXPERIMENT_ROOT, STUDY[args.target], "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or osp.join(results_dir, f"store_baselines_{args.target}.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out_path} in {payload['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
