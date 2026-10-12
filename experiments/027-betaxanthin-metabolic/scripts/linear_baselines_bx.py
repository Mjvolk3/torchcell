# experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx.py
# [[experiments.027-betaxanthin-metabolic.cpu-baselines-2026-09-14]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx
"""The linear baselines for the 027 metabolic targets, on the partition the arms train on.

WHICH TEMPLATE, AND WHY NOT THE OTHER ONE. The 010/025 additive ladder builds a one-hot gene
incidence plus a pair incidence over pairs recurring in train. The Cachera screen is SINGLE
deletions and every test record names a distinct gene, so each gene appears in exactly one
record: the gene-only ridge memorizes train and emits the intercept for every unseen test
gene, and the pair vocabulary is empty. The ladder is uninformative here by construction.

019's expression baselines were built for exactly this regime and say so: every validation
strain is an unseen perturbation, which is why they represent a perturbation by an EXTERNAL
embedding of the deleted gene rather than by an identity code. This script is that ladder,
field for field:

  B0  train mean                      B2  Y ~ G W P^T + b, ridge, rank-swept
  B1  no change (zeros)               B3  mean fit residual of the k nearest deleted genes

over the same four embedding sets (``prot_T5_all``, ``calm``, ``species_lm_5p_3p`` =
``fudt_upstream`` + ``fudt_downstream``, ``normalized_chrom_pathways``), with fit-split
standardization, 019's ``ok`` mask so a gene without an embedding is excluded rather than
zero-filled, and 019's hard self-check that B0 scores exactly 0 on per-feature Pearson and
that its NMSE lands in (0.9, 1.15).

TWO THINGS DIFFER FROM 019 AND ARE DELIBERATE.

* **B2 collapses to ridge on a scalar target.** 019's B2 is a rank-k bilinear because its
  target is a 6,169-column expression profile. Betaxanthin and isobutanol are scalars, so
  the gene basis is one-dimensional and the model IS plain ridge from the embedding to the
  scalar. The amino-acid target is 19 columns, so the bilinear form is kept there with the
  rank grid capped at 19.
* **The target matrix has holes.** 019's expression matrix is dense; the amino-acid panel is
  not measured on every genotype. Missing entries are set to the fit-column mean (zero after
  centering) for the SVD and the ridge solve, and scoring uses observed entries only. Stated
  rather than hidden, because an imputed cell is not a measurement.

THE SPLIT IS THE TRAINER'S, RESOLVED THE TRAINER'S WAY. The datamodule cache holds TWO pinned
split files with identical counts, so this script does not glob for one: it calls
``train_bx.resolve_pinned_test`` and builds the same ``CellDataModule`` the trainer builds
(``split_indices=["phenotype_label_index", "perturbation_count_index"]``,
``random_seed=cfg.data.split_seed``, ``pinned_test_indices=pinned``), which resolves the hash
identically and fails loudly on a drift. The reported statistic is on the PINNED test genes,
the same set every wave-7 dump scores, and that identity is asserted against the dumps.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \\
        experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx.py \\
        --target betaxanthin
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Any

import lmdb
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from cbm_scoring import assess  # noqa: E402

import train_bx  # noqa: E402

from torchcell.data import (  # noqa: E402
    DeletionKeyedGenotypeAggregator,
    MeanExperimentDeduplicator,
)
from torchcell.data.graph_processor import Perturbation  # noqa: E402
from torchcell.data.neo4j_cell import Neo4jCellDataset  # noqa: E402
from torchcell.datamodules import CellDataModule  # noqa: E402
from torchcell.datasets.node_embedding_builder import NodeEmbeddingBuilder  # noqa: E402
from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
DUMP_DIR = osp.join(DATA_ROOT, "test-predictions-027-wave7")

#: 019's four perturbation representations, unchanged.
EMBEDDINGS: dict[str, tuple[str, ...]] = {
    "prot_T5_all": ("prot_T5_all",),
    "calm": ("calm",),
    "species_lm_5p_3p": ("fudt_upstream", "fudt_downstream"),
    "normalized_chrom_pathways": ("normalized_chrom_pathways",),
}
#: 019's grids. 19 is added so the amino-acid bilinear can reach full rank; everything above
#: 19 is dropped for that target because the gene basis cannot exceed the column count.
RANK_GRID = [1, 2, 4, 8, 16, 19, 32, 33, 64, 128, 256]
RIDGE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
KNN_GRID = [1, 3, 5, 10, 25]

AA_KEYS = [
    "alanine", "arginine", "asparagine", "aspartate", "glutamate", "glutamine",
    "glycine", "histidine", "isoleucine", "leucine", "lysine", "methionine",
    "phenylalanine", "proline", "serine", "threonine", "tryptophan", "tyrosine",
    "valine",
]

TARGETS: dict[str, dict[str, Any]] = {
    "betaxanthin": {
        "dataset_subdir": "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer",
        "query": "019-simb-multimodal/queries/fig6_pigment_transfer.cql",
        "label": "metabolite_level",
        "columns": ["betaxanthin"],
        "dump_head": "betaxanthin",
    },
    "amino_acids": {
        "dataset_subdir": "data/torchcell/experiments/019-simb-multimodal/fig6_pigment_transfer",
        "query": "019-simb-multimodal/queries/fig6_pigment_transfer.cql",
        "label": "metabolite_level",
        "columns": AA_KEYS,
        "dump_head": "mulleder19",
    },
    "isobutanol": {
        # NOT fig6_pigment_transfer: that build carries no isobutanol column. MEASURED in
        # fig6_build: 4,244 records hold a width-20 metabolite_level (the Mulleder 19 plus
        # isobutanol, merged by the deletion-keyed aggregator) and 189 hold a width-1
        # isobutanol-only record, so 4,433 genotypes carry the Lopez biosensor readout.
        "dataset_subdir": "data/torchcell/experiments/019-simb-multimodal/fig6_build",
        "query": "019-simb-multimodal/queries/fig6_build.cql",
        "label": "metabolite_level",
        "columns": ["isobutanol"],
        "dump_head": None,
    },
}


# --------------------------------------------------------------------------------------
# 019 helpers, copied verbatim in behavior from experiments/019-simb-multimodal/scripts/
# expression_baselines.py so the two ladders cannot silently disagree.
# --------------------------------------------------------------------------------------
def per_feature_pearson(pred: np.ndarray, true: np.ndarray, obs: np.ndarray) -> float:
    """Correlate each column across strains, average over columns, on observed entries.

    Columns whose prediction or target is (near-)constant have an undefined correlation and
    are DROPPED rather than counted as zero, which is why a constant baseline returns 0.0.
    """
    out: list[float] = []
    for f in range(true.shape[1]):
        m = obs[:, f]
        if m.sum() < 3:
            continue
        p = pred[m, f] - pred[m, f].mean()
        t = true[m, f] - true[m, f].mean()
        den = np.linalg.norm(p) * np.linalg.norm(t)
        if den > 1e-8:
            out.append(float((p * t).sum() / den))
    return float(np.mean(out)) if out else 0.0


def nmse(
    pred: np.ndarray, true: np.ndarray, mu_fit: np.ndarray, obs: np.ndarray
) -> float:
    """MSE divided by the target's variance about the FIT column mean, observed entries."""
    num = float((((pred - true) ** 2)[obs]).mean())
    den = float((((true - mu_fit) ** 2)[obs]).mean())
    return num / den


def bilinear(
    r_fit: np.ndarray,
    p_fit: np.ndarray,
    p_eval: np.ndarray,
    k_gene: int,
    ridge: float,
) -> np.ndarray:
    """Fit ``Y ~ G W P^T`` on fit residuals, predict eval residuals.

    G is the top-``k_gene`` right singular vectors of the fit residuals; W is the ridge
    solution of ``R_fit G ~ P_fit W``; the prediction is ``(P_eval W) G^T``. On a
    one-column target G is 1x1 and this is plain ridge.
    """
    g_basis = np.linalg.svd(r_fit, full_matrices=False)[2][:k_gene].T
    c_fit = r_fit @ g_basis
    ptp = p_fit.T @ p_fit
    w = np.linalg.solve(ptp + ridge * np.eye(ptp.shape[0]), p_fit.T @ c_fit)
    return (p_eval @ w) @ g_basis.T


def embedding_matrix(
    names: tuple[str, ...], data_root: str, genome: Any, graph: Any
) -> dict[str, np.ndarray]:
    """Gene -> concatenated embedding vector, 019's construction."""
    per_gene: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for name in names:
        built = NodeEmbeddingBuilder.build(
            embedding_names=[name], data_root=data_root, genome=genome, graph=graph
        )
        for item in built[name]:
            vec = torch.cat([t.flatten() for t in item.embeddings.values()]).cpu().numpy()
            per_gene.setdefault(item.id, []).append(vec)
            counts[item.id] = counts.get(item.id, 0) + 1
    return {
        g: np.concatenate(v).astype(np.float32)
        for g, v in per_gene.items()
        if counts[g] == len(names)
    }


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def open_dataset(spec: dict[str, Any], genome: Any, graph: Any) -> Neo4jCellDataset:
    """The 027 dataset, opened with the nine-graph prot_T5 backbone. Never rebuilds."""
    with open(osp.join(EXPERIMENT_ROOT, spec["query"])) as fh:
        query = fh.read()
    return Neo4jCellDataset(
        root=osp.join(DATA_ROOT, spec["dataset_subdir"]),
        query=query,
        gene_set=genome.gene_set,
        graphs=build_gene_multigraph(graph=graph, graph_names=train_bx.STRONG_GRAPHS),
        incidence_graphs=None,
        node_embeddings=NodeEmbeddingBuilder.build(
            embedding_names=list(train_bx.STRONG_NODE_EMBEDDINGS),
            data_root=DATA_ROOT,
            genome=genome,
            graph=graph,
        ),
        converter=None,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=DeletionKeyedGenotypeAggregator,
        graph_processor=Perturbation(),
    )


def read_targets(
    base: str, indices: list[int], label: str, columns: list[str]
) -> np.ndarray:
    """``[len(indices), len(columns)]`` of measured values, NaN where not measured.

    A record's experiments are scanned for the first ``label`` phenotype carrying each
    requested key. This is what lets the deletion-keyed aggregator's merged width-20
    amino-acid-plus-isobutanol record and a width-1 isobutanol-only record be read by one
    rule.
    """
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    out = np.full((len(indices), len(columns)), np.nan)
    with env.begin() as txn:
        for i, idx in enumerate(indices):
            raw = txn.get(f"{idx}".encode())
            if raw is None:
                raise KeyError(f"record {idx} not in {base}")
            recs = json.loads(raw.decode())
            if isinstance(recs, dict):
                recs = [recs]
            for rec in recs:
                phenotype = rec["experiment"]["phenotype"]
                if phenotype["label_name"] != label:
                    continue
                values = phenotype[label]
                if not isinstance(values, dict):
                    continue
                for f, key in enumerate(columns):
                    if key in values and np.isnan(out[i, f]):
                        out[i, f] = float(values[key])
    env.close()
    return out


def record_genes(dataset: Any) -> dict[int, list[str]]:
    """record index -> its DELETION genes, the index the trainer's dump keys on."""
    out: dict[int, list[str]] = {}
    for gene, rows in dataset.is_any_deletion_gene_index.items():
        for row in rows:
            out.setdefault(int(row), []).append(str(gene))
    return out


def dump_targets(head: str) -> dict[str, Any]:
    """{gene -> pinned target} from the wave-7 ``pooled`` dumps, for the identity check."""
    # Selected by CONTENT, not by file name: since 2026-09-16 a dump is named after its
    # result file, so the arm no longer appears as a prefix.
    for name in sorted(os.listdir(DUMP_DIR)):
        if not name.endswith(".json"):
            continue
        with open(osp.join(DUMP_DIR, name)) as fh:
            dump = json.load(fh)
        if dump["head"] != head or dump["cell"]["arm"] != "pooled":
            continue
        return {
            rec["genes"][0]: rec["target"]
            for rec in dump["predictions"]
            if rec["pinned"] and len(rec["genes"]) == 1 and rec["target"] is not None
        }
    raise FileNotFoundError(f"no pooled {head} dump under {DUMP_DIR}")


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target", required=True, choices=sorted(TARGETS), help="which phenotype"
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
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
    print(f"dataset: {len(dataset)} aggregated genotypes", flush=True)

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

    with open(
        osp.join(EXPERIMENT_ROOT, cfg.data.pinned_test_split_file)
    ) as fh:
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
            f"  {name}: {len(indices)} records -> {int(has.sum())} with a measurement, "
            f"{int(keep_pinned[name].sum())} of them pinned",
            flush=True,
        )
    obs = {name: np.isfinite(y[name]) for name in y}

    # IDENTITY CHECK. The pinned test targets this script reads from the LMDB must be the
    # ones the trainer dumped, or baseline and model are scoring different numbers.
    if spec["dump_head"] is not None:
        theirs = dump_targets(spec["dump_head"])
        mine = {
            perts["test"][i][0]: y["test"][i]
            for i in range(len(perts["test"]))
            if keep_pinned["test"][i] and len(perts["test"][i]) == 1
        }
        shared = sorted(set(mine) & set(theirs))
        assert len(shared) >= 500, f"only {len(shared)} pinned genes match the dump"
        drift = []
        for gene in shared:
            theirs_vec = np.array(
                [np.nan if v is None else float(v) for v in np.atleast_1d(theirs[gene])]
            )
            if not np.allclose(mine[gene], theirs_vec, equal_nan=True, atol=1e-5):
                drift.append(gene)
        assert not drift, (
            f"{len(drift)} pinned targets differ from the wave-7 dump, e.g. {drift[:5]}"
        )
        print(
            f"  identity check: {len(shared)} pinned targets match the wave-7 dump exactly",
            flush=True,
        )

    mu = np.array(
        [
            y["train"][obs["train"][:, f], f].mean() if obs["train"][:, f].any() else 0.0
            for f in range(len(columns))
        ]
    )[None, :]
    resid = {name: y[name] - mu for name in y}
    # The SVD and the ridge solve need a dense matrix; an unmeasured cell is set to the fit
    # column mean, which is zero in residual space. Scoring never touches an imputed cell.
    resid_dense = {name: np.where(obs[name], resid[name], 0.0) for name in resid}

    payload: dict[str, Any] = {
        "generated_by": "experiments/027-betaxanthin-metabolic/scripts/linear_baselines_bx.py",
        "target": args.target,
        "columns": columns,
        "dataset_subdir": spec["dataset_subdir"],
        "pinned_split": pinned_report,
        "split_seed": cfg.data.split_seed,
        "n": {name: int(len(y[name])) for name in y},
        "n_pinned": {name: int(keep_pinned[name].sum()) for name in y},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "imputed_cells_in_fit": int((~obs["train"]).sum()),
    }

    def score(
        pred: np.ndarray,
        name: str,
        pinned_only: bool,
        covered: np.ndarray | None = None,
        full_stats: bool = False,
    ) -> dict[str, Any]:
        """Score one prediction block. ``covered`` is 019's ``ok`` mask.

        A genotype whose deleted gene has no embedding is EXCLUDED rather than zero-filled
        into the correlation, which is what 019 does and the reason its ``n_scored`` is
        reported beside every cell.
        """
        sel = keep_pinned[name] if pinned_only else np.ones(len(y[name]), dtype=bool)
        if covered is not None:
            sel = sel & covered
        n_perm = 2000 if full_stats else 0
        n_boot = 2000 if full_stats else 0
        block: dict[str, Any] = {
            "per_feature_pearson": per_feature_pearson(
                pred[sel], y[name][sel], obs[name][sel]
            ),
            "nmse": nmse(pred[sel], y[name][sel], mu, obs[name][sel]),
            "n": int(sel.sum()),
        }
        rhos = []
        per_feature: dict[str, Any] = {}
        for f, column in enumerate(columns):
            m = sel & obs[name][:, f]
            cell = assess(pred[m, f], y[name][m, f], rng, n_perm=n_perm, n_boot=n_boot)
            per_feature[column] = cell
            rhos.append(0.0 if cell["spearman"] is None else float(cell["spearman"]))
        block["mean_spearman"] = float(np.mean(rhos))
        block["per_feature"] = per_feature
        return block

    # ---- B0 and B1, with 019's hard self-check -----------------------------------------
    for name in ("val", "test"):
        b0 = score(np.repeat(mu, len(y[name]), axis=0), name, pinned_only=False)
        b1 = score(np.zeros_like(y[name]), name, pinned_only=False)
        payload[f"B0_train_mean_{name}"] = b0
        payload[f"B1_zeros_{name}"] = b1
        payload[f"B0_train_mean_{name}_pinned"] = score(
            np.repeat(mu, len(y[name]), axis=0), name, pinned_only=True
        )
        print(
            f"  {name}: B0 pf={b0['per_feature_pearson']:+.6f} nmse={b0['nmse']:.4f}; "
            f"B1 pf={b1['per_feature_pearson']:+.4f} nmse={b1['nmse']:.4f}",
            flush=True,
        )
        if abs(b0["per_feature_pearson"]) > 1e-9:
            raise ValueError(
                f"B0 per-feature Pearson on {name} is {b0['per_feature_pearson']}, not 0"
            )
        if not 0.9 < b0["nmse"] < 1.15:
            raise ValueError(f"B0 nmse on {name} is {b0['nmse']}, expected ~1.0")

    # ---- B2 and B3 over the four embedding sets ----------------------------------------
    ranks = [k for k in RANK_GRID if k <= len(columns)]
    b2: dict[str, Any] = {}
    b3: dict[str, Any] = {}
    for emb_name, members in EMBEDDINGS.items():
        emb = embedding_matrix(members, DATA_ROOT, genome, graph)
        dim = len(next(iter(emb.values())))
        p: dict[str, np.ndarray] = {}
        ok: dict[str, np.ndarray] = {}
        for name in y:
            mat = np.zeros((len(perts[name]), dim))
            good = np.zeros(len(perts[name]), dtype=bool)
            for i, genes in enumerate(perts[name]):
                vecs = [emb[g] for g in genes if g in emb]
                if vecs:
                    mat[i] = np.mean(vecs, axis=0)
                    good[i] = True
            p[name], ok[name] = mat, good
        if ok["train"].sum() < 50 or any(ok[s].sum() < 20 for s in ("val", "test")):
            print(f"  {emb_name}: too few genotypes covered; skipped", flush=True)
            continue
        pm = p["train"][ok["train"]].mean(axis=0, keepdims=True)
        ps = p["train"][ok["train"]].std(axis=0, keepdims=True) + 1e-8
        for name in p:
            p[name] = (p[name] - pm) / ps
        r_tr, p_tr = resid_dense["train"][ok["train"]], p["train"][ok["train"]]

        cells: list[dict[str, Any]] = []
        for k_rank in ranks:
            if k_rank > min(r_tr.shape):
                continue
            for ridge in RIDGE_GRID:
                cell: dict[str, Any] = {"k_gene": k_rank, "ridge": ridge}
                for name in ("val", "test"):
                    pred = np.zeros_like(y[name])
                    pred[ok[name]] = bilinear(r_tr, p_tr, p[name][ok[name]], k_rank, ridge)
                    pred = pred + mu
                    cell[name] = score(pred, name, False, covered=ok[name])
                    if name == "test":
                        cell["test_pinned"] = score(pred, name, True, covered=ok[name])
                        cell["_pred_test"] = pred
                cells.append(cell)
        best2 = max(cells, key=lambda c: float(c["val"]["per_feature_pearson"]))
        b2[emb_name] = {
            "emb_dim": int(dim),
            "selected_on_val": {
                "k_gene": best2["k_gene"],
                "ridge": best2["ridge"],
                "val": best2["val"],
                "test": best2["test"],
                # The SELECTED cell is re-scored with the permutation null and the bootstrap
                # interval; the swept cells carry the point estimate only.
                "test_pinned": score(
                    best2["_pred_test"], "test", True, covered=ok["test"], full_stats=True
                ),
            },
            "cells": [
                {
                    "k_gene": c["k_gene"],
                    "ridge": c["ridge"],
                    "val_per_feature_pearson": c["val"]["per_feature_pearson"],
                    "val_mean_spearman": c["val"]["mean_spearman"],
                    "test_mean_spearman": c["test"]["mean_spearman"],
                    "test_pinned_mean_spearman": c["test_pinned"]["mean_spearman"],
                }
                for c in cells
            ],
            "n_scored": {s: int(ok[s].sum()) for s in ("val", "test")},
        }

        e_tr = p_tr / (np.linalg.norm(p_tr, axis=1, keepdims=True) + 1e-12)
        cells3: list[dict[str, Any]] = []
        for k in KNN_GRID:
            cell = {"k": k}
            for name in ("val", "test"):
                e_s = p[name] / (np.linalg.norm(p[name], axis=1, keepdims=True) + 1e-12)
                neighbors = np.argsort(-(e_s @ e_tr.T), axis=1)[:, :k]
                pred = r_tr[neighbors].mean(axis=1) + mu
                cell[name] = score(pred, name, False, covered=ok[name])
                if name == "test":
                    cell["test_pinned"] = score(pred, name, True, covered=ok[name])
                    cell["_pred_test"] = pred
            cells3.append(cell)
        best3 = max(cells3, key=lambda c: float(c["val"]["per_feature_pearson"]))
        b3[emb_name] = {
            "selected_on_val": {
                "k": best3["k"],
                "val": best3["val"],
                "test": best3["test"],
                "test_pinned": score(
                    best3["_pred_test"], "test", True, covered=ok["test"], full_stats=True
                ),
            },
            "cells": [
                {
                    "k": c["k"],
                    "val_per_feature_pearson": c["val"]["per_feature_pearson"],
                    "test_pinned_mean_spearman": c["test_pinned"]["mean_spearman"],
                }
                for c in cells3
            ],
            "n_scored": {s: int(ok[s].sum()) for s in ("val", "test")},
        }
        print(
            f"  {emb_name:<28s} B2 val pf {best2['val']['per_feature_pearson']:+.4f} "
            f"test-pinned rho {best2['test_pinned']['mean_spearman']:+.4f} "
            f"(k={best2['k_gene']}, ridge={best2['ridge']}) | "
            f"B3 val pf {best3['val']['per_feature_pearson']:+.4f} "
            f"test-pinned rho {best3['test_pinned']['mean_spearman']:+.4f} (k={best3['k']})",
            flush=True,
        )

    # ---- the headline cell, with its own permutation null and bootstrap CI --------------
    payload["B2_bilinear"] = {"by_embedding": b2}
    payload["B3_neighbor_average"] = {"by_embedding": b3}
    for key, table in (("B2_bilinear", b2), ("B3_neighbor_average", b3)):
        if not table:
            continue
        best = max(
            table,
            key=lambda e: float(table[e]["selected_on_val"]["val"]["per_feature_pearson"]),
        )
        payload[key]["best_embedding"] = best
        payload[key]["headline_test_pinned"] = table[best]["selected_on_val"][
            "test_pinned"
        ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = args.out or osp.join(RESULTS_DIR, f"linear_baselines_{args.target}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
