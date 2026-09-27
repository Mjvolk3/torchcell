# experiments/028-gene-essentiality/scripts/graph_label_propagation.py
# [[experiments.028-gene-essentiality.scripts.graph_label_propagation]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/graph_label_propagation

r"""Do the nine gene graphs carry essentiality at all: one-hop label propagation on the split.

The betaxanthin study's read (``target_graph_homophily.py``, 2026-09-27) on the released
essentiality split: every held-out gene is scored by the fraction of its labeled
neighbors (the training and validation genes) that are essential, in each of the nine
graphs the trainer masks and regularizes and in their union, with nothing fitted. The
read is the AUROC on the 195 held-out genes with at least one labeled neighbor, the
same with every uncovered gene filled by the labeled base rate on all 195, the same
within two hops, a degree-matched null in which each gene's labeled neighbors are
replaced by as many labeled genes drawn at random, and the predictor scored
leave-one-out on the labeled genes. Beside the message-passing baselines
(``graph_baselines.py``: a graph convolution over the union with ProtT5 inputs reads
0.834 against 0.754 for the same inputs with no graph) it says how much of what the
convolution reads is the labels of the neighbors, which the perturbed-graph transformer
never sees.

With ``--write-store <name>`` the one-hop read is also written as an affine store over
every gene of the genome (``ess_solve_encodings_<name>.npz``, the layout the trainer's
``--ess-solve-features pfba`` reads): per graph the labeled-neighbor essential fraction
(a labeled gene's row leave-one-out, the labeled base rate where no labeled neighbor)
and $\log(1 + n)$ of the count, twenty columns, read as ``log1p`` like the flux store.
No held-out gene's label is read. Stores are never overwritten.

Writes ``results/graph_label_propagation.json``. Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/028-gene-essentiality/scripts/graph_label_propagation.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import os.path as osp
import sys
import time

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
from target_graph_homophily import GRAPHS, neighbor_mean  # noqa: E402

from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402


class EssentialityLabelStoreProvenance(BaseModel):
    """Which genes' labels the propagated columns read, and how."""

    model_config = {"extra": "forbid"}

    method: str
    split: str
    n_labeled_genes: int
    n_held_out_genes_excluded: int
    labeled_base_rate: float
    graphs: list[str]
    n_genes: int
    n_columns: int
    column_layout: str
    leave_one_out: bool
    fill_for_no_labeled_neighbor: str
    python_hash_seed: str
    flux_sha256: str


class GraphRead(BaseModel):
    """One graph at one reach: the labeled-neighbor essential fraction on the held-out genes."""

    graph: str
    hops: int
    n_nodes: int
    n_edges: int
    n_test: int
    n_test_covered: int
    coverage: float
    mean_labeled_neighbors: float
    auroc_covered: float
    auroc_filled: float
    null_mean: float
    null_sd: float
    z: float
    n_labeled_covered: int
    labeled_loo_auroc: float


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--write-store",
        default=None,
        help="also write the one-hop read as ess_solve_encodings_<name>.npz",
    )
    args = ap.parse_args()
    store_path = None
    if args.write_store is not None:
        store_path = osp.join(
            DATA_ROOT,
            "data/torchcell/reference_flux",
            f"ess_solve_encodings_{args.write_store}.npz",
        )
        assert not osp.exists(store_path), f"{store_path} exists; stores are never overwritten"
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    rng = np.random.default_rng(args.seed)
    with gzip.open(SPLIT, "rt") as fh:
        split = json.load(fh)
    labels = {str(g): int(v) for g, v in split["labels"].items()}
    labeled_genes = [str(g) for g in split["genes"]["train"] + split["genes"]["val"]]
    test_genes = [str(g) for g in split["genes"]["test"]]
    assert not set(labeled_genes) & set(test_genes)
    print(
        f"  labeled (train+val): {len(labeled_genes)} ({sum(labels[g] for g in labeled_genes)} "
        f"essential); held out: {len(test_genes)} ({sum(labels[g] for g in test_genes)} essential)",
        flush=True,
    )

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
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    undirected = {}
    for name in GRAPHS:
        g = multigraph[name].graph
        undirected[name] = g.to_undirected() if g.is_directed() else g
    undirected["union_of_9"] = nx.compose_all([undirected[n] for n in GRAPHS])

    genes = sorted(
        set().union(*(set(g.nodes) for g in undirected.values()))
        | set(labeled_genes)
        | set(test_genes)
        | set(genome.gene_set)
    )
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    y = np.full((n, 1), np.nan)
    lab_mask = np.zeros(n, dtype=bool)
    for g in labeled_genes:
        y[index[g], 0] = labels[g]
        lab_mask[index[g]] = True
    for g in test_genes:
        y[index[g], 0] = labels[g]
    y_lab = np.where(lab_mask[:, None], y, np.nan)
    base_rate = float(np.nanmean(y_lab))
    test_idx = np.array([index[g] for g in test_genes])
    lab_idx = np.flatnonzero(lab_mask)
    y_test = y[test_idx, 0]

    def auroc(score: np.ndarray, truth: np.ndarray, rows: np.ndarray) -> float:
        return float(roc_auc_score(truth[rows], score[rows]))

    reads: list[GraphRead] = []
    store_blocks: list[np.ndarray] = []
    store_names: list[str] = []
    store_wild: list[float] = []
    for name, g in undirected.items():
        a = np.zeros((n, n), dtype=np.float32)
        src = [index[u] for u, v in g.edges if u != v]
        dst = [index[v] for u, v in g.edges if u != v]
        a[src, dst] = 1.0
        a[dst, src] = 1.0
        n_edges = int(a.sum() // 2)
        if store_path is not None:
            m_all, c_all = neighbor_mean(a, y_lab, lab_mask)
            frac_all = np.where(np.isfinite(m_all[:, 0]), m_all[:, 0], base_rate)
            store_blocks += [frac_all.astype(np.float32), np.log1p(c_all).astype(np.float32)]
            store_names += [f"{name}_nbr_essential_fraction", f"{name}_log1p_labeled_neighbors"]
            store_wild += [base_rate, float(np.log1p(c_all).mean())]
        two = None
        for hops in (1, 2):
            if hops == 2:
                two = ((a + a @ a) > 0).astype(np.float32)
                np.fill_diagonal(two, 0.0)
            reach = a if hops == 1 else two
            assert reach is not None
            r_test = reach[test_idx]
            means, counts = neighbor_mean(r_test, y_lab, lab_mask)
            covered = counts > 0
            frac = means[:, 0]
            filled = np.where(np.isfinite(frac), frac, base_rate)
            null = []
            k = counts.astype(int)
            for _ in range(args.permutations):
                r_null = np.zeros_like(r_test)
                for i, ki in enumerate(k):
                    if ki > 0:
                        r_null[i, rng.choice(lab_idx, ki, replace=False)] = 1.0
                m_null, _ = neighbor_mean(r_null, y_lab, lab_mask)
                null.append(auroc(m_null[:, 0], y_test, covered))
            null_mean, null_sd = float(np.mean(null)), float(np.std(null, ddof=1))
            m_lab, c_lab = neighbor_mean(reach[lab_idx], y_lab, lab_mask)
            lab_cov = c_lab > 0
            read = GraphRead(
                graph=name,
                hops=hops,
                n_nodes=int(g.number_of_nodes()),
                n_edges=n_edges,
                n_test=len(test_idx),
                n_test_covered=int(covered.sum()),
                coverage=float(covered.mean()),
                mean_labeled_neighbors=float(counts[covered].mean()),
                auroc_covered=auroc(frac, y_test, covered),
                auroc_filled=auroc(filled, y_test, np.ones(len(test_idx), bool)),
                null_mean=null_mean,
                null_sd=null_sd,
                z=(auroc(frac, y_test, covered) - null_mean) / null_sd if null_sd > 0 else float("nan"),
                n_labeled_covered=int(lab_cov.sum()),
                labeled_loo_auroc=auroc(m_lab[:, 0], y[lab_idx, 0], lab_cov),
            )
            reads.append(read)
            print(
                f"{name:26s} hop {hops} covered {read.n_test_covered:3d}/{read.n_test} "
                f"(mean {read.mean_labeled_neighbors:6.1f} nbrs) auroc {read.auroc_covered:.3f} "
                f"filled {read.auroc_filled:.3f} null {read.null_mean:.3f}+-{read.null_sd:.3f} "
                f"z {read.z:+.1f} LOO {read.labeled_loo_auroc:.3f} (n {read.n_labeled_covered})",
                flush=True,
            )
    out = {
        "generated_by": "experiments/028-gene-essentiality/scripts/graph_label_propagation.py",
        "split": SPLIT,
        "n_labeled": len(labeled_genes),
        "n_test": len(test_genes),
        "labeled_base_rate": base_rate,
        "permutations": args.permutations,
        "statistic": (
            "AUROC on the held-out genes of the essential fraction among a gene's labeled "
            "(train and validation) neighbors within the reach; covered = at least one "
            "labeled neighbor; filled = the labeled base rate where none; a degree-matched "
            "random-neighbor null; leave-one-out on the labeled genes"
        ),
        "reads": [r.model_dump() for r in reads],
        "seconds": time.time() - started,
    }
    path = osp.join(RESULTS, "graph_label_propagation.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(path)
    if store_path is not None:
        values = np.stack(store_blocks, axis=1)
        assert np.isfinite(values).all() and (values >= 0).all()
        in_genome = np.array([g in genome.gene_set for g in genes])
        provenance = EssentialityLabelStoreProvenance(
            method=(
                "one-hop label propagation: the essential fraction among a gene's labeled "
                "neighbors in each graph and log1p of their count; written by "
                "graph_label_propagation.py --write-store"
            ),
            split=SPLIT,
            n_labeled_genes=len(labeled_genes),
            n_held_out_genes_excluded=len(test_genes),
            labeled_base_rate=base_rate,
            graphs=list(undirected),
            n_genes=int(in_genome.sum()),
            n_columns=len(store_names),
            column_layout="per graph: <graph>_nbr_essential_fraction, <graph>_log1p_labeled_neighbors",
            leave_one_out=True,
            fill_for_no_labeled_neighbor="the labeled base rate; count 0",
            python_hash_seed=os.environ["PYTHONHASHSEED"],
            flux_sha256=hashlib.sha256(values[in_genome].tobytes()).hexdigest(),
        )
        np.savez(
            store_path,
            genes=np.array([g for g, k in zip(genes, in_genome, strict=True) if k]),
            v_deletion=values[in_genome],
            v_wild_type=np.array(store_wild, np.float32),
            rxn_ids=np.array(store_names),
            provenance=np.array(provenance.model_dump_json()),
        )
        with open(osp.join(RESULTS, f"graph_label_store_{args.write_store}.json"), "w") as fh:
            json.dump(
                {
                    "generated_by": "experiments/028-gene-essentiality/scripts/graph_label_propagation.py",
                    "store": store_path,
                    "shape": [int(in_genome.sum()), len(store_names)],
                    "columns": store_names,
                    "provenance": provenance.model_dump(),
                },
                fh,
                indent=2,
            )
        print(f"{store_path}: {values[in_genome].shape}", flush=True)


if __name__ == "__main__":
    main()
