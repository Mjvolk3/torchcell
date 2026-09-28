# experiments/031-betaxanthin-module/scripts/target_graph_feature_store.py
# [[experiments.031-betaxanthin-module.scripts.target_graph_feature_store]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-betaxanthin-module/scripts/target_graph_feature_store

r"""One-hop propagation of a gene's OWN features over the graphs, as an affine store.

``target_graph_label_store.py`` propagates the target's labels over the graphs and
lifts betaxanthin (w239, 0.288 against the pool's 0.278) and essentiality (w140, 0.925
against 0.908), but a label store is TRANSDUCTIVE: a gene's row is the mean label of its
neighbors, so it needs those labels at inference and cannot be carried to a strain, a
screen or an organism nobody has labeled. This script builds the inductive analogue.
Each gene's row is the mean over its one-hop neighbors of a feature that exists for
every gene without a measurement: the propagated GO multi-hot (``YPD_bx_go``) or the
parsimonious flux vector of its deletion (``YPD_bx_pfba``). No label enters the store,
so a pinned gene's row is computed exactly as a training gene's, and the same row can
be built for any gene that has neighbors, in any organism with a graph.

Columns, per graph named in ``--graphs`` (any of the nine the trainer masks, or
``union_of_9``): the mean base feature over the gene's neighbors that carry a base row,
and $\log(1 + n)$ of their count. A gene whose neighbors carry no base row (or that has
no neighbors) carries the base store's wild-type row and a count of zero. The gene's
own row is never included (no graph has self-loops after their removal). Rows: every
gene of the genome. The base store's values are stored as they are, so the store is
read with the base store's own encoding (standardized for GO, ``log1p`` for a flux).
Stores are never overwritten. Run from the repo root::

    PYTHONHASHSEED=0 PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/031-betaxanthin-module/scripts/target_graph_feature_store.py \
        --base YPD_bx_go --graphs string12_0_coexpression union_of_9 --out YPD_bx_gonbr
"""

from __future__ import annotations

import argparse
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

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
sys.path.insert(0, osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "scripts"))
from target_graph_homophily import GRAPHS  # noqa: E402

from torchcell.graph import SCerevisiaeGraph  # noqa: E402
from torchcell.graph.graph import build_gene_multigraph  # noqa: E402
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome  # noqa: E402

TABLE_DIR = osp.join(DATA_ROOT, "data/torchcell/reference_flux")
UNION = "union_of_9"


class FeatureStoreProvenance(BaseModel):
    """Which base store was propagated over which graphs, and how gaps were filled."""

    model_config = {"extra": "forbid"}

    method: str
    base_store: str
    base_sha256: str
    n_base_genes: int
    n_base_columns: int
    graphs: list[str]
    n_genes: int
    n_columns: int
    column_layout: str
    self_excluded: bool
    fill_for_no_covered_neighbor: str
    labels_read: bool
    python_hash_seed: str
    flux_sha256: str


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base store name: ess_solve_encodings_<base>")
    ap.add_argument(
        "--graphs", nargs="+", default=["string12_0_coexpression", UNION], choices=GRAPHS + [UNION]
    )
    ap.add_argument("--out", required=True, help="store name: ess_solve_encodings_<out>")
    ap.add_argument("--out-dir", default=TABLE_DIR)
    args = ap.parse_args()
    assert os.environ.get("PYTHONHASHSEED") == "0", "PYTHONHASHSEED must be 0"
    started = time.time()
    path = osp.join(args.out_dir, f"ess_solve_encodings_{args.out}.npz")
    assert not osp.exists(path), f"{path} exists; stores are never overwritten"
    base_path = osp.join(TABLE_DIR, f"ess_solve_encodings_{args.base}.npz")
    z = np.load(base_path, allow_pickle=False)
    base_genes = [str(g) for g in z["genes"]]
    base = z["v_deletion"].astype(np.float64)
    base_wild = z["v_wild_type"].astype(np.float64)
    base_cols = [str(c) for c in z["rxn_ids"]]
    assert base.shape == (len(base_genes), len(base_cols))
    with open(base_path, "rb") as fh:
        base_sha = hashlib.sha256(fh.read()).hexdigest()

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
    undirected[UNION] = nx.compose_all([undirected[n] for n in GRAPHS])

    genes = sorted(genome.gene_set)
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    covered = np.zeros(n, dtype=bool)
    feat = np.tile(base_wild, (n, 1))
    n_outside = 0
    for g, row in zip(base_genes, base, strict=True):
        if g not in index:
            n_outside += 1
            continue
        feat[index[g]] = row
        covered[index[g]] = True
    print(
        f"  base {args.base}: {int(covered.sum())}/{n} genome genes carry a row, "
        f"{n_outside} base genes outside the genome, {len(base_cols)} columns",
        flush=True,
    )

    blocks: list[np.ndarray] = []
    names: list[str] = []
    wild: list[float] = []
    for name in args.graphs:
        g = undirected[name]
        a = np.zeros((n, n), dtype=np.float32)
        edges = [(index[u], index[v]) for u, v in g.edges if u != v and u in index and v in index]
        src = [u for u, _ in edges]
        dst = [v for _, v in edges]
        a[src, dst] = 1.0
        a[dst, src] = 1.0
        assert not a.diagonal().any()
        reach = a * covered[None, :].astype(np.float32)
        counts = reach.sum(axis=1).astype(np.float64)
        sums = reach.astype(np.float64) @ feat
        means = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], base_wild[None, :])
        blocks.append(means.astype(np.float32))
        names.extend(f"{name}_nbr_mean_{c}" for c in base_cols)
        wild.extend(float(v) for v in base_wild)
        blocks.append(np.log1p(counts).astype(np.float32)[:, None])
        names.append(f"{name}_log1p_covered_neighbors")
        wild.append(float(np.log1p(counts).mean()))
        print(
            f"  {name:26s} genes with a covered neighbor {int((counts > 0).sum()):5d}/{n}; "
            f"mean count {counts[counts > 0].mean():.1f}",
            flush=True,
        )
    values = np.concatenate(blocks, axis=1)
    assert np.isfinite(values).all()
    assert values.shape[1] == len(names) == len(wild)
    provenance = FeatureStoreProvenance(
        method=(
            "one-hop feature propagation: the mean base feature over a gene's neighbors "
            "that carry a base row in each graph and log1p of their count; written by "
            "target_graph_feature_store.py"
        ),
        base_store=args.base,
        base_sha256=base_sha,
        n_base_genes=len(base_genes),
        n_base_columns=len(base_cols),
        graphs=list(args.graphs),
        n_genes=n,
        n_columns=len(names),
        column_layout="per graph: <graph>_nbr_mean_<base column> ..., <graph>_log1p_covered_neighbors",
        self_excluded=True,
        fill_for_no_covered_neighbor="the base store's wild-type row; count 0",
        labels_read=False,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        flux_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        path,
        genes=np.array(genes),
        v_deletion=values,
        v_wild_type=np.array(wild, np.float32),
        rxn_ids=np.array(names),
        provenance=np.array(provenance.model_dump_json()),
    )
    results_dir = osp.join(EXPERIMENT_ROOT, "031-betaxanthin-module", "results")
    with open(osp.join(results_dir, f"target_graph_feature_store_{args.out}.json"), "w") as fh:
        json.dump(
            {
                "generated_by": "experiments/031-betaxanthin-module/scripts/target_graph_feature_store.py",
                "store": path,
                "shape": list(values.shape),
                "n_columns": len(names),
                "provenance": provenance.model_dump(),
                "seconds": time.time() - started,
            },
            fh,
            indent=2,
        )
    print(path)


if __name__ == "__main__":
    main()
