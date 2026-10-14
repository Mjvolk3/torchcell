# experiments/025-solid-growth/scripts/graph_reg_khop_density.py
# [[experiments.025-solid-growth.scripts.graph_reg_khop_density]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/025-solid-growth/scripts/graph_reg_khop_density
"""How much of the genome each gene-gene graph reaches at one, two and three hops.

The round-2 reach arms give a head the k-hop neighborhood of its graph as its support
(mask) or target (prior). A head whose three-hop support is most of the genome has no
prior left, so before those arms are launched this tabulates, for each of the nine
graphs the sweep regularizes, the number of ordered pairs reachable within k steps and
the fraction of all N(N-1) pairs that is, as stored (directed for regulatory and TFLink)
and with every edge made undirected (the mask's default). Uses the same
:func:`khop_reach` the model uses, on the same graphs the trainer builds.

Writes experiments/025-solid-growth/results/graph_reg_khop_density.json and
notes-tex/025-graph-reg-sweep/tables/t5-khop-density.tex.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import torch
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.models.equivariant_cell_graph_transformer import khop_reach
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "025-solid-growth", "results")
TABLES_DIR = osp.join(osp.dirname(EXPERIMENT_ROOT), "notes-tex", "025-graph-reg-sweep", "tables")
GRAPHS = [
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
HOPS = (1, 2, 3)


class Density(BaseModel):
    """Reach of one graph at one hop count under one edge treatment."""

    graph: str
    symmetric: bool
    hops: int
    pairs: int  # ordered (i, j), i != j, reachable within `hops`
    fraction: float  # pairs / (N (N - 1))
    mean_out_degree: float  # pairs / N


def main() -> None:
    """Build the graphs the trainer builds and count reach."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPHS)
    genes = sorted(genome.gene_set)
    index = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    rows: list[Density] = []
    for name in GRAPHS:
        g = multigraph.graphs[name].graph
        edges = [(index[u], index[v]) for u, v in g.edges() if u in index and v in index]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        stored_directed = g.is_directed()
        for symmetric in (False, True):
            if symmetric and not stored_directed:
                continue  # an undirected graph is its own symmetrization
            for hops in HOPS:
                reach = khop_reach(edge_index, n, hops, symmetric=symmetric or not stored_directed)
                reach.fill_diagonal_(False)
                pairs = int(reach.sum())
                rows.append(
                    Density(
                        graph=name,
                        symmetric=symmetric or not stored_directed,
                        hops=hops,
                        pairs=pairs,
                        fraction=pairs / (n * (n - 1)),
                        mean_out_degree=pairs / n,
                    )
                )
                print(f"{name:26s} sym={rows[-1].symmetric!s:5s} hops={hops} pairs={pairs:>12,} frac={rows[-1].fraction:.4f} mean deg={rows[-1].mean_out_degree:,.1f}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "graph_reg_khop_density.json"), "w") as fh:
        json.dump({"n_genes": n, "rows": [r.model_dump() for r in rows]}, fh, indent=1)
    _table(rows, n)


def _table(rows: list[Density], n: int) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    by = {(r.graph, r.symmetric, r.hops): r for r in rows}
    lines = [
        "%% SOURCE: experiments/025-solid-growth/scripts/graph_reg_khop_density.py -- GENERATED, do not edit",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{3}{c}{mean reach per gene} & \\multicolumn{3}{c}{fraction of all pairs} \\\\",
        "graph & edges & 1 hop & 2 hops & 3 hops & 1 hop & 2 hops & 3 hops \\\\",
        "\\midrule",
    ]
    for name in GRAPHS:
        for symmetric in (False, True):
            if (name, symmetric, 1) not in by:
                continue
            treat = "undirected" if symmetric else "directed"
            cells = [by[(name, symmetric, h)] for h in HOPS]
            lines.append(
                f"{name.replace('_', chr(92) + '_')} & {treat} & "
                + " & ".join(f"{c.mean_out_degree:,.0f}" for c in cells)
                + " & "
                + " & ".join(f"{c.fraction:.3f}" for c in cells)
                + " \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(osp.join(TABLES_DIR, "t5-khop-density.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"N = {n}; wrote {TABLES_DIR}/t5-khop-density.tex")


if __name__ == "__main__":
    main()
