# experiments/028-gene-essentiality/scripts/wl_knockout_test.py
# [[experiments.028-gene-essentiality.scripts.wl_knockout_test]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/028-gene-essentiality/scripts/wl_knockout_test

r"""The Weisfeiler-Leman test on single-deletion gene graphs, and the WL kernel as a rung.

Color refinement assigns every node a color, then repeatedly replaces it by a hash of
its own color and the multiset of its neighbors' colors; after $k$ rounds a node's color
summarizes its $k$-hop neighborhood, and the multiset of colors is the 1-WL signature of
the graph. Two graphs with different signatures are not isomorphic; two with the same
signature may or may not be. A message-passing GNN is at most as expressive as this test
(Xu et al. 2019), and the cell graph transformer's perturbed graph is one gene graph with
the deleted gene's token removed, so the question of whether deletions are even
*distinguishable* by 1-WL is the question of whether a message-passing model could tell
them apart by structure alone.

Three measurements on the gene graphs the transformer reads (the physical interaction
graph, and the union of its nine graphs):

1. **The test.** For every labeled gene, the 1-WL hash of the graph with that gene's node
   removed, at $k = 1, 2, 3$ iterations with degree as the initial color. The number of
   distinct hashes among the 1,121 deletion graphs says how many deletions the test
   separates; a collision is a pair of deletions no 1-WL-bounded model can tell apart from
   structure.
2. **The kernel as an encoding.** Each gene's 2-hop ego graph, refined for 3 iterations
   with degree as the initial color, gives a bag of colors; hashed into 1,024 bins it is a
   per-gene vector, the WL subtree kernel's feature map. Written to the encodings file so
   the ladder can fit it (\texttt{logistic\_baselines.py}).
3. **The kernel as a classifier.** Logistic regression on that vector on the split, its
   C chosen by five-fold cross-validation inside the training genes, scored once on the
   195 held-out genes: what structure alone gives.

    python experiments/028-gene-essentiality/scripts/wl_knockout_test.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import os.path as osp
import time
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "028-gene-essentiality", "results")
SPLIT = osp.join(RESULTS, "splits", "ess_fcl_val0.2_s0.json.gz")
ENCODING_NPZ = osp.join(DATA_ROOT, "data/torchcell/reference_flux/ess_wl_encodings.npz")
GRAPH_NAMES = (
    "physical",
    "regulatory",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
)
ITERATIONS = (1, 2, 3)
EGO_RADIUS = 2
EGO_ITERATIONS = 3
N_BINS = 1024
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


class TestRow(BaseModel):
    """1-WL on the deletion graphs of one gene graph at one iteration count."""

    graph: str
    n_nodes: int
    n_edges: int
    iterations: int
    n_deletions: int
    n_distinct_hashes: int
    n_in_collision: int
    largest_collision: int
    essential_in_collision: int
    seconds: float


class KernelRow(BaseModel):
    """Logistic regression on the WL kernel vector, on the split."""

    graph: str
    dim: int
    C: float
    cv_auroc: float
    val_auroc: float
    test_auroc: float
    n_train: int
    n_val: int
    n_test: int


def gene_graphs() -> dict[str, nx.Graph]:
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

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
    gene_ids = sorted(genome.gene_set)
    out: dict[str, nx.Graph] = {}
    union = nx.Graph()
    union.add_nodes_from(gene_ids)
    for name in GRAPH_NAMES:
        g_raw = getattr(graph, f"G_{name}").graph
        g = nx.Graph()
        g.add_nodes_from(gene_ids)
        g.add_edges_from((u, v) for u, v in g_raw.edges() if u in union and v in union)
        if name == "physical":
            out["physical"] = g
        union.add_edges_from(g.edges())
    out["union of nine"] = union
    return out


def degree_labels(g: nx.Graph) -> dict[str, str]:
    return {n: str(d) for n, d in g.degree()}


def wl_test(
    name: str, g: nx.Graph, genes: list[str], labels: dict[str, int]
) -> list[TestRow]:
    rows = []
    for k in ITERATIONS:
        t = time.time()
        hashes: dict[str, str] = {}
        for gene in genes:
            h = g.copy()
            h.remove_node(gene)
            nx.set_node_attributes(h, degree_labels(h), "c")
            hashes[gene] = nx.weisfeiler_lehman_graph_hash(
                h, node_attr="c", iterations=k
            )
        groups = Counter(hashes.values())
        colliding = [gene for gene in genes if groups[hashes[gene]] > 1]
        rows.append(
            TestRow(
                graph=name,
                n_nodes=g.number_of_nodes(),
                n_edges=g.number_of_edges(),
                iterations=k,
                n_deletions=len(genes),
                n_distinct_hashes=len(groups),
                n_in_collision=len(colliding),
                largest_collision=max(groups.values()),
                essential_in_collision=sum(labels[x] for x in colliding),
                seconds=time.time() - t,
            )
        )
        print(rows[-1].model_dump_json())
    return rows


def wl_kernel_vectors(g: nx.Graph, genes: list[str]) -> np.ndarray:
    """Bag of WL colors of each gene's ego graph, hashed to ``N_BINS`` bins."""
    vectors = np.zeros((len(genes), N_BINS), dtype=np.float32)
    for i, gene in enumerate(genes):
        ego = nx.ego_graph(g, gene, radius=EGO_RADIUS)
        nx.set_node_attributes(ego, degree_labels(ego), "c")
        per_node = nx.weisfeiler_lehman_subgraph_hashes(
            ego, node_attr="c", iterations=EGO_ITERATIONS
        )
        for node_hashes in per_node.values():
            for depth, h in enumerate(node_hashes):
                bin_ = (
                    int(hashlib.sha1(f"{depth}:{h}".encode()).hexdigest(), 16) % N_BINS
                )
                vectors[i, bin_] += 1.0
    return np.log1p(vectors)


def fit_kernel(
    name: str, x: np.ndarray, genes: list[str], split: dict[str, Any]
) -> KernelRow:
    labels = {k: int(v) for k, v in split["labels"].items()}
    idx = {g: i for i, g in enumerate(genes)}
    parts = {
        k: [g for g in split["genes"][k] if g in idx] for k in ("train", "val", "test")
    }
    xs = {k: x[[idx[g] for g in v]] for k, v in parts.items()}
    ys = {k: np.array([labels[g] for g in v]) for k, v in parts.items()}
    scaler = StandardScaler().fit(xs["train"])
    xs = {k: scaler.transform(v) for k, v in xs.items()}
    best: tuple[float, float] | None = None
    for c in C_GRID:
        scores = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(
            xs["train"], ys["train"]
        ):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            clf.fit(xs["train"][tr], ys["train"][tr])
            scores.append(
                roc_auc_score(ys["train"][va], clf.decision_function(xs["train"][va]))
            )
        mean = float(np.mean(scores))
        if best is None or mean > best[1]:
            best = (c, mean)
    assert best is not None
    clf = LogisticRegression(C=best[0], class_weight="balanced", max_iter=2000)
    clf.fit(xs["train"], ys["train"])
    row = KernelRow(
        graph=name,
        dim=int(x.shape[1]),
        C=best[0],
        cv_auroc=best[1],
        val_auroc=float(roc_auc_score(ys["val"], clf.decision_function(xs["val"]))),
        test_auroc=float(roc_auc_score(ys["test"], clf.decision_function(xs["test"]))),
        n_train=len(ys["train"]),
        n_val=len(ys["val"]),
        n_test=len(ys["test"]),
    )
    print(row.model_dump_json())
    return row


def main() -> None:
    with gzip.open(SPLIT, "rt") as fh:
        split = json.load(fh)
    labels = {k: int(v) for k, v in split["labels"].items()}
    graphs = gene_graphs()
    genes = [g for g in sorted(labels) if g in graphs["physical"]]
    print(f"{len(genes)} labeled genes in the gene graph")
    tests: list[TestRow] = []
    kernels: list[KernelRow] = []
    encodings: dict[str, np.ndarray] = {}
    for name, g in graphs.items():
        tests.extend(wl_test(name, g, genes, labels))
        x = wl_kernel_vectors(g, genes)
        encodings[name] = x
        kernels.append(fit_kernel(name, x, genes, split))
    os.makedirs(osp.dirname(ENCODING_NPZ), exist_ok=True)
    np.savez(
        ENCODING_NPZ,
        genes=np.array(genes),
        wl_ego_physical=encodings["physical"],
        wl_ego_union=encodings["union of nine"],
        provenance=np.array(
            json.dumps(
                {
                    "ego_radius": EGO_RADIUS,
                    "iterations": EGO_ITERATIONS,
                    "n_bins": N_BINS,
                    "initial_color": "degree",
                    "transform": "log1p of the bin counts",
                }
            )
        ),
    )
    out = osp.join(RESULTS, "wl_knockout_test.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "tests": [r.model_dump() for r in tests],
                "kernel_rungs": [r.model_dump() for r in kernels],
                "encodings": ENCODING_NPZ,
            },
            fh,
            indent=2,
        )
    print(out)


if __name__ == "__main__":
    main()
