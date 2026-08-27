# experiments/019-simb-multimodal/scripts/graph_prior_probe.py
# [[experiments.019-simb-multimodal.scripts.graph_prior_probe]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/019-simb-multimodal/scripts/graph_prior_probe
"""Does graph proximity predict which reporters respond to a deletion?

THE ASSUMPTION UNDER TEST. The nine interaction graphs do not inject features into the
model. They enter as a hard additive attention mask, so all they do is constrain where each
token may look. Stacking L layers composes the attention matrices, which makes the
influence of gene X on gene Y's final token approximately the L-step random-walk
reachability of Y from X on that graph. The entire content of the graph channel is
therefore one falsifiable claim about the data:

    deleting X perturbs the genes that are close to X on graph k.

Nobody had checked it. If it is false, pulling attention toward the graph pushes the model
toward the WRONG prior and no value of the regularization weight rescues that; tuning the
weight only controls how hard one pushes toward a target nobody validated. Nine graphs also
consume nine attention heads, so if only some carry signal the rest are constrained toward
noise and are worse than free heads.

WHAT IS COMPUTED. For deleted gene X and reporter g, response magnitude is |y_{X,g}| from
the measured log2 ratios. The predictor is the t-step row-normalized random-walk weight
w^{(k,t)}_{X,g} = (A_k^t)_{gX}. Two statistics per graph and walk length:

  Spearman   per deleted gene, the across-reporter rank correlation between w and |y|,
             averaged over deleted genes.
  AUC        the probability that a randomly chosen RESPONDING reporter is closer to X on
             graph k than a randomly chosen non-responder, averaged over deleted genes.
             This is the readable one: 0.5 means the graph says nothing.

Responders are the top RESPONDER_QUANTILE of reporters by |y| WITHIN each strain. A
per-strain definition avoids calibrating an absolute threshold against a per-reporter noise
scale, and it keeps the AUC defined for every strain; a second quantile is reported so the
choice is visible rather than load-bearing.

TWO CONTROLS, without which the number means nothing:

  rewired    a configuration-model redraw from the same degree sequence. Destroys
             topology, keeps each node's degree, so anything above this is real structure
             rather than a hub artifact.
             This is the control that matters: a high-degree gene is close to everything,
             and reporters that respond to many deletions are also high-variance, so degree
             alone can manufacture an AUC above 0.5.
  random     a matched-density Erdos-Renyi graph. The floor.

COST. The graphs are sparse and only the 1,484 deleted-gene COLUMNS of A^t are needed, so
the walk is computed by propagating a 6,607 x 1,484 dense block through a sparse matrix
rather than by densifying A^t (which would be 175 MB per power per graph). Minutes on CPU.

Run from repo root:
  PYTHONPATH=. python experiments/019-simb-multimodal/scripts/graph_prior_probe.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from dotenv import load_dotenv
from matplotlib.ticker import MultipleLocator
from scipy.stats import rankdata

from torchcell.graph.graph import SCerevisiaeGraph, build_gene_multigraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    mm_to_in,
    savefig_true_size_svg,
)

EXPERIMENT = "019-simb-multimodal"
DATASET_TAG = "019-simb-multimodal/fig3_core"

# The nine graphs the model masks attention with, one per head, exactly as the
# `head_graphs` block of the 019/023 configs lists them, so the probe tests the deployed
# prior rather than a plausible-looking selection.
#
# TWO NAMING CONVENTIONS, and they differ for the first two entries. The configs use the
# HeteroData RELATION names that `to_cell_data` emits, which are suffixed
# (`physical_interaction`); `build_gene_multigraph` takes the BUILDER names, which are not
# (`physical`). Mapping them here rather than renaming either side keeps the probe's graph
# list literally copy-pasteable from the config it is testing.
CONFIG_TO_BUILDER = {
    "physical_interaction": "physical",
    "regulatory_interaction": "regulatory",
}
GRAPH_NAMES = [
    "physical_interaction",
    "regulatory_interaction",
    "tflink",
    "string12_0_neighborhood",
    "string12_0_fusion",
    "string12_0_cooccurence",
    "string12_0_coexpression",
    "string12_0_experimental",
    "string12_0_database",
]
BUILDER_NAMES = [CONFIG_TO_BUILDER.get(n, n) for n in GRAPH_NAMES]

WALK_LENGTHS = (1, 2, 3)
RESPONDER_QUANTILE = 0.01
RESPONDER_QUANTILE_ALT = 0.05
RANDOM_SEED = 0


def load_response(data_root: str) -> tuple[list[str], list[str], np.ndarray]:
    """(deleted genes, reporter genes, |log2 ratio|) over single-deletion strains."""
    base = osp.join(data_root, "data/torchcell/experiments", DATASET_TAG)
    env = lmdb.open(
        osp.join(base, "processed", "lmdb"), readonly=True, lock=False, subdir=True
    )
    rows: dict[str, np.ndarray] = {}
    reporters: list[str] | None = None
    with env.begin() as txn:
        cursor = txn.cursor()
        for _key, value in cursor:
            recs = json.loads(value.decode())
            if isinstance(recs, dict):
                recs = [recs]
            perts = recs[0]["experiment"]["genotype"]["perturbations"]
            if len(perts) != 1:
                continue  # single deletions only; the probe is indexed by deleted gene
            gene = perts[0]["systematic_gene_name"]
            for rec in recs:
                phen = rec["experiment"]["phenotype"]
                if phen.get("label_name") != "expression_log2_ratio":
                    continue
                values = phen["expression_log2_ratio"]
                if reporters is None:
                    reporters = sorted(values)
                rows[gene] = np.abs(
                    np.array([values[k] for k in reporters], dtype=np.float32)
                )
    env.close()
    if reporters is None:
        raise RuntimeError("no expression records found in the fig3_core build")
    deleted = sorted(rows)
    matrix = np.vstack([rows[g] for g in deleted])
    return deleted, reporters, matrix


def row_normalized(
    graph: nx.Graph, index: dict[str, int], symmetrize: bool = True
) -> sp.csr_matrix:
    """Row-stochastic adjacency over the fixed node index, as the model's mask implies.

    Built through `to_scipy_sparse_array` rather than a Python loop over `graph.edges()`.
    Two of these graphs carry about a million edges and each is rebuilt three times (real,
    rewired, random), so the loop form was several million Python iterations per graph and
    was the slowest thing here by a wide margin.
    """
    n = len(index)
    node_list = [g for g in graph.nodes() if g in index]
    if not node_list:
        return sp.csr_matrix((n, n), dtype=np.float32)
    sub = graph.subgraph(node_list)
    local = nx.to_scipy_sparse_array(
        sub, nodelist=node_list, format="coo", weight=None, dtype=np.float32
    )
    # Vectorized local-to-global reindex; `to_scipy_sparse_array` already emits both
    # directions of an undirected edge, so no symmetrization is needed here.
    to_global = np.array([index[g] for g in node_list], dtype=np.int64)
    adjacency = sp.coo_matrix(
        (local.data, (to_global[local.row], to_global[local.col])),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    if symmetrize:
        # Exactly what `_build_attention_mask` does: set both (i,j) and (j,i).
        adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0  # collapse any parallel edges
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # An isolated node keeps an all-zero row rather than becoming a uniform distribution:
    # a gene with no edges on this graph should contribute no proximity signal, and
    # spreading mass uniformly would invent one.
    inverse = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sp.diags(inverse).astype(np.float32) @ adjacency


def walk_columns(
    adjacency: sp.csr_matrix, columns: np.ndarray, walks: tuple[int, ...]
) -> dict[int, np.ndarray]:
    """(A^t)[:, columns] for each t, by repeated sparse-times-dense propagation.

    Only the deleted-gene columns are ever needed, so this never densifies A^t.
    """
    n = adjacency.shape[0]
    block = np.zeros((n, len(columns)), dtype=np.float32)
    block[columns, np.arange(len(columns))] = 1.0
    out: dict[int, np.ndarray] = {}
    current = block
    for t in range(1, max(walks) + 1):
        current = adjacency @ current
        if t in walks:
            out[t] = current.copy()
    return out


def score(
    weights: np.ndarray, response: np.ndarray, quantile: float
) -> tuple[float, float, float]:
    """Mean AUC, mean Spearman, and the fraction of deleted genes that were scorable.

    `weights` is (reporters x deleted); `response` is (deleted x reporters).

    Fully vectorized over deleted genes. The per-gene form called `rankdata` twice on a
    6,169-element vector for every gene, graph, walk length and control variant, which is
    about 27,000 sorts per graph; ranking each matrix once along the reporter axis does the
    same work in a handful of calls.
    """
    n_deleted = response.shape[0]
    resp_t = response.T  # (reporters x deleted), to match `weights`

    # A deleted gene with no edges on this graph gets no prediction from it. Excluded
    # rather than scored at chance, and the excluded fraction is reported.
    scorable = np.asarray((weights > 0).any(axis=0)).ravel()
    if not scorable.any():
        return float("nan"), float("nan"), 0.0
    w = weights[:, scorable]
    y = resp_t[:, scorable]

    cuts = np.quantile(y, 1.0 - quantile, axis=0)
    responder = y > cuts
    n_pos = responder.sum(axis=0).astype(float)
    n_neg = float(y.shape[0]) - n_pos

    w_ranks = rankdata(w, axis=0)
    rank_sum = (w_ranks * responder).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        auc[(n_pos == 0) | (n_neg == 0)] = np.nan

    # Spearman is Pearson on the ranks, computed column-wise.
    y_ranks = rankdata(y, axis=0)
    wc = w_ranks - w_ranks.mean(axis=0, keepdims=True)
    yc = y_ranks - y_ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt((wc**2).sum(axis=0) * (yc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(denom > 0, (wc * yc).sum(axis=0) / denom, np.nan)

    return (
        float(np.nanmean(auc)),
        float(np.nanmean(rho)),
        float(scorable.sum() / n_deleted),
    )


def rewired(graph: nx.Graph, rng: np.random.Generator) -> tuple[nx.Graph, float]:
    """Degree-preserving null by the configuration model, and how well degree survived.

    WHY NOT `double_edge_swap`, which preserves degree exactly. Mixing needs roughly ten
    swaps per edge, and two of these graphs carry about a million edges (coexpression
    996,199; experimental 822,094), so that control alone is tens of millions of swaps in
    pure Python and runs for hours. The configuration model draws a fresh graph from the
    same degree sequence in O(m).

    The trade is that it produces a multigraph with self-loops, and collapsing those to a
    simple graph lowers the degree of high-degree nodes slightly. That matters here because
    degree is exactly what this control exists to hold fixed, so the Spearman correlation
    between the original and rewired degree sequences is returned and recorded rather than
    assumed to be 1.
    """
    nodes = list(graph.nodes())
    degrees = [d for _n, d in graph.degree()]
    multi = nx.configuration_model(
        degrees, seed=int(rng.integers(0, 2**31 - 1)), create_using=nx.MultiGraph
    )
    simple = nx.Graph(multi)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    simple = nx.relabel_nodes(simple, {i: nodes[i] for i in range(len(nodes))})
    new_degrees = np.array([simple.degree(n) for n in nodes], dtype=float)
    old_degrees = np.array(degrees, dtype=float)
    if old_degrees.std() == 0 or new_degrees.std() == 0:
        preserved = float("nan")
    else:
        preserved = float(
            np.corrcoef(rankdata(old_degrees), rankdata(new_degrees))[0, 1]
        )
    return simple, preserved


def random_matched(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Erdos-Renyi at matched node and edge count."""
    return nx.gnm_random_graph(
        graph.number_of_nodes(),
        graph.number_of_edges(),
        seed=int(rng.integers(0, 2**31 - 1)),
    )


def make_figure(results: dict, out_png: str, out_svg: str) -> None:
    # EVERY size is set explicitly rather than left to resolve from `font.size`. The
    # relative sizes (`medium`, `large`) are what a repo mplstyle or a stale rc can
    # override, and the failure is silent: the layout is right and the type is twice the
    # Nature minimum, which is only obvious once a label runs off its panel.
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 6,
            "axes.labelsize": 6,
            "axes.titlesize": 6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 4.5,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
        }
    )
    names = [g for g in GRAPH_NAMES if g in results["graphs"]]
    directed = [g for g in names if results["graphs"][g].get("directed")]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(56.0)),
        constrained_layout=True,
    )

    # (a) the DEPLOYED prior: symmetric mask, against both controls.
    ax = axes[0]
    ys = np.arange(len(names))
    real = [results["graphs"][g]["t1"]["auc"] for g in names]
    rw = [results["graphs"][g]["t1"]["auc_rewired"] for g in names]
    rd = [results["graphs"][g]["t1"]["auc_random"] for g in names]
    ax.barh(ys + 0.24, real, height=0.30, color=PLOT_PALETTE[0], edgecolor="black",
            linewidth=0.4, label="deployed (symmetric)", zorder=3)
    ax.barh(ys, rw, height=0.30, color=PLOT_PALETTE[2], edgecolor="black",
            linewidth=0.4, label="degree-preserving rewire", zorder=3)
    ax.barh(ys - 0.24, rd, height=0.30, color=PLOT_PALETTE[5], edgecolor="black",
            linewidth=0.4, label="random, matched density", zorder=3)
    ax.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([n.replace("string12_0_", "string ") for n in names], fontsize=5)
    ax.set_xlabel("AUC ($t = 1$)")
    ax.set_xlim(0.45, 0.60)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.legend(fontsize=4.5, loc="lower right", frameon=False)
    ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.set_title("a  The deployed prior is at chance", loc="left", fontsize=6,
                 fontweight="bold")

    # (b) direction, for the graphs where it exists.
    ax = axes[1]
    if directed:
        ys = np.arange(len(directed))
        series = [
            ("TF $\\to$ target", "auc_transposed", PLOT_PALETTE[1]),
            ("deployed (symmetric)", "auc", PLOT_PALETTE[0]),
            ("target $\\to$ TF", "auc_forward", PLOT_PALETTE[3]),
            ("rewired", "auc_rewired", PLOT_PALETTE[2]),
        ]
        height = 0.8 / len(series)
        for j, (label, key, color) in enumerate(series):
            offset = (len(series) - 1) / 2 * height - j * height
            ax.barh(ys + offset,
                    [results["graphs"][g]["t1"][key] for g in directed],
                    height=height * 0.9, color=color, edgecolor="black",
                    linewidth=0.4, label=label, zorder=3)
        ax.axvline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(directed, fontsize=5)
        ax.set_xlim(0.45, 0.60)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.legend(fontsize=4.5, loc="lower right", frameon=False)
        ax.grid(axis="x", linewidth=0.3, color="#DDDDDD")
        ax.set_axisbelow(True)
    ax.set_xlabel("AUC ($t = 1$)")
    ax.set_title("b  Direction is what the mask throws away", loc="left", fontsize=6,
                 fontweight="bold")

    # (c) decay with walk length, deployed orientation.
    ax = axes[2]
    for i, g in enumerate(names):
        vals = [results["graphs"][g][f"t{t}"]["auc"] for t in WALK_LENGTHS]
        ax.plot(list(WALK_LENGTHS), vals, marker="o", markersize=2.5, linewidth=0.8,
                color=PLOT_PALETTE[i % 18],
                label=g.replace("string12_0_", "string "))
    ax.axhline(0.5, color="black", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlabel("walk length $t$")
    ax.set_ylabel("AUC")
    ax.set_xticks(list(WALK_LENGTHS))
    ax.legend(fontsize=4, ncol=2, frameon=False, loc="best")
    ax.grid(linewidth=0.3, color="#DDDDDD")
    ax.set_axisbelow(True)
    ax.set_title("c  Walk length does not rescue it", loc="left", fontsize=6,
                 fontweight="bold")

    for axis in axes:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    fig.savefig(out_png, dpi=300)
    savefig_true_size_svg(fig, out_svg)
    plt.close(fig)


def main() -> None:
    # --figure-only redraws from the committed JSON without recomputing. The walk and the
    # controls take tens of minutes on the two million-edge graphs, and figure layout wants
    # more iterations than that allows.
    figure_only = "--figure-only" in sys.argv
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    images_dir = os.environ["ASSET_IMAGES_DIR"]
    experiment_root = os.environ["EXPERIMENT_ROOT"]

    results_dir = osp.join(experiment_root, EXPERIMENT, "results")
    out_dir = osp.join(images_dir, EXPERIMENT)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    png = osp.join(out_dir, "graph_prior_probe.png")
    svg = osp.join(out_dir, "graph_prior_probe.svg")
    json_path = osp.join(results_dir, "graph_prior_probe.json")

    if figure_only:
        with open(json_path) as fh:
            existing = json.load(fh)
        make_figure(existing, png, svg)
        print(f"redrew from {json_path}\n-> {svg}")
        return

    deleted, reporters, response = load_response(data_root)
    print(f"deleted genes {len(deleted)}  reporters {len(reporters)}  "
          f"response {response.shape}", flush=True)

    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    scer = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(scer, BUILDER_NAMES)

    # One node index shared by every graph, so an AUC is comparable across them.
    nodes = sorted(set(genome.gene_set) | set(deleted) | set(reporters))
    index = {g: i for i, g in enumerate(nodes)}
    deleted_cols = np.array([index[g] for g in deleted], dtype=np.int64)
    reporter_rows = np.array([index[g] for g in reporters], dtype=np.int64)
    print(f"node index {len(nodes)}", flush=True)

    rng = np.random.default_rng(RANDOM_SEED)
    out: dict[str, object] = {
        "n_deleted": len(deleted),
        "n_reporters": len(reporters),
        "n_nodes": len(nodes),
        "responder_quantile": RESPONDER_QUANTILE,
        "responder_quantile_alt": RESPONDER_QUANTILE_ALT,
        "walk_lengths": list(WALK_LENGTHS),
        "graphs": {},
    }

    for name in GRAPH_NAMES:
        gene_graph = multigraph.graphs.get(CONFIG_TO_BUILDER.get(name, name))
        if gene_graph is None:
            print(f"  {name}: absent from the multigraph, skipped", flush=True)
            continue
        graph = gene_graph.graph
        per_graph: dict[str, object] = {
            "n_nodes": int(graph.number_of_nodes()),
            "n_edges": int(graph.number_of_edges()),
        }
        rewired_graph, degree_preserved = rewired(graph, rng)
        per_graph["rewired_degree_rank_corr"] = degree_preserved
        variants = {
            "": graph,
            "_rewired": rewired_graph,
            "_random": random_matched(graph, rng),
        }
        # The random control is generated with integer node labels, so it is relabeled
        # onto this graph's own node names to share the index.
        node_list = list(graph.nodes())
        variants["_random"] = nx.relabel_nodes(
            variants["_random"], {i: node_list[i] for i in range(len(node_list))}
        )
        # ORIENTATION IS LOAD-BEARING for the two directed graphs (regulatory, tflink) and
        # the deployed answer is SYMMETRIC. `_build_attention_mask` in
        # equivariant_cell_graph_transformer.py sets both
        # `head_mask[edge_index[0]+1, edge_index[1]+1]` and its transpose, so whatever
        # direction `to_cell_data` emitted is symmetrized before the model ever sees it.
        #
        # The symmetric orientation is therefore the one that answers "is the DEPLOYED
        # prior sound", and it is the primary number here. The other two are reported
        # because they are not the same question and they do not give the same answer:
        # `forward` follows edges as stored, `transposed` follows them the other way, and
        # for a TF graph stored as TF -> target only ONE of those is "deleting this TF
        # perturbs its targets". For the seven undirected graphs all three coincide, which
        # makes the extra columns self-checking.
        per_graph["directed"] = bool(graph.is_directed())
        for suffix, variant in variants.items():
            adjacency = row_normalized(variant, index)
            blocks = walk_columns(adjacency, deleted_cols, WALK_LENGTHS)
            for t, block in blocks.items():
                weights = block[reporter_rows, :]
                auc, rho, frac = score(weights, response, RESPONDER_QUANTILE)
                key = f"t{t}"
                per_graph.setdefault(key, {})
                per_graph[key][f"auc{suffix}"] = auc
                per_graph[key][f"spearman{suffix}"] = rho
                per_graph[key][f"scorable_fraction{suffix}"] = frac
                if suffix == "" and t == 1:
                    auc_alt, _, _ = score(weights, response, RESPONDER_QUANTILE_ALT)
                    per_graph[key]["auc_alt_quantile"] = auc_alt
            if suffix == "":
                directed = row_normalized(variant, index, symmetrize=False)
                for label, matrix in (
                    ("forward", directed),
                    ("transposed", directed.T.tocsr()),
                ):
                    for t, block in walk_columns(
                        matrix, deleted_cols, WALK_LENGTHS
                    ).items():
                        weights = block[reporter_rows, :]
                        auc, _rho, _frac = score(
                            weights, response, RESPONDER_QUANTILE
                        )
                        per_graph[f"t{t}"][f"auc_{label}"] = auc
        out["graphs"][name] = per_graph
        t1 = per_graph["t1"]
        print(
            f"  {name:26s} edges={per_graph['n_edges']:>8,}  "
            f"AUC t1 sym {t1['auc']:.4f}  fwd {t1['auc_forward']:.4f}  "
            f"rev {t1['auc_transposed']:.4f}  rewired {t1['auc_rewired']:.4f}  "
            f"random {t1['auc_random']:.4f}  scorable {t1['scorable_fraction']:.2f}",
            flush=True,
        )

    make_figure(out, png, svg)
    out["figure"] = {"png": png, "svg": svg, "written_at": timestamp()}
    with open(json_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"-> {svg}")


if __name__ == "__main__":
    main()
