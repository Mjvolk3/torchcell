# experiments/010-kuzmin-tmi/scripts/graph_statistics.py
# [[experiments.010-kuzmin-tmi.scripts.graph_statistics]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/graph_statistics
"""Statistics and SI panels for the gene--gene graphs that regularize CGT attention.

Every experiment-010 training config (``conf/equivariant_cell_graph_transformer_*.yaml``)
aligns one attention head to each of the same nine graphs: SGD physical, SGD regulatory,
TFLink, and the six STRING v12.0 evidence channels. This script rebuilds those graphs from
the cached ``SCerevisiaeGraph`` pickles, plus the STRING v9.1 and v11.0 channels, and reports

* per-graph size: covered genes, edges, mean degree, density, and the share of a graph's
  edges found in no other graph;
* pairwise overlap: Jaccard index, shared-pair counts, and the asymmetric containment
  |E_i ∩ E_j| / |E_i|;
* how many of the nine graphs support each distinct gene pair (edge multiplicity);
* degree distributions;
* per-graph structure: largest-component fraction, mean clustering coefficient, degree
  assortativity, and two-hop reach (genes within two edges of a gene, averaged);
* hub genes: the top-degree genes of each graph and how often a hub recurs across graphs;
* the two directed transcription-factor graphs (SGD regulatory, TFLink): shared regulators,
  targets, and directed edges, and the per-regulator agreement of target sets;
* STRING release drift: per-channel nodes/edges for v9.1, v11.0, v12.0 and, per consecutive
  release, the pairs retained, added, and dropped.

Directed graphs (regulatory, TFLink) are compared as undirected gene pairs except in the
transcription-factor comparison, which keeps direction; the size table keeps the native
directed edge count. Edges are counted over the S288C gene vocabulary the graph builder
enforces (``genome.gene_set``, 6,607 genes). Structure statistics use genes with at least
one edge to another gene (self-loops dropped).

Outputs
-------
results  : experiments/010-kuzmin-tmi/results/graphs/*.csv
panels   : $ASSET_IMAGES_DIR/010-kuzmin-tmi/graphs_*.{svg,png}   (true-size, draw.io-ready)
tables   : paper/nature-biotech/sections/tab-graphs.tex, tab-string-versions.tex

Run from the repo root:
    python experiments/010-kuzmin-tmi/scripts/graph_statistics.py
"""

import os
import os.path as osp
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FixedLocator, LogLocator, MaxNLocator, NullFormatter

from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    PLOT_PALETTE_FILL,
    mm_to_in,
    savefig_true_size_svg,
)

# Set AFTER the torchcell imports: torchcell.graph applies the repo mplstyle on import,
# which would otherwise override these (bbox=tight inflates the fixed-width panels).
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "hatch.linewidth": 0.4,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.01,
    }
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

RESULTS_DIR = "experiments/010-kuzmin-tmi/results/graphs"
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
TEX_DIR = "paper/nature-biotech/sections"

# The nine attention-prior graphs of experiment 010, in config order. The palette index is
# fixed per graph so every panel colors a graph the same way.
CGT_GRAPHS = [
    # key            display label            source family
    ("physical", "Physical (SGD)", "SGD"),
    ("regulatory", "Regulatory (SGD)", "SGD"),
    ("tflink", "TFLink", "TFLink"),
    ("string12_0_neighborhood", "STRING neighborhood", "STRING v12.0"),
    ("string12_0_fusion", "STRING fusion", "STRING v12.0"),
    ("string12_0_cooccurence", "STRING co-occurrence", "STRING v12.0"),
    ("string12_0_coexpression", "STRING co-expression", "STRING v12.0"),
    ("string12_0_experimental", "STRING experimental", "STRING v12.0"),
    ("string12_0_database", "STRING database", "STRING v12.0"),
]
KEYS = [k for k, _, _ in CGT_GRAPHS]
GRAPH_COLOR = {key: PLOT_PALETTE[i] for i, (key, _, _) in enumerate(CGT_GRAPHS)}
LABEL = {key: lbl for key, lbl, _ in CGT_GRAPHS}

STRING_VERSIONS = ["9_1", "11_0", "12_0"]
STRING_VERSION_LABEL = {"9_1": "v9.1", "11_0": "v11.0", "12_0": "v12.0"}
STRING_CHANNELS = [
    "neighborhood",
    "fusion",
    "cooccurence",
    "coexpression",
    "experimental",
    "database",
]
CHANNEL_LABEL = {
    "neighborhood": "neighborhood",
    "fusion": "fusion",
    "cooccurence": "co-occurrence",
    "coexpression": "co-expression",
    "experimental": "experimental",
    "database": "database",
}

# Sequential colormap for the heatmaps: white to the dark red slot (green-free).
HEAT_CMAP = LinearSegmentedColormap.from_list("tc_heat", ["#FFFFFF", PLOT_PALETTE[7]])

# Hub panel: a gene counts as a hub of a graph when it is among the graph's TOP_K highest-degree
# genes; hubs recurring in at least two graphs are shown. Recurrence is also counted at the
# top 1% of the vocabulary (66 genes per graph).
TOP_K = 10
TOP_FRAC = 0.01

# Shared panel geometry (mm). Panels in one row use the same height and the same top and
# bottom margins so their axes tops and bottoms align in the composed figure.
TOP_MM = 1.5
LABEL_LEFT_MM = 32.0  # axes left for panels with graph names on the y-axis
ROW1_H, ROW2_H, ROW3_H = 54.0, 58.0, 44.0


# ----------------------------------------------------------------------------- loading
def load_graph_builder() -> SCerevisiaeGraph:
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    return SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )


def standard_names(genome: SCerevisiaeGenome) -> dict[str, str]:
    """Systematic id -> SGD standard name (from the GFF ``gene`` attribute)."""
    out = {}
    for std, ids in genome.feature_index["standard_to_ids"].items():
        for i in ids:
            out[i] = std
    return out


def undirected_pairs(G: nx.Graph) -> set[tuple[str, str]]:
    """Distinct unordered gene pairs with an edge, self-loops excluded."""
    return {tuple(sorted((u, v))) for u, v in G.edges() if u != v}


# ------------------------------------------------------------------------- statistics
def size_table(graphs: dict[str, nx.Graph], pairs: dict[str, set], n_vocab: int) -> pd.DataFrame:
    rows = []
    for key, lbl, family in CGT_GRAPHS:
        G = graphs[key]
        E = pairs[key]
        others = set().union(*(pairs[k] for k in pairs if k != key))
        unique = len(E - others)
        n_nodes = G.number_of_nodes()
        rows.append(
            {
                "graph": key,
                "label": lbl,
                "family": family,
                "directed": G.is_directed(),
                "nodes": n_nodes,
                "node_coverage": n_nodes / n_vocab,
                "edges_native": G.number_of_edges(),
                "edges_pairs": len(E),
                "mean_degree": 2 * len(E) / n_nodes,
                "density_vocab": 2 * len(E) / (n_vocab * (n_vocab - 1)),
                "unique_edges": unique,
                "unique_frac": unique / len(E),
            }
        )
    return pd.DataFrame(rows)


def pairwise_table(pairs: dict[str, set]) -> pd.DataFrame:
    rows = []
    for a, b in combinations(KEYS, 2):
        inter = len(pairs[a] & pairs[b])
        union = len(pairs[a] | pairs[b])
        rows.append(
            {
                "graph_a": a,
                "graph_b": b,
                "edges_a": len(pairs[a]),
                "edges_b": len(pairs[b]),
                "shared": inter,
                "union": union,
                "jaccard": inter / union,
                "contain_a_in_b": inter / len(pairs[a]),
                "contain_b_in_a": inter / len(pairs[b]),
                "overlap_coefficient": inter / min(len(pairs[a]), len(pairs[b])),
            }
        )
    return pd.DataFrame(rows)


def multiplicity_table(pairs: dict[str, set]) -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    for E in pairs.values():
        for e in E:
            counts[e] = counts.get(e, 0) + 1
    mult = pd.Series(list(counts.values()), name="n_graphs").value_counts().sort_index()
    df = mult.rename("n_pairs").reset_index().rename(columns={"index": "n_graphs"})
    df["frac_of_union"] = df["n_pairs"] / df["n_pairs"].sum()
    return df


def degree_table(pairs: dict[str, set]) -> pd.DataFrame:
    rows = []
    for key in pairs:
        H = nx.Graph()
        H.add_edges_from(pairs[key])
        deg = pd.Series(dict(H.degree())).value_counts().sort_index()
        for d, c in deg.items():
            rows.append({"graph": key, "degree": int(d), "n_nodes": int(c)})
    return pd.DataFrame(rows)


def structure_table(pairs: dict[str, set], n_vocab: int) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Per-graph structure from the dense adjacency over covered genes.

    Mean clustering follows ``nx.average_clustering`` (genes of degree < 2 contribute 0);
    triangles come from diag(A^3)/2. Two-hop reach of a gene is the number of other genes at
    distance 1 or 2. Returns the table and each graph's degree series (for the hub panel).
    """
    rows, degrees = [], {}
    for key in KEYS:
        H = nx.Graph()
        H.add_edges_from(pairs[key])
        nodes = list(H.nodes)
        n = len(nodes)
        li = {g: i for i, g in enumerate(nodes)}
        A = np.zeros((n, n), dtype=np.float32)
        for u, v in pairs[key]:
            A[li[u], li[v]] = 1.0
            A[li[v], li[u]] = 1.0
        d = A.sum(axis=1)
        A2 = A @ A
        tri = (A2 * A).sum(axis=1) / 2.0
        clust = np.zeros(n)
        m = d >= 2
        clust[m] = 2.0 * tri[m] / (d[m] * (d[m] - 1.0))
        reach2 = ((A2 > 0) | (A > 0)).sum(axis=1) - 1  # exclude the gene itself
        lcc = max(len(c) for c in nx.connected_components(H))
        rows.append(
            {
                "graph": key,
                "label": LABEL[key],
                "nodes": n,
                "components": nx.number_connected_components(H),
                "lcc_nodes": lcc,
                "lcc_frac": lcc / n,
                "mean_clustering": float(clust.mean()),
                "degree_assortativity": float(nx.degree_assortativity_coefficient(H)),
                "mean_degree": float(d.mean()),
                "mean_two_hop_reach": float(reach2.mean()),
                "median_two_hop_reach": float(np.median(reach2)),
                "mean_two_hop_frac_vocab": float((reach2 / (n_vocab - 1)).mean()),
            }
        )
        degrees[key] = pd.Series(d.astype(int), index=nodes)
    return pd.DataFrame(rows), degrees


def hub_tables(degrees: dict[str, pd.Series], std: dict[str, str], n_vocab: int):
    """Top-degree genes per graph, hub recurrence at TOP_K and at the top 1%, and the
    percentile-rank matrix of the hubs that recur in at least two graphs."""
    ranks = {k: s.rank(ascending=False, method="min").astype(int) for k, s in degrees.items()}

    def by_degree(s: pd.Series) -> pd.Series:
        # Degree descending, ties broken by systematic name so the ordering (and the
        # written CSV) is identical run to run; a plain sort_values leaves ties arbitrary.
        return s.sort_index().sort_values(ascending=False, kind="stable")

    hub_rows = []
    for k in KEYS:
        s = by_degree(degrees[k])
        for r, (g, dv) in enumerate(s.iloc[:TOP_K].items(), start=1):
            hub_rows.append({"graph": k, "rank": r, "gene": g, "name": std.get(g, g), "degree": int(dv)})
    hubs = pd.DataFrame(hub_rows)

    n_top = int(round(TOP_FRAC * n_vocab))
    rec_rows = []
    for label, kk in (("top_k", TOP_K), ("top_1pct", n_top)):
        cnt = pd.Series([g for k in KEYS for g in by_degree(degrees[k]).index[:kk]]).value_counts()
        hist = cnt.value_counts().sort_index()
        for n_graphs, n_genes in hist.items():
            rec_rows.append({"criterion": label, "genes_per_graph": kk, "n_graphs": int(n_graphs), "n_genes": int(n_genes)})
    recurrence = pd.DataFrame(rec_rows)

    cnt_k = hubs["gene"].value_counts()
    recurring = cnt_k[cnt_k >= 2]
    order = sorted(recurring.index, key=lambda g: (-recurring[g], std.get(g, g)))
    mat_rows = []
    for g in order:
        for k in KEYS:
            present = g in degrees[k].index
            mat_rows.append(
                {
                    "gene": g,
                    "name": std.get(g, g),
                    "n_graphs_top_k": int(recurring[g]),
                    "graph": k,
                    "degree": int(degrees[k][g]) if present else 0,
                    "rank": int(ranks[k][g]) if present else np.nan,
                    "percentile": 100.0 * (1.0 - (ranks[k][g] - 1) / len(degrees[k])) if present else np.nan,
                }
            )
    matrix = pd.DataFrame(mat_rows)
    return hubs, recurrence, matrix


def tf_overlap_tables(graphs: dict[str, nx.Graph], std: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SGD regulatory vs TFLink as DIRECTED regulator -> target graphs, self-regulation kept
    (the native directed edge sets)."""
    Er = set(graphs["regulatory"].edges())
    Et = set(graphs["tflink"].edges())
    sets = {
        "regulators": ({u for u, _ in Er}, {u for u, _ in Et}),
        "targets": ({v for _, v in Er}, {v for _, v in Et}),
        "directed_edges": (Er, Et),
    }
    rows = []
    for entity, (a, b) in sets.items():
        rows.append(
            {
                "entity": entity,
                "regulatory": len(a),
                "tflink": len(b),
                "regulatory_only": len(a - b),
                "both": len(a & b),
                "tflink_only": len(b - a),
                "union": len(a | b),
                "jaccard": len(a & b) / len(a | b),
            }
        )
    overlap = pd.DataFrame(rows)
    shared = sets["regulators"][0] & sets["regulators"][1]
    tr, tt = {}, {}
    for u, v in Er:
        tr.setdefault(u, set()).add(v)
    for u, v in Et:
        tt.setdefault(u, set()).add(v)
    per_tf = pd.DataFrame(
        [
            {
                "regulator": g,
                "name": std.get(g, g),
                "regulatory_targets": len(tr[g]),
                "tflink_targets": len(tt[g]),
                "shared_targets": len(tr[g] & tt[g]),
                "jaccard": len(tr[g] & tt[g]) / len(tr[g] | tt[g]),
            }
            for g in sorted(shared)
        ]
    ).sort_values(["shared_targets", "regulator"], ascending=[False, True], kind="stable")
    # `shared` is a set, so without the sort the row order (and ties in shared_targets)
    # would follow Python's per-process string hash seed.
    return overlap, per_tf


def string_version_table(builder: SCerevisiaeGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-channel sizes per release and the pairwise drift between releases. For each
    ordered pair (a before b): shared = pairs in both, added = in b only, dropped = in a
    only; ``consecutive`` marks v9.1 -> v11.0 and v11.0 -> v12.0."""
    sizes, drift = [], []
    for ch in STRING_CHANNELS:
        E = {}
        for v in STRING_VERSIONS:
            G = getattr(builder, f"G_string{v}_{ch}").graph
            E[v] = undirected_pairs(G)
            sizes.append(
                {
                    "channel": ch,
                    "version": STRING_VERSION_LABEL[v],
                    "nodes": G.number_of_nodes(),
                    "edges": len(E[v]),
                }
            )
        for ia, ib in combinations(range(len(STRING_VERSIONS)), 2):
            va, vb = STRING_VERSIONS[ia], STRING_VERSIONS[ib]
            inter = len(E[va] & E[vb])
            drift.append(
                {
                    "channel": ch,
                    "version_a": STRING_VERSION_LABEL[va],
                    "version_b": STRING_VERSION_LABEL[vb],
                    "consecutive": ib == ia + 1,
                    "edges_a": len(E[va]),
                    "edges_b": len(E[vb]),
                    "shared": inter,
                    "added": len(E[vb] - E[va]),
                    "dropped": len(E[va] - E[vb]),
                    "jaccard": inter / len(E[va] | E[vb]),
                    "retained_frac_of_a": inter / len(E[va]),
                    "retained_frac_of_b": inter / len(E[vb]),
                }
            )
    return pd.DataFrame(sizes), pd.DataFrame(drift)


# ------------------------------------------------------------------------------ panels
def _box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")


def _ticks(ax):
    ax.tick_params(length=2, width=0.5)


def _save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, f"{name}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def _fig(w_mm: float, h_mm: float):
    return plt.figure(figsize=(mm_to_in(w_mm), mm_to_in(h_mm)))


def _axes_mm(fig, w_mm, h_mm, left, bottom, width, height):
    """Add an axes by its rectangle in mm from the panel's bottom-left corner."""
    return fig.add_axes([left / w_mm, bottom / h_mm, width / w_mm, height / h_mm])


def _knum(v: float) -> str:
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def _named_barh_row(fig, w_mm, h_mm, n_axes, widths, bottom_mm):
    """A row of barh axes sharing the graph-name y-axis at LABEL_LEFT_MM, top at TOP_MM."""
    right_mm = 1.5
    gap_mm = 2.5
    avail = w_mm - LABEL_LEFT_MM - right_mm - gap_mm * (n_axes - 1)
    unit = avail / sum(widths)
    axes, x = [], LABEL_LEFT_MM
    height = h_mm - TOP_MM - bottom_mm
    for wgt in widths:
        ax = _axes_mm(fig, w_mm, h_mm, x, bottom_mm, wgt * unit, height)
        axes.append(ax)
        x += wgt * unit + gap_mm
    return axes


def panel_sizes(sizes: pd.DataFrame):
    """Half-width panel, one bar per graph: covered genes, edges (log axis), and the share of a
    graph's edges found in no other graph."""
    w, h = PANEL_WIDTHS_MM["half"], ROW1_H
    fig = _fig(w, h)
    ax_n, ax_e, ax_u = _named_barh_row(fig, w, h, 3, [1, 1.3, 1], bottom_mm=8.5)
    y = np.arange(len(sizes))[::-1]
    colors = [GRAPH_COLOR[k] for k in sizes["graph"]]
    ax_n.barh(y, sizes["nodes"], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax_n.axvline(6607, color="black", linewidth=0.5, linestyle=":")
    ax_n.set_xlim(0, 7000)
    ax_n.set_xticks([0, 3000, 6000])
    ax_n.set_xticklabels(["0", "3k", "6k"])
    ax_n.set_xlabel("Genes")
    ax_n.set_yticks(y)
    ax_n.set_yticklabels(sizes["label"])
    ax_e.barh(y, sizes["edges_pairs"], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax_e.set_xscale("log")
    ax_e.set_xlim(1e3, 3e6)
    ax_e.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax_e.set_xticklabels(["$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax_e.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax_e.xaxis.set_minor_formatter(NullFormatter())
    ax_e.set_xlabel("Edges")
    ax_u.barh(y, sizes["unique_frac"], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
    ax_u.set_xlim(0, 1)
    ax_u.set_xticks([0, 0.5, 1.0])
    ax_u.set_xticklabels(["0", "0.5", "1"])
    ax_u.xaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.1)))
    ax_u.tick_params(which="minor", length=0)
    ax_u.set_xlabel("Unique fraction")
    for ax in (ax_n, ax_e, ax_u):
        ax.set_ylim(-0.6, len(sizes) - 0.4)
        if ax is not ax_n:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        _box(ax)
        ax.grid(axis="x", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        _ticks(ax)
    _save(fig, "graphs_sizes")


def panel_degree(deg: pd.DataFrame):
    """Half-width panel: complementary CDF of undirected degree, log-log, one line per graph.
    Same height and top margin as panel_sizes so the two axes tops align."""
    w, h = PANEL_WIDTHS_MM["half"], ROW1_H
    fig = _fig(w, h)
    bottom = 16.5  # x label + three legend rows
    ax = _axes_mm(fig, w, h, 9.5, bottom, w - 9.5 - 1.5, h - TOP_MM - bottom)
    for key, lbl, _ in CGT_GRAPHS:
        d = deg[deg["graph"] == key].sort_values("degree")
        n = d["n_nodes"].to_numpy()
        ccdf = 1 - np.cumsum(n) / n.sum() + n / n.sum()  # P(degree >= k)
        ax.plot(d["degree"], ccdf, color=GRAPH_COLOR[key], linewidth=0.9, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("Degree k", labelpad=1.5)
    ax.set_ylabel("Fraction of genes, degree ≥ k")
    ax.grid(color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    _ticks(ax)
    fig.legend(frameon=False, loc="lower left", ncol=3, handlelength=1.0, columnspacing=0.6,
               handletextpad=0.4, labelspacing=0.25, bbox_to_anchor=(9.5 / w, 0.0), borderaxespad=0.2)
    _box(ax)
    _save(fig, "graphs_degree_ccdf")


def _heatmap(matrix: np.ndarray, labels: list[str], name: str, fmt, cbar_label: str,
             tri: str | None, norm=None, vmax: float | None = None, dark_above: float = 0.6):
    """Half-width square heatmap with in-cell annotations, explicit geometry: rows labeled at
    LABEL_LEFT_MM, columns labeled on top at 45 degrees, colorbar on the right."""
    w, h = PANEL_WIDTHS_MM["half"], ROW2_H
    fig = _fig(w, h)
    top_labels, bottom = 17.0, 1.0
    side = h - top_labels - bottom
    ax = _axes_mm(fig, w, h, LABEL_LEFT_MM, bottom, side, side)
    M = matrix.astype(float).copy()
    if tri == "lower":
        M[np.triu_indices_from(M, k=1)] = np.nan
    kw = {"norm": norm} if norm is not None else {"vmin": 0, "vmax": vmax}
    im = ax.imshow(M, cmap=HEAT_CMAP, aspect="auto", **kw)
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]):
                continue
            v = M[i, j]
            frac = im.norm(v)
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=5,
                    color="white" if frac > dark_above else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    _box(ax)
    cax = _axes_mm(fig, w, h, LABEL_LEFT_MM + side + 2.5, bottom + 0.1 * side, 2.2, 0.8 * side)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5)
    _save(fig, name)


def panel_overlap(pw: pd.DataFrame, sizes: pd.DataFrame):
    labels = [LABEL[k] for k in KEYS]
    n = len(KEYS)
    J = np.eye(n)
    C = np.eye(n)
    S = np.diag(sizes.set_index("graph").loc[KEYS, "edges_pairs"].to_numpy().astype(float))
    for _, r in pw.iterrows():
        i, j = KEYS.index(r["graph_a"]), KEYS.index(r["graph_b"])
        J[i, j] = J[j, i] = r["jaccard"]
        C[i, j] = r["contain_a_in_b"]  # row graph's edges found in column graph
        C[j, i] = r["contain_b_in_a"]
        S[i, j] = S[j, i] = r["shared"]
    _heatmap(J, labels, "graphs_jaccard", fmt=lambda v: f"{v:.2f}", vmax=0.5,
             cbar_label="Jaccard index", tri="lower")
    _heatmap(C, labels, "graphs_containment", fmt=lambda v: f"{v:.2f}", vmax=1.0,
             cbar_label="Row edges in column graph", tri=None)
    S[S < 1] = 1.0
    _heatmap(S, labels, "graphs_shared_pairs", fmt=_knum, norm=LogNorm(vmin=1e2, vmax=1e6),
             cbar_label="Shared gene pairs", tri="lower", dark_above=0.7)


def panel_multiplicity(mult: pd.DataFrame, sizes: pd.DataFrame):
    """Third-width panel: distinct gene pairs supported by exactly n of the nine graphs."""
    w, h = PANEL_WIDTHS_MM["third"], ROW3_H
    fig = _fig(w, h)
    bottom = 8.5
    ax = _axes_mm(fig, w, h, 11.0, bottom, w - 11.0 - 1.5, h - TOP_MM - bottom)
    ax.bar(mult["n_graphs"], mult["n_pairs"], color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, width=0.7)
    for x, y in zip(mult["n_graphs"], mult["n_pairs"]):
        ax.text(x, y * 1.25, f"{y:,}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Supporting graphs")
    ax.set_ylabel("Gene pairs")
    ax.grid(axis="y", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    _ticks(ax)
    _box(ax)
    _save(fig, "graphs_edge_multiplicity")


def panel_structure(struct: pd.DataFrame):
    """Wide panel, one bar per graph: largest-component fraction, mean clustering coefficient,
    degree assortativity, and mean two-hop reach. Same row geometry as panel_multiplicity."""
    w, h = PANEL_WIDTHS_MM["wide"], ROW3_H
    fig = _fig(w, h)
    axes = _named_barh_row(fig, w, h, 4, [1, 1, 1.15, 1], bottom_mm=8.5)
    y = np.arange(len(struct))[::-1]
    colors = [GRAPH_COLOR[k] for k in struct["graph"]]
    cols = [
        ("lcc_frac", "Largest component", (0, 1.0), [0, 0.5, 1.0], ["0", "0.5", "1"]),
        ("mean_clustering", "Mean clustering", (0, 1.0), [0, 0.5, 1.0], ["0", "0.5", "1"]),
        ("degree_assortativity", "Degree assortativity", (-0.6, 1.0), [-0.5, 0, 0.5, 1.0], ["−0.5", "0", "0.5", "1"]),
        ("mean_two_hop_reach", "Two-hop reach (genes)", (0, 7000), [0, 3000, 6000], ["0", "3k", "6k"]),
    ]
    for ax, (col, xlabel, xlim, ticks, ticklabels) in zip(axes, cols):
        ax.barh(y, struct[col], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.6, len(struct) - 0.4)
        ax.set_yticks(y)
        if col == "degree_assortativity":
            ax.axvline(0, color="black", linewidth=0.5)
        if col == "mean_two_hop_reach":
            ax.axvline(6607, color="black", linewidth=0.5, linestyle=":")
        if col in ("lcc_frac", "mean_clustering"):
            ax.xaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.1)))
            ax.tick_params(which="minor", length=0)
        _box(ax)
        ax.grid(axis="x", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        _ticks(ax)
    axes[0].set_yticklabels(struct["label"])
    for ax in axes[1:]:
        ax.set_yticklabels([])
    _save(fig, "graphs_structure")


def panel_hubs(matrix: pd.DataFrame):
    """Half-width heatmap: hubs recurring in at least two graphs (rows) by graph (columns),
    colored by the gene's degree percentile in that graph; ranks <= TOP_K annotated."""
    w, h = PANEL_WIDTHS_MM["half"], ROW2_H
    genes = list(dict.fromkeys(matrix["gene"]))
    names = [matrix[matrix["gene"] == g]["name"].iloc[0] for g in genes]
    P = matrix.pivot(index="gene", columns="graph", values="percentile").loc[genes, KEYS].to_numpy()
    R = matrix.pivot(index="gene", columns="graph", values="rank").loc[genes, KEYS].to_numpy()
    fig = _fig(w, h)
    top_labels, bottom = 17.0, 1.0
    cell_w = 4.4
    ax = _axes_mm(fig, w, h, LABEL_LEFT_MM, bottom, cell_w * len(KEYS), h - top_labels - bottom)
    im = ax.imshow(P, cmap=HEAT_CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(len(genes)):
        for j in range(len(KEYS)):
            if np.isnan(P[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=5, color="#888888")
            elif R[i, j] <= TOP_K:
                ax.text(j, i, f"{int(R[i, j])}", ha="center", va="center", fontsize=5, fontweight="bold",
                        color="white" if P[i, j] > 60 else "black")
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([LABEL[k] for k in KEYS], rotation=45, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(names)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(-0.5, len(KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    _box(ax)
    side = h - top_labels - bottom
    cax = _axes_mm(fig, w, h, LABEL_LEFT_MM + cell_w * len(KEYS) + 2.5, bottom + 0.1 * side, 2.2, 0.8 * side)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Degree percentile in graph")
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5)
    _save(fig, "graphs_hubs")


def panel_tf_overlap(overlap: pd.DataFrame, per_tf: pd.DataFrame):
    """Half-width panel. Top: regulators, targets, and directed edges split into
    regulatory-only / both / TFLink-only. Bottom: per shared regulator, Jaccard of its two
    target sets."""
    w, h = PANEL_WIDTHS_MM["half"], ROW2_H
    fig = _fig(w, h)
    seg_color = [PLOT_PALETTE[1], PLOT_PALETTE[5], PLOT_PALETTE[2]]
    seg_label = ["Regulatory (SGD) only", "Both", "TFLink only"]
    rows = [("regulators", "Regulators"), ("targets", "Targets"), ("directed_edges", "Directed edges")]
    left, right = 22.0, 1.5
    bar_h, bar_gap = 5.2, 1.6
    top_block = TOP_MM + 4.0  # legend row
    for i, (ent, lbl) in enumerate(rows):
        r = overlap[overlap["entity"] == ent].iloc[0]
        y0 = h - top_block - (i + 1) * bar_h - i * bar_gap
        ax = _axes_mm(fig, w, h, left, y0, w - left - right, bar_h)
        total = r["union"]
        x = 0.0
        prev_small = False
        for val, c, lab in zip([r["regulatory_only"], r["both"], r["tflink_only"]], seg_color, seg_label):
            ax.barh(0, val, left=x, color=c, edgecolor="black", linewidth=0.4, height=0.7,
                    label=lab if i == 0 else None)
            frac = val / total
            txt = f"{int(val):,}"
            if frac >= 0.11:
                ax.text(x + val / 2, 0, txt, ha="center", va="center", fontsize=5, color="white")
                prev_small = False
            elif prev_small:  # two narrow segments in a row: label the second below the bar
                ax.text(x + val / 2, -0.42, txt, ha="center", va="top", fontsize=5)
                prev_small = False
            else:
                ax.text(x + val / 2, 0.42, txt, ha="center", va="bottom", fontsize=5)
                prev_small = True
            x += val
        ax.set_xlim(0, total)
        ax.set_ylim(-0.95, 0.95)
        ax.set_xticks([])
        ax.set_yticks([0])
        ax.set_yticklabels([f"{lbl}\n(n = {int(total):,})"])
        ax.tick_params(length=0, pad=2)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.0,
                      columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    # bottom: per-regulator target agreement
    hist_top = h - top_block - 3 * bar_h - 2 * bar_gap - 5.0
    bottom = 8.5
    ax = _axes_mm(fig, w, h, 12.0, bottom, w - 12.0 - right, hist_top - bottom)
    bins = np.arange(0, 1.001, 0.05)
    ax.hist(per_tf["jaccard"], bins=bins, color=PLOT_PALETTE[5], edgecolor="black", linewidth=0.4)
    med = per_tf["jaccard"].median()
    ax.axvline(med, color="black", linewidth=0.6, linestyle="--")
    top_tf = per_tf.iloc[0]  # most shared targets
    ax.text(0.55, ax.get_ylim()[1] * 0.92,
            f"median {med:.2f} (dashed)\n{top_tf['name']}: {top_tf['jaccard']:.2f}, "
            f"{int(top_tf['shared_targets']):,} shared targets",
            ha="center", va="top", fontsize=6)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.2)))
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel(f"Jaccard of target sets, {len(per_tf)} shared regulators", labelpad=1.5)
    ax.set_ylabel("Regulators")
    ax.grid(axis="y", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    _ticks(ax)
    _box(ax)
    _save(fig, "graphs_tf_overlap")


def panel_string_releases(sizes: pd.DataFrame, drift: pd.DataFrame):
    """Half-width panel, one small axes per channel, releases in time order. Each release's bar
    is the pairs in that release; the solid part is what the previous release already had, the
    pale part is new; the hatched bar below zero is what the previous release lost."""
    w, h = PANEL_WIDTHS_MM["half"], ROW2_H
    fig = _fig(w, h)
    ncol, nrow = 3, 2
    left, right, top, bottom = 11.0, 1.5, 6.5, 7.5
    wgap, hgap = 7.5, 8.0
    aw = (w - left - right - wgap * (ncol - 1)) / ncol
    ah = (h - top - bottom - hgap * (nrow - 1)) / nrow
    vers = [STRING_VERSION_LABEL[v] for v in STRING_VERSIONS]
    c_in, c_keep, c_drop = PLOT_PALETTE_FILL[0], PLOT_PALETTE[0], PLOT_PALETTE[5]
    for idx, ch in enumerate(STRING_CHANNELS):
        r, c = divmod(idx, ncol)
        ax = _axes_mm(fig, w, h, left + c * (aw + wgap), h - top - (r + 1) * ah - r * hgap, aw, ah)
        edges = [sizes[(sizes["channel"] == ch) & (sizes["version"] == v)]["edges"].item() for v in vers]
        x = np.arange(3)
        ax.bar(x, edges, 0.62, color=c_in, edgecolor="black", linewidth=0.4, label="In release")
        keep, drop = [0], [0]
        for va, vb in zip(vers[:-1], vers[1:]):
            d = drift[(drift["channel"] == ch) & (drift["version_a"] == va) & (drift["version_b"] == vb)].iloc[0]
            keep.append(d["shared"])
            drop.append(d["dropped"])
        ax.bar(x, keep, 0.62, color=c_keep, edgecolor="black", linewidth=0.4, label="Retained from previous")
        ax.bar(x, [-v for v in drop], 0.62, color="white", edgecolor="black", linewidth=0.4, hatch="////",
               label="Dropped from previous")
        ax.axhline(0, color="black", linewidth=0.5)
        ymax = max(edges) * 1.08
        ymin = -max(drop) * 1.15 if max(drop) > 0 else 0
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ticks = [t for t in ax.get_yticks() if ymin <= t <= ymax]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([_knum(abs(t)) if t >= 0 else "−" + _knum(abs(t)) for t in ticks])
        ax.set_xticks(x)
        ax.set_xticklabels([v[1:] for v in vers])
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(CHANNEL_LABEL[ch], pad=2)
        ax.grid(axis="y", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        _ticks(ax)
        _box(ax)
        if idx == 0:
            fig.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(left / w, 1.0),
                       handlelength=1.0, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.2)
    fig.text(0.6 / w, 0.5, "Gene pairs", rotation=90, ha="left", va="center")
    fig.text(0.5 + (left - right) / (2 * w), 0.3 / h, "STRING release", ha="center", va="bottom")
    _save(fig, "graphs_string_releases")


# ------------------------------------------------------------------------------ tables
def write_tex_tables(sizes: pd.DataFrame, mult: pd.DataFrame, sv: pd.DataFrame, drift: pd.DataFrame):
    src = "experiments/010-kuzmin-tmi/scripts/graph_statistics.py"
    union = int(mult["n_pairs"].sum())
    sgd = sizes[sizes["family"] == "SGD"]
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        "%% Node = gene with at least one edge; Edges = native count (directed for regulatory and",
        "%% TFLink); Pairs = distinct unordered gene pairs; Unique = pairs found in no other graph.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Gene--gene graphs that serve as attention priors in the trigenic experiment,",
        "over the shared S288C vocabulary of 6,607 genes. \\textbf{Physical} and \\textbf{Regulatory}",
        "are the curated SGD graphs; TFLink and the six STRING~v12.0 evidence channels add the",
        "remaining rows. \\emph{Edges} is the native count (directed for Regulatory and TFLink),",
        "\\emph{Pairs} the distinct unordered gene pairs used for overlap, \\emph{Unique} the share of",
        "those pairs found in no other graph. The nine graphs together cover",
        f"{union:,} distinct gene pairs; relative to the SGD-native baseline (Physical\\,$+$\\,Regulatory),",
        f"the sum over graphs is {sizes['nodes'].sum() / sgd['nodes'].sum():.2f}$\\times$ in nodes and",
        f"{sizes['edges_native'].sum() / sgd['edges_native'].sum():.2f}$\\times$ in edges.",
        "(STRING~v9.1 and v11.0 are in \\supptab{tab:string-versions}.)}",
        "\\label{tab:databases}",
        "\\begin{tabular}{@{}l r r r r r@{}}",
        "\\toprule",
        "\\textbf{Graph} & \\textbf{Nodes} & \\textbf{Edges} & \\textbf{Pairs} & \\textbf{Mean degree} & \\textbf{Unique}\\\\",
        "\\midrule",
    ]
    for _, r in sizes.iterrows():
        lines.append(
            f"{r['label']} & {r['nodes']:,} & {r['edges_native']:,} & {r['edges_pairs']:,} & "
            f"{r['mean_degree']:.1f} & {100 * r['unique_frac']:.0f}\\%\\\\"
        )
    lines += [
        "\\midrule",
        f"Sum (all graphs) & {sizes['nodes'].sum():,} & {sizes['edges_native'].sum():,} & {sizes['edges_pairs'].sum():,} & & \\\\",
        f"Union of pairs & & & {union:,} & & \\\\",
        f"Sum (Physical, Regulatory) & {sgd['nodes'].sum():,} & {sgd['edges_native'].sum():,} & {sgd['edges_pairs'].sum():,} & & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    with open(osp.join(TEX_DIR, "tab-graphs.tex"), "w") as f:
        f.write("\n".join(lines))

    piv_e = sv.pivot(index="channel", columns="version", values="edges").reindex(STRING_CHANNELS)
    piv_n = sv.pivot(index="channel", columns="version", values="nodes").reindex(STRING_CHANNELS)
    vers = ["v9.1", "v11.0", "v12.0"]
    steps = [("v9.1", "v11.0"), ("v11.0", "v12.0")]

    def retained(ch, va, vb):
        d = drift[(drift["channel"] == ch) & (drift["version_a"] == va) & (drift["version_b"] == vb)].iloc[0]
        return 100 * d["retained_frac_of_a"]

    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{STRING evidence channels for \\emph{S.\\ cerevisiae} across the three releases",
        "integrated into TorchCell, over the shared 6,607-gene vocabulary. Nodes are genes with at",
        "least one edge; edges are distinct unordered gene pairs with a nonzero channel score;",
        "\\emph{Retained} is the share of a release's pairs still present in the next release.",
        "DANGO was published on v9.1 with a v9.1$\\to$v11.0 comparison; the CGT trigenic",
        "experiment uses v12.0.}",
        "\\label{tab:string-versions}",
        "\\begin{tabular}{@{}l rrr rrr rr@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{Nodes}} & \\multicolumn{3}{c}{\\textbf{Edges}} & \\multicolumn{2}{c}{\\textbf{Retained}}\\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}",
        "\\textbf{Channel} & " + " & ".join(vers) + " & " + " & ".join(vers)
        + " & " + " & ".join(f"{a[1:]}$\\to${b[1:]}" for a, b in steps) + "\\\\",
        "\\midrule",
    ]
    for ch in STRING_CHANNELS:
        lines.append(
            f"{CHANNEL_LABEL[ch]} & " + " & ".join(f"{int(piv_n.loc[ch, v]):,}" for v in vers)
            + " & " + " & ".join(f"{int(piv_e.loc[ch, v]):,}" for v in vers)
            + " & " + " & ".join(f"{retained(ch, a, b):.0f}\\%" for a, b in steps) + "\\\\"
        )
    lines += [
        "\\midrule",
        "Sum & " + " & ".join(f"{int(piv_n[v].sum()):,}" for v in vers)
        + " & " + " & ".join(f"{int(piv_e[v].sum()):,}" for v in vers) + " & & \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    with open(osp.join(TEX_DIR, "tab-string-versions.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {TEX_DIR}/tab-graphs.tex and tab-string-versions.tex")


# -------------------------------------------------------------------------------- main
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    builder = load_graph_builder()
    n_vocab = len(builder.genome.gene_set)
    std = standard_names(builder.genome)
    print(f"gene vocabulary: {n_vocab}")

    graphs = {key: getattr(builder, f"G_{key}").graph for key in KEYS}
    pairs = {key: undirected_pairs(G) for key, G in graphs.items()}
    for key, G in graphs.items():
        print(f"{key:26s} nodes={G.number_of_nodes():5d} edges={G.number_of_edges():9,d} pairs={len(pairs[key]):9,d}")

    sizes = size_table(graphs, pairs, n_vocab)
    pw = pairwise_table(pairs)
    mult = multiplicity_table(pairs)
    deg = degree_table(pairs)
    struct, degrees = structure_table(pairs, n_vocab)
    hubs, recurrence, hub_matrix = hub_tables(degrees, std, n_vocab)
    tf_overlap, per_tf = tf_overlap_tables(graphs, std)
    sv_sizes, sv_drift = string_version_table(builder)

    sizes.to_csv(osp.join(RESULTS_DIR, "graph_sizes.csv"), index=False)
    pw.to_csv(osp.join(RESULTS_DIR, "pairwise_overlap.csv"), index=False)
    mult.to_csv(osp.join(RESULTS_DIR, "edge_multiplicity.csv"), index=False)
    deg.to_csv(osp.join(RESULTS_DIR, "degree_distribution.csv"), index=False)
    # 10 significant digits: the assortativity and two-hop means differ in the last
    # floating-point digit between BLAS builds, which otherwise churns the committed CSV.
    struct.to_csv(osp.join(RESULTS_DIR, "graph_structure.csv"), index=False, float_format="%.10g")
    hubs.to_csv(osp.join(RESULTS_DIR, "hub_genes.csv"), index=False)
    recurrence.to_csv(osp.join(RESULTS_DIR, "hub_recurrence.csv"), index=False)
    hub_matrix.to_csv(osp.join(RESULTS_DIR, "hub_matrix.csv"), index=False)
    tf_overlap.to_csv(osp.join(RESULTS_DIR, "tf_overlap.csv"), index=False)
    per_tf.to_csv(osp.join(RESULTS_DIR, "tf_overlap_per_regulator.csv"), index=False)
    sv_sizes.to_csv(osp.join(RESULTS_DIR, "string_version_sizes.csv"), index=False)
    sv_drift.to_csv(osp.join(RESULTS_DIR, "string_version_drift.csv"), index=False)
    print(f"wrote CSVs to {RESULTS_DIR}")
    print(sizes[["label", "nodes", "edges_native", "edges_pairs", "mean_degree", "unique_frac"]].to_string(index=False))
    print(mult.to_string(index=False))
    print(struct[["label", "lcc_frac", "mean_clustering", "degree_assortativity", "mean_two_hop_reach"]].to_string(index=False))
    print(recurrence.to_string(index=False))
    print(hubs[hubs["rank"] <= 3].to_string(index=False))
    print(tf_overlap.to_string(index=False))
    print(per_tf.head(8).to_string(index=False))
    print(sv_drift[["channel", "version_a", "version_b", "shared", "added", "dropped", "retained_frac_of_a"]].to_string(index=False))

    panel_sizes(sizes)
    panel_degree(deg)
    panel_overlap(pw, sizes)
    panel_multiplicity(mult, sizes)
    panel_structure(struct)
    panel_hubs(hub_matrix)
    panel_tf_overlap(tf_overlap, per_tf)
    panel_string_releases(sv_sizes, sv_drift)
    write_tex_tables(sizes, mult, sv_sizes, sv_drift)


if __name__ == "__main__":
    main()
