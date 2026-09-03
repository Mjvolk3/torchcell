# experiments/010-kuzmin-tmi/scripts/graph_statistics.py
# [[experiments.010-kuzmin-tmi.scripts.graph_statistics]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/graph_statistics
"""Statistics and SI panels for the gene--gene graphs that regularize CGT attention.

Every experiment-010 training config (``conf/equivariant_cell_graph_transformer_*.yaml``)
aligns one attention head to each of the same nine graphs: SGD physical, SGD regulatory,
TFLink, and the six STRING v12.0 evidence channels. This script rebuilds those graphs from
the cached ``SCerevisiaeGraph`` pickles, plus the STRING v9.1 and v11.0 channels used in the
DANGO replication, and reports

* per-graph size: covered genes, edges, mean degree, density, and the share of a graph's
  edges found in no other graph;
* pairwise overlap: Jaccard index and the asymmetric containment |E_i ∩ E_j| / |E_i|;
* how many of the nine graphs support each distinct gene pair (edge multiplicity);
* degree distributions;
* STRING version drift: per-channel nodes/edges for v9.1, v11.0, v12.0 and the Jaccard
  index between versions of the same channel.

Directed graphs (regulatory, TFLink) are compared as undirected gene pairs; the table keeps
the native directed edge count. Edges are counted over the S288C gene vocabulary the graph
builder enforces (``genome.gene_set``, 6,607 genes).

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
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
VERSION_COLOR = {v: PLOT_PALETTE[i] for i, v in enumerate(STRING_VERSIONS)}

# Sequential colormap for the overlap heatmaps: white to the dark red slot (green-free).
HEAT_CMAP = LinearSegmentedColormap.from_list("tc_heat", ["#FFFFFF", PLOT_PALETTE[7]])


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
    keys = [k for k, _, _ in CGT_GRAPHS]
    rows = []
    for a, b in combinations(keys, 2):
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


def string_version_table(builder: SCerevisiaeGraph) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        for va, vb in combinations(STRING_VERSIONS, 2):
            inter = len(E[va] & E[vb])
            drift.append(
                {
                    "channel": ch,
                    "version_a": STRING_VERSION_LABEL[va],
                    "version_b": STRING_VERSION_LABEL[vb],
                    "edges_a": len(E[va]),
                    "edges_b": len(E[vb]),
                    "shared": inter,
                    "jaccard": inter / len(E[va] | E[vb]),
                    "retained_frac_of_a": inter / len(E[va]),
                }
            )
    return pd.DataFrame(sizes), pd.DataFrame(drift)


# ------------------------------------------------------------------------------ panels
def _box(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("black")


def _save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    svg = osp.join(IMG_DIR, f"{name}.svg")
    savefig_true_size_svg(fig, svg)
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  wrote {svg}")


def panel_sizes(sizes: pd.DataFrame):
    """Half-width panel, one bar per graph: covered genes, edges (log axis), and the share of a
    graph's edges found in no other graph."""
    w, h = PANEL_WIDTHS_MM["half"], 56.0
    fig, (ax_n, ax_e, ax_u) = plt.subplots(
        1, 3, figsize=(mm_to_in(w), mm_to_in(h)), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.3, 1], "wspace": 0.16},
    )
    fig.subplots_adjust(left=0.37, right=0.985, top=0.97, bottom=0.16)
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
        _box(ax)
        ax.grid(axis="x", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, width=0.5)
    _save(fig, "graphs_sizes")


def panel_degree(deg: pd.DataFrame):
    """Half-width panel: complementary CDF of undirected degree, log-log, one line per graph."""
    w, h = PANEL_WIDTHS_MM["half"], 58.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.37)
    for key, lbl, _ in CGT_GRAPHS:
        d = deg[deg["graph"] == key].sort_values("degree")
        n = d["n_nodes"].to_numpy()
        ccdf = 1 - np.cumsum(n) / n.sum() + n / n.sum()  # P(degree >= k)
        ax.plot(d["degree"], ccdf, color=GRAPH_COLOR[key], linewidth=0.9, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Fraction of genes, degree $\\geq$ k")
    ax.grid(color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)
    fig.legend(frameon=False, loc="lower center", ncol=3, handlelength=1.0, columnspacing=0.6,
               handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
    _box(ax)
    _save(fig, "graphs_degree_ccdf")


def _heatmap(matrix: np.ndarray, labels: list[str], name: str, vmax: float, fmt: str, cbar_label: str, tri: str | None):
    """Half-width square heatmap with in-cell annotations."""
    w = PANEL_WIDTHS_MM["half"]
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(w * 0.72)))
    fig.subplots_adjust(left=0.30, right=0.84, top=0.71, bottom=0.03)
    M = matrix.copy()
    if tri == "lower":
        M[np.triu_indices_from(M, k=1)] = np.nan
    im = ax.imshow(M, cmap=HEAT_CMAP, vmin=0, vmax=vmax, aspect="equal")
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if np.isnan(M[i, j]):
                continue
            v = M[i, j]
            ax.text(
                j, i, fmt.format(v), ha="center", va="center", fontsize=5,
                color="white" if v > 0.6 * vmax else "black",
            )
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    _box(ax)
    cax = fig.add_axes([0.86, 0.08, 0.025, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5)
    _save(fig, name)


def panel_overlap(pw: pd.DataFrame, sizes: pd.DataFrame):
    keys = [k for k, _, _ in CGT_GRAPHS]
    labels = [LABEL[k] for k in keys]
    n = len(keys)
    J = np.eye(n)
    C = np.eye(n)
    for _, r in pw.iterrows():
        i, j = keys.index(r["graph_a"]), keys.index(r["graph_b"])
        J[i, j] = J[j, i] = r["jaccard"]
        C[i, j] = r["contain_a_in_b"]  # row graph's edges found in column graph
        C[j, i] = r["contain_b_in_a"]
    _heatmap(J, labels, "graphs_jaccard", vmax=0.5, fmt="{:.2f}", cbar_label="Jaccard index", tri="lower")
    _heatmap(C, labels, "graphs_containment", vmax=1.0, fmt="{:.2f}",
             cbar_label="Row edges in column graph", tri=None)


def panel_multiplicity(mult: pd.DataFrame, sizes: pd.DataFrame):
    """Third-width panel: distinct gene pairs supported by exactly n of the nine graphs."""
    w, h = PANEL_WIDTHS_MM["third"], 46.0
    fig, ax = plt.subplots(figsize=(mm_to_in(w), mm_to_in(h)))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.21)
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
    ax.tick_params(length=2, width=0.5)
    _box(ax)
    _save(fig, "graphs_edge_multiplicity")


def panel_string_versions(sizes: pd.DataFrame, drift: pd.DataFrame):
    """Half-width panel: edges per STRING channel across versions (top) and the Jaccard index
    between versions of the same channel (bottom)."""
    w, h = PANEL_WIDTHS_MM["half"], 72.0
    fig, (ax_e, ax_j) = plt.subplots(2, 1, figsize=(mm_to_in(w), mm_to_in(h)), sharex=True,
                                     gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.10})
    fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.17)
    x = np.arange(len(STRING_CHANNELS))
    bw = 0.26
    for i, v in enumerate(STRING_VERSIONS):
        lab = STRING_VERSION_LABEL[v]
        e = [sizes[(sizes["channel"] == ch) & (sizes["version"] == lab)]["edges"].item() for ch in STRING_CHANNELS]
        ax_e.bar(x + (i - 1) * bw, e, bw, color=VERSION_COLOR[v], edgecolor="black", linewidth=0.4, label=lab)
    ax_e.set_yscale("log")
    ax_e.set_ylim(1e2, 1e8)
    ax_e.set_yticks([1e2, 1e4, 1e6])
    ax_e.set_ylabel("Edges")
    ax_e.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.0, columnspacing=1.0)
    pairs = [("v9.1", "v11.0"), ("v11.0", "v12.0"), ("v9.1", "v12.0")]
    pair_color = [PLOT_PALETTE[3], PLOT_PALETTE[4], PLOT_PALETTE[5]]
    for i, ((va, vb), c) in enumerate(zip(pairs, pair_color)):
        j = [drift[(drift["channel"] == ch) & (drift["version_a"] == va) & (drift["version_b"] == vb)]["jaccard"].item()
             for ch in STRING_CHANNELS]
        ax_j.bar(x + (i - 1) * bw, j, bw, color=c, edgecolor="black", linewidth=0.4,
                 label=f"{va[1:]} vs {vb[1:]}")
    ax_j.set_ylim(0, 1)
    ax_j.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax_j.yaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.2)))
    ax_j.tick_params(which="minor", length=0)
    ax_j.set_ylabel("Jaccard")
    ax_j.legend(frameon=False, ncol=3, loc="upper left", handlelength=0.8, columnspacing=0.6,
                handletextpad=0.4)
    ax_j.set_xticks(x)
    ax_j.set_xticklabels([CHANNEL_LABEL[ch] for ch in STRING_CHANNELS], rotation=30, ha="right", rotation_mode="anchor")
    for ax in (ax_e, ax_j):
        ax.grid(axis="y", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(length=2, width=0.5)
        _box(ax)
    _save(fig, "graphs_string_versions")


# ------------------------------------------------------------------------------ tables
def write_tex_tables(sizes: pd.DataFrame, mult: pd.DataFrame, sv: pd.DataFrame):
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
        "(STRING~v9.1 and v11.0, used in the DANGO replication, are in \\supptab{tab:string-versions}.)}",
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
    lines = [
        f"%% SOURCE: {src} -- AUTO-GENERATED, do not hand-edit; rerun the script.",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{STRING evidence channels for \\emph{S.\\ cerevisiae} across the three releases",
        "integrated into TorchCell, over the shared 6,607-gene vocabulary. Nodes are genes with at",
        "least one edge; edges are distinct unordered gene pairs with a nonzero channel score.",
        "DANGO was published on v9.1 with a v9.1$\\to$v11.0 comparison; the CGT trigenic",
        "experiment uses v12.0.}",
        "\\label{tab:string-versions}",
        "\\begin{tabular}{@{}l rrr rrr@{}}",
        "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{Nodes}} & \\multicolumn{3}{c}{\\textbf{Edges}}\\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
        "\\textbf{Channel} & " + " & ".join(vers) + " & " + " & ".join(vers) + "\\\\",
        "\\midrule",
    ]
    for ch in STRING_CHANNELS:
        lines.append(
            f"{CHANNEL_LABEL[ch]} & " + " & ".join(f"{int(piv_n.loc[ch, v]):,}" for v in vers)
            + " & " + " & ".join(f"{int(piv_e.loc[ch, v]):,}" for v in vers) + "\\\\"
        )
    lines += [
        "\\midrule",
        "Sum & " + " & ".join(f"{int(piv_n[v].sum()):,}" for v in vers)
        + " & " + " & ".join(f"{int(piv_e[v].sum()):,}" for v in vers) + "\\\\",
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
    print(f"gene vocabulary: {n_vocab}")

    graphs = {key: getattr(builder, f"G_{key}").graph for key, _, _ in CGT_GRAPHS}
    pairs = {key: undirected_pairs(G) for key, G in graphs.items()}
    for key, G in graphs.items():
        print(f"{key:26s} nodes={G.number_of_nodes():5d} edges={G.number_of_edges():9,d} pairs={len(pairs[key]):9,d}")

    sizes = size_table(graphs, pairs, n_vocab)
    pw = pairwise_table(pairs)
    mult = multiplicity_table(pairs)
    deg = degree_table(pairs)
    sv_sizes, sv_drift = string_version_table(builder)

    sizes.to_csv(osp.join(RESULTS_DIR, "graph_sizes.csv"), index=False)
    pw.to_csv(osp.join(RESULTS_DIR, "pairwise_overlap.csv"), index=False)
    mult.to_csv(osp.join(RESULTS_DIR, "edge_multiplicity.csv"), index=False)
    deg.to_csv(osp.join(RESULTS_DIR, "degree_distribution.csv"), index=False)
    sv_sizes.to_csv(osp.join(RESULTS_DIR, "string_version_sizes.csv"), index=False)
    sv_drift.to_csv(osp.join(RESULTS_DIR, "string_version_drift.csv"), index=False)
    print(f"wrote CSVs to {RESULTS_DIR}")
    print(sizes[["label", "nodes", "edges_native", "edges_pairs", "mean_degree", "unique_frac"]].to_string(index=False))
    print(mult.to_string(index=False))
    print(sv_drift[["channel", "version_a", "version_b", "jaccard", "retained_frac_of_a"]].to_string(index=False))

    panel_sizes(sizes)
    panel_degree(deg)
    panel_overlap(pw, sizes)
    panel_multiplicity(mult, sizes)
    panel_string_versions(sv_sizes, sv_drift)
    write_tex_tables(sizes, mult, sv_sizes)


if __name__ == "__main__":
    main()
