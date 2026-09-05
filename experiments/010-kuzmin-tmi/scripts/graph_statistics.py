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
  the three highest-degree genes of every graph, with their degree percentile in all nine
  graphs and a short SGD description each;
* the union of the nine graphs: genes with an edge in any graph, genes with an edge in
  none, and the distinct gene pairs;
* the two directed transcription-factor graphs (SGD regulatory, TFLink): shared regulators,
  targets, and directed edges, and the per-regulator agreement of target sets;
* STRING release drift: per-channel nodes/edges for v9.1, v11.0, v12.0 and, per consecutive
  release, the pairs retained, added, and dropped;
* how each graph relates to the other components of the cell representation: the share of
  its covered genes that are Yeast9 metabolic genes or SGD-essential genes, its coverage of
  the Kuzmin 2018 and 2020 trigenic gene panels, and the share of its gene pairs that share a
  Yeast9 reaction, a Yeast9 subsystem, or a GO biological-process term, each against a
  degree-preserving random graph.

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

Paths are resolved from this file, so the script runs from any working directory:
    python experiments/010-kuzmin-tmi/scripts/graph_statistics.py
"""

import json
import os
import os.path as osp
import re
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
from torchcell.metabolism.yeast_GEM import YeastGEM
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

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
RESULTS_DIR = osp.join(REPO_ROOT, "experiments/010-kuzmin-tmi/results/graphs")
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
TEX_DIR = osp.join(REPO_ROOT, "paper/nature-biotech/sections")

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
# The pale companion of each graph's color: the lighter member of a two-level bar (house
# convention: one series in the line color, the other in its fill; no hatching).
GRAPH_FILL = {key: PLOT_PALETTE_FILL[i] for i, (key, _, _) in enumerate(CGT_GRAPHS)}
LABEL = {key: lbl for key, lbl, _ in CGT_GRAPHS}
UNION_LABEL = "Union (all nine)"
# Short labels for the 9 x 9 matrix axes (third-width heatmaps); the STRING prefix is
# dropped and the captions say so.
SHORT_LABEL = {
    "physical": "Physical",
    "regulatory": "Regulatory",
    "tflink": "TFLink",
    "string12_0_neighborhood": "neighborhood",
    "string12_0_fusion": "fusion",
    "string12_0_cooccurence": "co-occur.",
    "string12_0_coexpression": "co-expr.",
    "string12_0_experimental": "experimental",
    "string12_0_database": "database",
}

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

# Hub tables: a gene counts as a hub of a graph when it is among the graph's TOP_K highest-degree
# genes (hub_genes.csv); recurrence is counted at TOP_K and at the top 1% of the vocabulary
# (66 genes per graph). The hub PANEL shows the HUB_TOP_N highest-degree genes of every graph
# (ties broken by systematic name), the union ordered by the first graph (config order) whose
# top HUB_TOP_N contains the gene, then by degree in that graph.
TOP_K = 10
TOP_FRAC = 0.01
HUB_TOP_N = 3

# Other components of the cell representation (panel f of figure 1). GO co-annotation uses
# biological_process terms, the root excluded, with at most GO_MAX_GENES directly annotated
# genes (SGD annotations of any evidence code, not propagated up the DAG). The random
# reference is a degree-preserving stub matching (configuration model) of each graph with
# self-loops and duplicate pairs dropped, seed RANDOM_SEED.
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}
GO_MAX_GENES = 500
RANDOM_SEED = 0
YEAST_GEM_VERSION = "9.0.2"
PAIR_SHARE_MIN = 1e-6  # left edge of the logarithmic pair-share axes of panel f

# Shared panel geometry (mm). Panels in one row use the same height and the same top margin
# (TOP_MM) so their axes tops align in the composed figure; heatmaps carry their column
# labels BELOW the matrix for the same reason. Figure 1 ("each graph on its own"): row 1
# sizes + degree (half + half), row 2 structure + hubs (half + half), row 3 other components
# (full). Figure 2 ("how the graphs relate"): row 1 Jaccard, containment, shared pairs
# (three thirds), row 2 multiplicity (third) + transcription-factor graphs (wide).
TOP_MM = 1.5
LABEL_LEFT_MM = 32.0  # axes left for panels with graph names on the y-axis
SHORT_LEFT_MM = 14.5  # axes left for the third-width matrices with SHORT_LABEL rows
HEAT_BOTTOM_MM = 11.5  # room under a matrix for its 45-degree column labels
F1_ROW1_H, F1_ROW2_H, F1_ROW3_H = 48.0, 60.0, 44.0
F2_ROW1_H, F2_ROW2_H = 44.0, 58.0
STRING_RELEASES_H = 52.0  # graphs_string_releases.svg, embedded by the DANGO figure; keep fixed
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "pad": 1}  # numbers over gridlines


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


def restrict_to_vocab(G: nx.Graph, vocab, key: str) -> nx.Graph:
    """The graph over the gene vocabulary only. The cached SGD regulatory graph keeps every
    feature a regulation record names: 35 regulators are SGD protein complexes (``CPX-``
    ids) and 405 targets are non-coding RNAs, LTRs, and Ty ORFs outside the 6,607
    protein-coding genes (435 nodes, 1,869 of 39,636 directed edges); the other eight
    graphs are already within the vocabulary. Every statistic is over the vocabulary, so
    those nodes and their edges are dropped here; genes left without an edge are dropped
    with them (none in practice)."""
    vocab = set(vocab)
    out = [n for n in G.nodes if n not in vocab]
    H = G.subgraph([n for n in G.nodes if n in vocab]).copy()
    isolated = [n for n in H.nodes if H.degree(n) == 0]
    H.remove_nodes_from(isolated)
    if out:
        print(
            f"{key}: dropped {len(out)} nodes outside the vocabulary "
            f"({G.number_of_edges() - H.number_of_edges():,} of {G.number_of_edges():,} edges), "
            f"{len(isolated)} genes left isolated"
        )
    return H


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
                "uncovered": n_vocab - n_nodes,
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


def union_table(graphs: dict[str, nx.Graph], pairs: dict[str, set], n_vocab: int) -> pd.DataFrame:
    """One row for the union of the nine graphs: genes with an edge in any graph, genes with
    an edge in none, and the distinct gene pairs (the sum of edge_multiplicity.csv)."""
    nodes = set().union(*(set(G.nodes) for G in graphs.values()))
    E = set().union(*pairs.values())
    return pd.DataFrame(
        [
            {
                "graph": "union",
                "label": UNION_LABEL,
                "nodes": len(nodes),
                "uncovered": n_vocab - len(nodes),
                "node_coverage": len(nodes) / n_vocab,
                "edges_pairs": len(E),
                "mean_degree": 2 * len(E) / len(nodes),
                "density_vocab": 2 * len(E) / (n_vocab * (n_vocab - 1)),
            }
        ]
    )


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
    percentile-rank matrix (gene x graph) of the HUB_TOP_N highest-degree genes of every
    graph. Rows of the matrix are ordered by the first graph, in config order, whose top
    HUB_TOP_N contains the gene, then by degree in that graph (ties by systematic name)."""
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
    top_of: dict[str, tuple[int, int, int]] = {}  # gene -> (graph index, rank, degree)
    for gi, k in enumerate(KEYS):
        s = by_degree(degrees[k])
        for r, (g, dv) in enumerate(s.iloc[:HUB_TOP_N].items(), start=1):
            if g not in top_of:
                top_of[g] = (gi, r, int(dv))
    order = sorted(top_of, key=lambda g: (top_of[g][0], -top_of[g][2], g))
    mat_rows = []
    for g in order:
        gi, r, _ = top_of[g]
        for k in KEYS:
            present = g in degrees[k].index
            mat_rows.append(
                {
                    "gene": g,
                    "name": std.get(g, g),
                    "top_of_graph": KEYS[gi],
                    "top_of_rank": r,
                    "n_graphs_top_k": int(cnt_k[g]),
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


# Phrase boundaries of an SGD description: a new segment starts at a preposition, a
# conjunction, a relative pronoun, or a participle ("involved in", "associated with").
_DESC_FUNCTION_WORDS = (
    "of", "in", "with", "by", "to", "for", "and", "or", "that", "which", "from", "at", "on",
    "involved", "required", "associated",
)
_DESC_BREAK = re.compile(r"\s+(?=(?:" + "|".join(_DESC_FUNCTION_WORDS) + r")\b)")
_DESC_STRIP = set(_DESC_FUNCTION_WORDS) | {"the", "a", "an", "is"}
_DESC_PAREN = re.compile(r"\s*\([^)]*\)")
DESC_MAX_CHARS = 50  # about the description column of the hub panel at 5 pt


def short_description(description: str) -> str:
    """The leading phrase of an SGD locus description for a figure label, deterministic
    from the description alone: the first semicolon-delimited clause with parentheticals
    removed, split at phrase boundaries (before prepositions, conjunctions, relative
    pronouns, and participles), whole segments kept while the text stays within
    DESC_MAX_CHARS. A cut that would fall inside a coordination ("X of A and B" cut after
    A) also drops the conjunct already kept, unless it is the head segment, so a label
    never misstates the source. Trailing function words are stripped."""
    clause = _DESC_PAREN.sub("", description.split(";")[0])
    clause = re.sub(r"\s+", " ", clause).strip()
    segments = _DESC_BREAK.split(clause)
    kept: list[str] = []
    for seg in segments:
        if len(" ".join(kept + [seg])) > DESC_MAX_CHARS:
            if seg.split()[0] in ("and", "or") and len(kept) > 1:
                kept.pop()
            break
        kept.append(seg)
    words = " ".join(kept).split()
    while words and words[-1].lower().strip(",") in _DESC_STRIP:
        words.pop()
    return " ".join(words).rstrip(",")


def hub_description_table(matrix: pd.DataFrame, std: dict[str, str]) -> pd.DataFrame:
    """Short SGD description per hub of the panel, read from the per-gene SGD JSON
    (``locus.description``); the full first clause and the source file are kept."""
    rows = []
    for g in dict.fromkeys(matrix["gene"]):
        rel = f"data/sgd/genome/genes/{g}.json"
        with open(osp.join(DATA_ROOT, rel)) as f:
            desc = json.load(f)["locus"]["description"]
        rows.append(
            {
                "gene": g,
                "name": std.get(g, g),
                "short_description": short_description(desc),
                "sgd_description_first_clause": desc.split(";")[0].strip(),
                "source": f"$DATA_ROOT/{rel} locus.description",
            }
        )
    return pd.DataFrame(rows)


def load_components(builder: SCerevisiaeGraph) -> dict[str, object]:
    """Gene sets and gene-pair sets of the other components of the cell representation.

    Yeast9 (yeast-GEM 9.0.2) through ``torchcell.metabolism.yeast_GEM.YeastGEM``: metabolic
    genes are the genes of the model's gene--reaction rules; reaction pairs are gene pairs
    that share at least one reaction; subsystem pairs are gene pairs whose genes appear in
    reactions of the same subsystem. GO pairs are gene pairs directly co-annotated to a
    biological_process term (root excluded, at most GO_MAX_GENES genes). Essential genes
    follow ``GeneEssentialitySgdDataset``: an SGD null-mutant phenotype ``inviable`` in
    S288C. The trigenic panels are the perturbed genes of the Kuzmin 2018 build used by
    experiment 010 and the gene set of the Kuzmin 2020 trigenic dataset.
    """
    vocab = set(builder.genome.gene_set)
    gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast-GEM"), version=YEAST_GEM_VERSION)
    model = gem.model
    metabolic = {g.id for g in model.genes} & vocab
    rxn_pairs: set[tuple[str, str]] = set()
    sub_genes: dict[str, set[str]] = {}
    for r in model.reactions:
        genes = sorted({g.id for g in r.genes} & vocab)
        rxn_pairs.update(combinations(genes, 2))
        if r.subsystem:
            sub_genes.setdefault(r.subsystem, set()).update(genes)
    sub_pairs: set[tuple[str, str]] = set()
    for genes in sub_genes.values():
        sub_pairs.update(combinations(sorted(genes), 2))

    dag = builder.genome.go_dag
    go_pairs: set[tuple[str, str]] = set()
    n_go_terms = 0
    for go_id, genes in builder.go_to_genes.items():
        if dag[go_id].namespace != "biological_process" or go_id in GO_ROOTS:
            continue
        genes = sorted(set(genes) & vocab)
        if len(genes) < 2 or len(genes) > GO_MAX_GENES:
            continue
        n_go_terms += 1
        go_pairs.update(combinations(genes, 2))

    essential = set()
    for g in vocab:
        for ph in builder.G_raw.nodes[g].get("phenotype_details", []):
            if (
                ph["mutant_type"] == "null"
                and ph["strain"]["display_name"] == "S288C"
                and ph["phenotype"]["display_name"] == "inviable"
            ):
                essential.add(g)
                break

    k18_path = osp.join(
        DATA_ROOT,
        "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/processed/is_any_perturbed_gene_index.json",
    )
    with open(k18_path) as f:
        kuzmin2018 = set(json.load(f).keys()) & vocab
    with open(osp.join(DATA_ROOT, "data/torchcell/tmi_kuzmin2020/preprocess/gene_set.json")) as f:
        kuzmin2020 = set(json.load(f)) & vocab
    print(
        f"components: metabolic {len(metabolic)} genes, reaction pairs {len(rxn_pairs):,}, "
        f"{len(sub_genes)} subsystems ({len(sub_pairs):,} pairs), GO BP terms {n_go_terms} "
        f"({len(go_pairs):,} pairs), essential {len(essential)}, Kuzmin 2018 {len(kuzmin2018)}, "
        f"Kuzmin 2020 {len(kuzmin2020)}"
    )
    return {
        "n_vocab": len(vocab),
        "metabolic": metabolic,
        "essential": essential,
        "kuzmin2018": kuzmin2018,
        "kuzmin2020": kuzmin2020,
        "reaction_pairs": rxn_pairs,
        "subsystem_pairs": sub_pairs,
        "go_bp_pairs": go_pairs,
        "n_subsystems": len(sub_genes),
        "n_go_terms": n_go_terms,
    }


def random_pairs(pairs: set[tuple[str, str]], seed: int) -> set[tuple[str, str]]:
    """Degree-preserving random pairs: stub matching on the degree sequence of ``pairs``
    (configuration model), self-loops and duplicate pairs dropped."""
    deg: dict[str, int] = {}
    for u, v in pairs:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    nodes = sorted(deg)
    stubs = np.repeat(np.arange(len(nodes)), [deg[g] for g in nodes])
    rng = np.random.default_rng(seed)
    stubs = rng.permutation(stubs).reshape(-1, 2)
    out = set()
    for a, b in stubs:
        if a != b:
            out.add((nodes[min(a, b)], nodes[max(a, b)]))
    return out


def components_table(graphs: dict[str, nx.Graph], pairs: dict[str, set], comp: dict) -> pd.DataFrame:
    """Per graph: gene-level shares (metabolic, essential, trigenic-panel coverage) and
    pair-level shares (reaction, subsystem, GO process co-membership) with the
    degree-preserving random reference for the pair-level shares."""
    pair_sets = [("reaction", comp["reaction_pairs"]), ("subsystem", comp["subsystem_pairs"]), ("go_bp", comp["go_bp_pairs"])]
    rows = []
    for key in KEYS:
        covered = set(graphs[key].nodes)
        E = pairs[key]
        R = random_pairs(E, RANDOM_SEED)
        row = {
            "graph": key,
            "label": LABEL[key],
            "nodes": len(covered),
            "edges_pairs": len(E),
            "metabolic_genes": len(covered & comp["metabolic"]),
            "metabolic_frac": len(covered & comp["metabolic"]) / len(covered),
            "essential_genes": len(covered & comp["essential"]),
            "essential_frac": len(covered & comp["essential"]) / len(covered),
            "kuzmin2018_covered": len(covered & comp["kuzmin2018"]),
            "kuzmin2018_covered_frac": len(covered & comp["kuzmin2018"]) / len(comp["kuzmin2018"]),
            "kuzmin2020_covered": len(covered & comp["kuzmin2020"]),
            "kuzmin2020_covered_frac": len(covered & comp["kuzmin2020"]) / len(comp["kuzmin2020"]),
            "random_pairs": len(R),
        }
        for name, P in pair_sets:
            row[f"share_{name}_pairs"] = len(E & P)
            row[f"share_{name}_frac"] = len(E & P) / len(E)
            row[f"share_{name}_frac_random"] = len(R & P) / len(R)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["reference"] = {
        "n_vocab": comp["n_vocab"],
        "metabolic_frac_vocab": len(comp["metabolic"]) / comp["n_vocab"],
        "essential_frac_vocab": len(comp["essential"]) / comp["n_vocab"],
        "n_kuzmin2018": len(comp["kuzmin2018"]),
        "n_kuzmin2020": len(comp["kuzmin2020"]),
        "n_subsystems": comp["n_subsystems"],
        "n_go_terms": comp["n_go_terms"],
    }
    return df


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


def panel_sizes(sizes: pd.DataFrame, union: pd.DataFrame, n_vocab: int):
    """Half-width panel, one bar per graph plus a final row for the union of all nine:
    genes with at least one edge (dotted line, the 6,607-gene reference), genes with no
    edge (its own bar, the count printed at right), distinct gene pairs (log axis), and the
    share of a graph's pairs found in no other graph (empty for the union)."""
    w, h = PANEL_WIDTHS_MM["half"], F1_ROW1_H
    fig = _fig(w, h)
    bottom = 11.0  # two-line x labels
    ax_n, ax_u, ax_e, ax_f = _named_barh_row(fig, w, h, 4, [1.1, 1.15, 1.35, 0.75], bottom_mm=bottom)
    n = len(sizes)
    # Graph rows at 9.5 .. 1.5, the union row at 0 with a wider gap above it.
    y = np.concatenate([np.arange(n)[::-1] + 1.5, [0.0]])
    nodes = np.concatenate([sizes["nodes"], union["nodes"]])
    uncovered = np.concatenate([sizes["uncovered"], union["uncovered"]])
    edges = np.concatenate([sizes["edges_pairs"], union["edges_pairs"]])
    labels = list(sizes["label"]) + list(union["label"])
    colors = [GRAPH_COLOR[k] for k in sizes["graph"]] + ["white"]
    bar = {"edgecolor": "black", "linewidth": 0.4, "height": 0.7}
    ax_n.barh(y, nodes, color=colors, **bar)
    ax_n.axvline(n_vocab, color="black", linewidth=0.5, linestyle=":")
    ax_n.set_xlim(0, 7400)
    ax_n.set_xticks([0, 3000, 6000])
    ax_n.set_xticklabels(["0", "3k", "6k"])
    ax_n.set_xlabel("Genes with\nan edge")
    ax_n.set_yticks(y)
    ax_n.set_yticklabels(labels)
    ax_u.barh(y, uncovered, color=colors, **bar)
    for yi, u in zip(y, uncovered):
        ax_u.text(u + 180, yi, f"{u:,}", ha="left", va="center", fontsize=5, bbox=TEXT_BBOX)
    ax_u.set_xlim(0, 7800)
    ax_u.set_xticks([0, 2000, 4000])
    ax_u.set_xticklabels(["0", "2k", "4k"])
    ax_u.set_xlabel("Genes with\nno edge")
    ax_e.barh(y, edges, color=colors, **bar)
    ax_e.set_xscale("log")
    ax_e.set_xlim(1e3, 3e6)
    ax_e.set_xticks([1e4, 1e5, 1e6])  # the smallest graph has 11,085 pairs; 10^3 stays a minor tick
    ax_e.set_xticklabels(["$10^4$", "$10^5$", "$10^6$"])
    ax_e.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=20))
    ax_e.xaxis.set_minor_formatter(NullFormatter())
    ax_e.set_xlabel("Gene pairs")
    ax_f.barh(y[:n], sizes["unique_frac"], color=colors[:n], **bar)
    ax_f.set_xlim(0, 1)
    ax_f.set_xticks([0, 0.5, 1.0])
    ax_f.set_xticklabels(["0", "0.5", "1"])
    ax_f.xaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.1)))
    ax_f.set_xlabel("Unique\nfraction")
    for ax in (ax_n, ax_u, ax_e, ax_f):
        ax.tick_params(which="minor", length=0)
        ax.set_ylim(-0.6, n + 1.1)
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
    w, h = PANEL_WIDTHS_MM["half"], F1_ROW1_H
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
             tri: str | None, norm=None, vmax: float | None = None, dark_above: float = 0.6,
             width: str = "third", h: float = F2_ROW1_H):
    """Square heatmap with in-cell annotations, explicit geometry: matrix top at TOP_MM so
    the panel aligns with its row, rows labeled with the short graph names at
    SHORT_LEFT_MM, columns labeled BELOW the matrix at 45 degrees (anchored right),
    colorbar on the right. ``width`` is a PANEL_WIDTHS_MM key; the matrix side follows
    the height."""
    w = PANEL_WIDTHS_MM[width]
    fig = _fig(w, h)
    bottom = HEAT_BOTTOM_MM
    side = h - TOP_MM - bottom
    ax = _axes_mm(fig, w, h, SHORT_LEFT_MM, bottom, side, side)
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
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    _box(ax)
    cax = _axes_mm(fig, w, h, SHORT_LEFT_MM + side + 1.8, bottom + 0.1 * side, 2.0, 0.8 * side)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, labelpad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5, pad=1)
    _save(fig, name)


def _frac_label(v: float) -> str:
    """'1' on the diagonal, otherwise two decimals without the leading zero (fits a
    third-width cell at 5 pt)."""
    if v >= 0.995:
        return "1"
    return f"{v:.2f}"[1:]


def _thousands_label(v_k: float) -> str:
    """A count given in thousands, at most three characters: '556', '2.1', '.9', '.07'
    (fits a third-width cell at 5 pt)."""
    if v_k >= 10:
        return f"{v_k:.0f}"
    if v_k >= 1:
        return f"{v_k:.1f}"
    if v_k >= 0.1:
        return f"{v_k:.1f}"[1:]
    return f"{v_k:.2f}"[1:]


def panel_overlap(pw: pd.DataFrame, sizes: pd.DataFrame):
    labels = [SHORT_LABEL[k] for k in KEYS]
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
    _heatmap(J, labels, "graphs_jaccard", fmt=_frac_label, vmax=1.0,
             cbar_label="Jaccard index", tri="lower")
    _heatmap(C, labels, "graphs_containment", fmt=_frac_label, vmax=1.0,
             cbar_label="Row edges in column graph", tri=None)
    # Shared pairs in thousands so every cell label has at most three characters; the
    # smallest counts (23 to 74 pairs) sit below the color floor and print as '.02' to '.07'.
    _heatmap(S / 1e3, labels, "graphs_shared_pairs", fmt=_thousands_label,
             norm=LogNorm(vmin=0.1, vmax=1e3), cbar_label="Shared gene pairs (thousands)",
             tri="lower", dark_above=0.7)


def panel_multiplicity(mult: pd.DataFrame, sizes: pd.DataFrame):
    """Third-width panel: distinct gene pairs supported by exactly n of the nine graphs; the
    x-axis stops at the largest observed multiplicity."""
    w, h = PANEL_WIDTHS_MM["third"], F2_ROW2_H
    fig = _fig(w, h)
    bottom = 8.5
    ax = _axes_mm(fig, w, h, 11.0, bottom, w - 11.0 - 1.5, h - TOP_MM - bottom)
    ax.bar(mult["n_graphs"], mult["n_pairs"], color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, width=0.7)
    for x, y in zip(mult["n_graphs"], mult["n_pairs"]):
        ax.text(x, y * 1.3, f"{y:,}", ha="center", va="bottom", fontsize=5, rotation=90, bbox=TEXT_BBOX)
    ax.set_yscale("log")
    ax.set_ylim(1, 3e7)
    n_max = int(mult["n_graphs"].max())
    ax.set_xticks(range(1, n_max + 1))
    ax.set_xlim(0.4, n_max + 0.6)
    ax.set_xlabel("Supporting graphs")
    ax.set_ylabel("Gene pairs")
    ax.grid(axis="y", color="#CACACA", linewidth=0.4)
    ax.set_axisbelow(True)
    _ticks(ax)
    _box(ax)
    _save(fig, "graphs_edge_multiplicity")


def panel_structure(struct: pd.DataFrame, n_vocab: int):
    """Half-width panel, one bar per graph: largest-component fraction, mean clustering
    coefficient, degree assortativity, and mean two-hop reach."""
    w, h = PANEL_WIDTHS_MM["half"], F1_ROW2_H
    fig = _fig(w, h)
    axes = _named_barh_row(fig, w, h, 4, [1, 1, 1.2, 1], bottom_mm=11.0)
    y = np.arange(len(struct))[::-1]
    colors = [GRAPH_COLOR[k] for k in struct["graph"]]
    cols = [
        ("lcc_frac", "Largest\ncomponent", (0, 1.0), [0, 0.5, 1.0], ["0", "0.5", "1"]),
        ("mean_clustering", "Mean\nclustering", (0, 1.0), [0, 0.5, 1.0], ["0", "0.5", "1"]),
        ("degree_assortativity", "Degree\nassortativity", (-0.6, 1.0), [-0.5, 0, 0.5, 1.0], ["−0.5", "0", "0.5", "1"]),
        ("mean_two_hop_reach", "Two-hop\nreach (genes)", (0, 7000), [0, 3000, 6000], ["0", "3k", "6k"]),
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
            ax.axvline(n_vocab, color="black", linewidth=0.5, linestyle=":")
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


def panel_hubs(matrix: pd.DataFrame, desc: pd.DataFrame):
    """Half-width heatmap: the HUB_TOP_N highest-degree genes of every graph (rows, grouped
    by the graph they top, a thicker white rule between groups) by graph (columns), colored
    by the gene's degree percentile in that graph (a dash marks a gene absent from the
    graph); the short SGD description of each hub is printed right of the matrix, the
    column labels sit below it, and the colorbar lies under the description column."""
    w, h = PANEL_WIDTHS_MM["half"], F1_ROW2_H
    genes = list(dict.fromkeys(matrix["gene"]))
    first = matrix.drop_duplicates("gene").set_index("gene")
    names = [first.loc[g, "name"] for g in genes]
    groups = [first.loc[g, "top_of_graph"] for g in genes]
    short = desc.set_index("gene")["short_description"]
    P = matrix.pivot(index="gene", columns="graph", values="percentile").loc[genes, KEYS].to_numpy()
    fig = _fig(w, h)
    bottom = 10.5  # 45-degree column labels
    left, cell_w = 11.0, 3.0
    mat_w = cell_w * len(KEYS)
    ax = _axes_mm(fig, w, h, left, bottom, mat_w, h - TOP_MM - bottom)
    im = ax.imshow(P, cmap=HEAT_CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(len(genes)):
        for j in range(len(KEYS)):
            if np.isnan(P[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=5, color="#888888")
        ax.text(len(KEYS) - 0.5 + 0.3, i, short[genes[i]], ha="left", va="center", fontsize=5,
                clip_on=False)
        if i > 0 and groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color="white", linewidth=1.6)
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels([SHORT_LABEL[k] for k in KEYS], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(names)
    ax.tick_params(length=0, pad=1.5)
    ax.set_xticks(np.arange(-0.5, len(KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    _box(ax)
    cb_left = left + mat_w + 6.0
    cax = _axes_mm(fig, w, h, cb_left, 3.0, 26.0, 1.8)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 50, 100])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=2, width=0.5, pad=1)
    fig.text(cb_left / w, 5.4 / h, "Degree percentile in graph", ha="left", va="bottom")
    _save(fig, "graphs_hubs")


def panel_tf_overlap(overlap: pd.DataFrame, per_tf: pd.DataFrame):
    """Wide panel. Top: regulators, targets, and directed edges split into regulatory-only
    / both / TFLink-only, bars 2.9 mm thick so the in-bar counts have clearance. Bottom:
    per shared regulator, Jaccard of its two target sets."""
    w, h = PANEL_WIDTHS_MM["wide"], F2_ROW2_H
    fig = _fig(w, h)
    seg_color = [PLOT_PALETTE[1], PLOT_PALETTE[5], PLOT_PALETTE[2]]
    seg_label = ["Regulatory (SGD) only", "Both", "TFLink only"]
    rows = [("regulators", "Regulators"), ("targets", "Targets"), ("directed_edges", "Directed edges")]
    left, right = 22.0, 1.5
    bar_h, bar_gap = 8.0, 1.2  # axes height per bar (mm); the bar fills 0.8 / 2.2 of it
    bar_len = w - left - right
    label_min_mm = 6.5  # a segment narrower than this is labeled outside the bar
    top_block = TOP_MM + 4.0  # legend row
    for i, (ent, lbl) in enumerate(rows):
        r = overlap[overlap["entity"] == ent].iloc[0]
        y0 = h - top_block - (i + 1) * bar_h - i * bar_gap
        ax = _axes_mm(fig, w, h, left, y0, bar_len, bar_h)
        total = r["union"]
        x = 0.0
        prev_small = False
        # Bars span y in [-0.4, 0.4]; outside labels sit at |y| = 0.6, clear of the edge.
        for val, c, lab in zip([r["regulatory_only"], r["both"], r["tflink_only"]], seg_color, seg_label):
            ax.barh(0, val, left=x, color=c, edgecolor="black", linewidth=0.4, height=0.8,
                    label=lab if i == 0 else None)
            txt = f"{int(val):,}"
            if val / total * bar_len >= label_min_mm:
                ax.text(x + val / 2, 0, txt, ha="center", va="center", fontsize=5, color="white")
                prev_small = False
            elif prev_small:  # two narrow segments in a row: label the second below the bar
                ax.text(x + val / 2, -0.6, txt, ha="center", va="top", fontsize=5, bbox=TEXT_BBOX)
                prev_small = False
            else:
                ax.text(x + val / 2, 0.6, txt, ha="center", va="bottom", fontsize=5, bbox=TEXT_BBOX)
                prev_small = True
            x += val
        ax.set_xlim(0, total)
        ax.set_ylim(-1.1, 1.1)
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
    hist_top = h - top_block - 3 * bar_h - 2 * bar_gap - 4.0
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
            ha="center", va="top", fontsize=6, bbox=TEXT_BBOX)
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
    pale part is new; the hatched bar below zero is what the previous release lost. This
    panel is embedded read-only in the DANGO reproduction figure (compose_dango_si_figures.py),
    so its size is fixed by STRING_RELEASES_H and must stay stable."""
    w, h = PANEL_WIDTHS_MM["half"], STRING_RELEASES_H
    fig = _fig(w, h)
    ncol, nrow = 3, 2
    left, right, top, bottom = 11.0, 1.5, 6.5, 7.5
    wgap, hgap = 7.5, 7.0
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


def panel_components(comp: pd.DataFrame):
    """Full-width panel, one bar per graph and six axes: the share of covered genes that are
    Yeast9 metabolic or SGD-essential genes (dotted line, the same share over the 6,607-gene
    reference), the share of the Kuzmin 2018 (graph color) and 2020 (its pale fill) trigenic
    panel genes that have an edge in the graph, and the share of the graph's gene pairs that
    share a Yeast9 reaction, a Yeast9 subsystem, or a GO biological-process term (open
    circle, the degree-preserving random graph). The legend is one line under the panel."""
    ref = comp.attrs["reference"]
    w, h = PANEL_WIDTHS_MM["full"], F1_ROW3_H
    fig = _fig(w, h)
    bottom = 13.5  # two-line x labels + one legend row
    axes = _named_barh_row(fig, w, h, 6, [1] * 6, bottom_mm=bottom)
    y = np.arange(len(comp))[::-1]
    colors = [GRAPH_COLOR[k] for k in comp["graph"]]
    fills = [GRAPH_FILL[k] for k in comp["graph"]]
    gene_cols = [
        ("metabolic_frac", "Metabolic genes\n(Yeast9)", ref["metabolic_frac_vocab"]),
        ("essential_frac", "Essential genes\n(SGD)", ref["essential_frac_vocab"]),
    ]
    pair_cols = [
        ("share_reaction_frac", "Pairs sharing\na reaction"),
        ("share_subsystem_frac", "Pairs sharing\na subsystem"),
        ("share_go_bp_frac", "Pairs sharing\na GO process"),
    ]
    for ax, (col, xlabel, base) in zip(axes[:2], gene_cols):
        ax.barh(y, comp[col], color=colors, edgecolor="black", linewidth=0.4, height=0.7)
        ax.axvline(base, color="black", linewidth=0.5, linestyle=":")
        ax.set_xlabel(xlabel)
    ax = axes[2]
    ax.barh(y + 0.19, comp["kuzmin2018_covered_frac"], color=colors, edgecolor="black", linewidth=0.4,
            height=0.36)
    ax.barh(y - 0.19, comp["kuzmin2020_covered_frac"], color=fills, edgecolor="black", linewidth=0.4,
            height=0.36)
    ax.set_xlabel("Panel genes\nwith an edge")
    # The pair-level shares span four orders of magnitude across graphs (reaction sharing
    # 5e-6 in TFLink to 0.04 in co-occurrence), so these three axes are logarithmic.
    for ax, (col, xlabel) in zip(axes[3:], pair_cols):
        ax.barh(y, comp[col], left=PAIR_SHARE_MIN, color=colors, edgecolor="black", linewidth=0.4, height=0.7)
        ax.plot(comp[f"{col}_random"], y, linestyle="none", marker="o", markersize=2.6,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        ax.set_xlabel(xlabel)
        ax.set_xscale("log")
        ax.set_xlim(PAIR_SHARE_MIN, 1)
        # Interior decades are labeled; the edge decades (1e-6, 1) are unlabeled so the
        # labels of neighboring axes do not run into each other.
        ax.set_xticks([1e-5, 1e-3, 1e-1])
        ax.set_xticklabels(["$10^{-5}$", "$10^{-3}$", "$10^{-1}$"])
        ax.xaxis.set_minor_locator(FixedLocator([1e-6, 1e-4, 1e-2, 1]))
        ax.xaxis.set_minor_formatter(NullFormatter())
    for ax in axes[:3]:
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(0.1, 1.0, 0.1)))
    for ax in axes:
        ax.tick_params(which="minor", length=0)
        ax.set_ylim(-0.6, len(comp) - 0.4)
        ax.set_yticks(y)
        _box(ax)
        ax.grid(axis="x", which="both", color="#CACACA", linewidth=0.4)
        ax.set_axisbelow(True)
        _ticks(ax)
    axes[0].set_yticklabels(comp["label"])
    for ax in axes[1:]:
        ax.set_yticklabels([])
    # Two-level legend keys in neutral gray (the classical-ML convention): the darker key
    # is the series drawn in each graph's line color, the paler key its fill.
    handles = [
        plt.Line2D([], [], color="black", linewidth=0.5, linestyle=":",
                   label=f"Share over the {ref['n_vocab']:,}-gene reference"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#7F7F7F", edgecolor="black", linewidth=0.4,
                      label=f"Kuzmin 2018 panel ({ref['n_kuzmin2018']:,} genes)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#E3E3E3", edgecolor="black", linewidth=0.4,
                      label=f"Kuzmin 2020 panel ({ref['n_kuzmin2020']:,} genes)"),
        plt.Line2D([], [], linestyle="none", marker="o", markersize=2.6, markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=0.5,
                   label=f"Degree-preserving random graph (seed {RANDOM_SEED})"),
    ]
    fig.legend(handles=handles, frameon=False, loc="lower left", ncol=4, handlelength=1.4,
               columnspacing=1.2, handletextpad=0.5, bbox_to_anchor=(0.0, 0.0), borderaxespad=0.2)
    _save(fig, "graphs_components")


# ------------------------------------------------------------------------------ tables
def write_tex_tables(sizes: pd.DataFrame, union_row: pd.DataFrame, mult: pd.DataFrame,
                     sv: pd.DataFrame, drift: pd.DataFrame, n_vocab: int):
    src = "experiments/010-kuzmin-tmi/scripts/graph_statistics.py"
    union = int(mult["n_pairs"].sum())
    assert union == int(union_row["edges_pairs"].iloc[0])
    u_nodes, u_uncov = int(union_row["nodes"].iloc[0]), int(union_row["uncovered"].iloc[0])
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
        f"{union:,} distinct gene pairs and {u_nodes:,} of the {n_vocab:,} genes ({u_uncov} genes have",
        "an edge in no graph); relative to the SGD-native baseline (Physical\\,$+$\\,Regulatory),",
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
        f"{UNION_LABEL} & {u_nodes:,} & & {union:,} & & \\\\",
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

    graphs = {key: restrict_to_vocab(getattr(builder, f"G_{key}").graph, builder.genome.gene_set, key) for key in KEYS}
    pairs = {key: undirected_pairs(G) for key, G in graphs.items()}
    for key, G in graphs.items():
        print(f"{key:26s} nodes={G.number_of_nodes():5d} edges={G.number_of_edges():9,d} pairs={len(pairs[key]):9,d}")

    sizes = size_table(graphs, pairs, n_vocab)
    union = union_table(graphs, pairs, n_vocab)
    pw = pairwise_table(pairs)
    mult = multiplicity_table(pairs)
    deg = degree_table(pairs)
    struct, degrees = structure_table(pairs, n_vocab)
    hubs, recurrence, hub_matrix = hub_tables(degrees, std, n_vocab)
    hub_desc = hub_description_table(hub_matrix, std)
    tf_overlap, per_tf = tf_overlap_tables(graphs, std)
    sv_sizes, sv_drift = string_version_table(builder)
    comp = components_table(graphs, pairs, load_components(builder))

    sizes.to_csv(osp.join(RESULTS_DIR, "graph_sizes.csv"), index=False)
    union.to_csv(osp.join(RESULTS_DIR, "graph_union.csv"), index=False)
    comp.to_csv(osp.join(RESULTS_DIR, "graph_components.csv"), index=False, float_format="%.10g")
    with open(osp.join(RESULTS_DIR, "graph_components_reference.json"), "w") as f:
        json.dump(comp.attrs["reference"], f, indent=2)
        f.write("\n")
    hub_desc.to_csv(osp.join(RESULTS_DIR, "hub_descriptions.csv"), index=False)
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
    print(union.to_string(index=False))
    print(mult.to_string(index=False))
    print(struct[["label", "lcc_frac", "mean_clustering", "degree_assortativity", "mean_two_hop_reach"]].to_string(index=False))
    print(recurrence.to_string(index=False))
    print(hubs[hubs["rank"] <= 3].to_string(index=False))
    print(tf_overlap.to_string(index=False))
    print(per_tf.head(8).to_string(index=False))
    print(sv_drift[["channel", "version_a", "version_b", "shared", "added", "dropped", "retained_frac_of_a"]].to_string(index=False))
    print(comp[["label", "metabolic_frac", "essential_frac", "kuzmin2018_covered_frac", "kuzmin2020_covered_frac",
                "share_reaction_frac", "share_reaction_frac_random", "share_subsystem_frac",
                "share_subsystem_frac_random", "share_go_bp_frac", "share_go_bp_frac_random"]].to_string(index=False))
    print(comp.attrs["reference"])
    print(hub_desc[["name", "short_description"]].to_string(index=False))

    panel_sizes(sizes, union, n_vocab)
    panel_degree(deg)
    panel_overlap(pw, sizes)
    panel_multiplicity(mult, sizes)
    panel_components(comp)
    panel_structure(struct, n_vocab)
    panel_hubs(hub_matrix, hub_desc)
    panel_tf_overlap(tf_overlap, per_tf)
    panel_string_releases(sv_sizes, sv_drift)
    write_tex_tables(sizes, union, mult, sv_sizes, sv_drift, n_vocab)


if __name__ == "__main__":
    main()
