# experiments/008-xue-ffa/scripts/perturbation_reach_analysis.py
# [[experiments.008-xue-ffa.scripts.perturbation_reach_analysis]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/perturbation_reach_analysis
"""HOW FAR DOES A PERTURBATION REACH, AND HOW MUCH OF THAT REACH IS UNMEASURED?

THE QUESTION. The document [[008-xue-ffa-epistasis]] reports trigenic interaction among
ten transcription-factor deletions read out on ONE metabolic locus, free fatty acid titer.
A flux model already showed that distance from the readout does not separate a regulator
perturbation from an enzyme perturbation. The next candidate is REACH: a transcription
factor deletion touches many genes, each of those genes touches reactions, and most of
those reactions are nowhere near the one locus the titer measures.

WHAT THIS SCRIPT IS. A descriptive count on annotation graphs. It measures, for three
kinds of perturbation seed, how many genes and how many Yeast9 reactions sit within one,
two and three hops, and what fraction of those reactions is OFF the measured locus. It
does NOT measure titer, flux, expression, or interaction. Nothing here is evidence that a
large reach produces an interaction score; a count of annotated neighbors is a property of
the annotation graphs, and the graphs are incomplete and unevenly curated. Any mechanism
connecting reach to the nonlinearity at the readout is a HYPOTHESIS that this script does
not test.

THE THREE SEED CLASSES.
  regulator        the ten deleted transcription factors of the source study. None of them
                   is a Yeast9 gene, so each acts on the model only through its targets.
  pathway enzyme   the thirteen core fatty acid pathway genes, a direct hit on the locus
                   the titer measures.
  adjacent enzyme  Yeast9 genes that are NOT pathway genes and whose reactions share at
                   least one non-currency metabolite with a reaction catalyzed by a
                   pathway gene. One reaction hop from the pathway. These are the genes a
                   "block the competing flux and divert it toward the pathway" strategy
                   deletes, so the class is the engineering-relevant comparison arm. Its
                   size is computed here, not assumed.

THE TEN GRAPHS, plus their union.
  Nine CGT gene graphs from SCerevisiaeGraph: physical, regulatory, tflink, and the six
  STRING 12.0 channels (neighborhood, fusion, cooccurence, coexpression, experimental,
  database). regulatory and tflink are DIRECTED and only out-edges are followed, which is
  the factor to target direction. physical and the six STRING channels are undirected.
  The tenth graph is built from Yeast9 itself: gene to the reactions it catalyzes under
  the model's gene-reaction rules, to the non-currency metabolites of those reactions, to
  the other reactions carrying those metabolites, to their genes. One gene hop on that
  graph is gene to reaction to metabolite to reaction to gene.
  The union carries an edge whenever any of the ten carries it.

THE READOUT LOCUS. The Yeast9 reactions catalyzed by the thirteen pathway genes, plus the
five fatty acid exchange reactions the regulator versus enzyme script uses as its readout.
A reaction outside that set is a place the titer does not look.

OUTPUT. reach.csv (one row per seed, graph and hop depth), classes.csv and summary.json
under results/perturbation_reach/, plus a four-panel figure.
"""

import json
import os
import os.path as osp
import sys
from collections import defaultdict

import cobra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import mannwhitneyu

from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import (
    PANEL_WIDTHS_MM,
    PLOT_PALETTE,
    apply_paper_style,
    mm_to_in,
    savefig_true_size_svg,
)

# The constants below are the ones the flux model already uses, imported rather than
# retyped so the two analyses cannot drift apart. This file lives beside that one, so the
# script directory is on sys.path when the script is run directly; the insert makes the
# import work when the module is imported from elsewhere too.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from regulator_vs_enzyme_epistasis_model import (  # noqa: E402
    CURRENCY,
    MODEL_PATH,
    PATHWAY_GENES,
    READOUT_EXCHANGES,
    REGULATOR_ARM,
    sha256_of_file,
)

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "008-xue-ffa/results/perturbation_reach")
IMAGES_DIR = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
FIG_STEM = "perturbation_reach"

# name in SCerevisiaeGraph -> (attribute, directed, short label for the figure axis)
CGT_GRAPHS = [
    ("physical", "G_physical", False, "physical"),
    ("regulatory", "G_regulatory", True, "regulatory"),
    ("tflink", "G_tflink", True, "tflink"),
    ("string12_0_neighborhood", "G_string12_0_neighborhood", False, "neighborhood"),
    ("string12_0_fusion", "G_string12_0_fusion", False, "fusion"),
    ("string12_0_cooccurence", "G_string12_0_cooccurence", False, "cooccurence"),
    ("string12_0_coexpression", "G_string12_0_coexpression", False, "coexpression"),
    ("string12_0_experimental", "G_string12_0_experimental", False, "experimental"),
    ("string12_0_database", "G_string12_0_database", False, "database"),
]
METABOLIC_GRAPH = "yeast9_metabolic"
UNION_GRAPH = "union"
GRAPH_ORDER = [name for name, _, _, _ in CGT_GRAPHS] + [METABOLIC_GRAPH, UNION_GRAPH]
GRAPH_LABEL = {name: label for name, _, _, label in CGT_GRAPHS}
GRAPH_LABEL[METABOLIC_GRAPH] = "metabolic"
GRAPH_LABEL[UNION_GRAPH] = "union"

DEPTHS = (1, 2, 3)
CLASSES = ("regulator", "pathway enzyme", "adjacent enzyme")
CLASS_COLOR = {
    "regulator": PLOT_PALETTE[0],
    "pathway enzyme": PLOT_PALETTE[1],
    "adjacent enzyme": PLOT_PALETTE[2],
}

CAVEATS = [
    "Every number here is a count on annotation graphs. None of it is a titer, a flux, "
    "or an interaction score.",
    "The graphs are incomplete and unevenly curated. A transcription factor with a long "
    "TFLink target list and one with a short list differ in how much the resource has "
    "recorded as well as in what the cell does.",
    "Reach is not effect. A gene one hop away on a coexpression graph need not change "
    "when the seed is deleted, and the size of any change is not modeled.",
    "Edge direction is used only where the graph carries it. The regulatory and TFLink "
    "edges are followed factor to target; activation versus repression is ignored.",
    "The metabolic graph is structural. A reaction with zero capacity in the study's "
    "chassis still carries an edge here.",
    "The adjacent enzyme class is defined by shared non-currency metabolites, so it "
    "depends on which metabolites the currency list drops.",
]


def bfs_levels(
    adjacency: dict[str, frozenset[str]], seed: str, max_depth: int
) -> list[frozenset[str]]:
    """Genes within 1, 2, ... max_depth hops of the seed, seed excluded.

    Each returned set is cumulative: entry k holds every gene at distance <= k. A seed
    that is absent from the graph returns empty sets, which is the honest answer rather
    than a missing row.
    """
    empty: frozenset[str] = frozenset()
    visited = {seed}
    frontier = {seed}
    levels: list[frozenset[str]] = []
    for _ in range(max_depth):
        nxt: set[str] = set()
        for node in frontier:
            nxt |= adjacency.get(node, empty)
        nxt -= visited
        visited |= nxt
        frontier = nxt
        levels.append(frozenset(visited - {seed}))
    return levels


def build_cgt_adjacency(graph, directed: bool) -> dict[str, frozenset[str]]:
    """Out-neighbors of every node, as frozensets keyed by ORF."""
    if directed != graph.is_directed():
        raise ValueError(
            f"graph directedness {graph.is_directed()} does not match the declared "
            f"{directed}; direction is load bearing here"
        )
    if directed:
        return {n: frozenset(graph.successors(n)) - {n} for n in graph.nodes}
    return {n: frozenset(graph.neighbors(n)) - {n} for n in graph.nodes}


def build_metabolic_adjacency(
    model: cobra.Model,
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]], int]:
    """Gene to gene via reaction, non-currency metabolite, reaction.

    Returns the adjacency, the gene to reaction map, and the number of gene-metabolite
    incidences dropped because the metabolite is a currency metabolite.
    """
    genes_of_metabolite: dict[str, set[str]] = defaultdict(set)
    metabolites_of_gene: dict[str, set[str]] = defaultdict(set)
    reactions_of_gene: dict[str, set[str]] = defaultdict(set)
    dropped = 0
    for rxn in model.reactions:
        gene_ids = {g.id for g in rxn.genes}
        for gid in gene_ids:
            reactions_of_gene[gid].add(rxn.id)
        for met in rxn.metabolites:
            if met.name in CURRENCY:
                dropped += len(gene_ids)
                continue
            genes_of_metabolite[met.id] |= gene_ids
            for gid in gene_ids:
                metabolites_of_gene[gid].add(met.id)
    adjacency: dict[str, frozenset[str]] = {}
    for gid, mets in metabolites_of_gene.items():
        partners: set[str] = set()
        for mid in mets:
            partners |= genes_of_metabolite[mid]
        adjacency[gid] = frozenset(partners - {gid})
    return (
        adjacency,
        {gid: frozenset(r) for gid, r in reactions_of_gene.items()},
        dropped,
    )


def adjacent_enzyme_orfs(
    model: cobra.Model, pathway_orfs: set[str]
) -> tuple[list[str], int]:
    """Yeast9 genes one reaction hop from a pathway gene's reaction.

    A gene qualifies when one of its reactions shares at least one non-currency metabolite
    with a reaction catalyzed by a pathway gene, and the gene is not itself a pathway gene.
    """
    pathway_rxns = {
        r.id for orf in pathway_orfs for r in model.genes.get_by_id(orf).reactions
    }
    shared_metabolites = {
        met.id
        for rid in pathway_rxns
        for met in model.reactions.get_by_id(rid).metabolites
        if met.name not in CURRENCY
    }
    hits: set[str] = set()
    for mid in shared_metabolites:
        for rxn in model.metabolites.get_by_id(mid).reactions:
            hits |= {g.id for g in rxn.genes}
    return sorted(hits - pathway_orfs), len(shared_metabolites)


def quartiles(values: np.ndarray) -> dict[str, float]:
    """Median with the interquartile range, or NaNs when nothing is finite."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "n": 0}
    return {
        "median": float(np.median(finite)),
        "q1": float(np.percentile(finite, 25)),
        "q3": float(np.percentile(finite, 75)),
        "n": int(finite.size),
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    print(f"reading {MODEL_PATH}")
    model = cobra.io.read_sbml_model(MODEL_PATH)
    model_genes = {g.id for g in model.genes}
    print(
        f"  Yeast9: {len(model.reactions)} reactions, {len(model.metabolites)} "
        f"metabolites, {len(model.genes)} genes"
    )

    print("\nloading the genome and the nine CGT gene graphs")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    graph_store = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    adjacency: dict[str, dict[str, frozenset[str]]] = {}
    graph_stats: dict[str, dict[str, object]] = {}
    for name, attr, directed, _label in CGT_GRAPHS:
        g = getattr(graph_store, attr).graph
        adjacency[name] = build_cgt_adjacency(g, directed)
        graph_stats[name] = {
            "directed": bool(directed),
            "n_nodes": int(g.number_of_nodes()),
            "n_edges": int(g.number_of_edges()),
        }
        print(
            f"  {name:26s} directed {str(directed):5s} "
            f"{g.number_of_nodes():5d} nodes  {g.number_of_edges():8d} edges"
        )

    print("\nbuilding the Yeast9 gene-reaction-metabolite graph")
    met_adjacency, reactions_of_gene, dropped = build_metabolic_adjacency(model)
    adjacency[METABOLIC_GRAPH] = met_adjacency
    n_met_edges = sum(len(v) for v in met_adjacency.values()) // 2
    graph_stats[METABOLIC_GRAPH] = {
        "directed": False,
        "n_nodes": len(met_adjacency),
        "n_edges": int(n_met_edges),
        "currency_incidences_dropped": int(dropped),
    }
    print(
        f"  {METABOLIC_GRAPH:26s} directed False {len(met_adjacency):5d} nodes  "
        f"{n_met_edges:8d} edges  ({dropped} gene-metabolite incidences dropped as "
        f"currency)"
    )

    print("\nbuilding the union of all ten")
    union: dict[str, set[str]] = defaultdict(set)
    for name in GRAPH_ORDER[:-1]:
        for node, nbrs in adjacency[name].items():
            union[node] |= nbrs
    adjacency[UNION_GRAPH] = {k: frozenset(v) for k, v in union.items()}
    n_union_edges = sum(len(v) for v in adjacency[UNION_GRAPH].values())
    graph_stats[UNION_GRAPH] = {
        "directed": False,
        "n_nodes": len(adjacency[UNION_GRAPH]),
        "n_edges_directed_count": int(n_union_edges),
    }
    print(
        f"  {UNION_GRAPH:26s} {len(adjacency[UNION_GRAPH]):5d} nodes  "
        f"{n_union_edges:8d} out-edges counted once per direction present"
    )

    # --- seed classes ------------------------------------------------------------------
    pathway_orfs = {}
    for name in PATHWAY_GENES:
        orf = genome.resolve_gene_name(name).systematic_name
        if orf not in model_genes:
            raise ValueError(f"pathway gene {name} ({orf}) is not a Yeast9 gene")
        pathway_orfs[name] = orf
    pathway_orf_set = set(pathway_orfs.values())

    regulator_orfs = {}
    for name in sorted(REGULATOR_ARM):
        orf = genome.resolve_gene_name(name).systematic_name
        if orf != REGULATOR_ARM[name]:
            raise ValueError(f"{name} resolves to {orf}, not {REGULATOR_ARM[name]}")
        regulator_orfs[name] = orf

    adjacent_orfs, n_shared_mets = adjacent_enzyme_orfs(model, pathway_orf_set)
    print(
        f"\nadjacent enzyme class: {len(adjacent_orfs)} Yeast9 genes share at least one "
        f"of the {n_shared_mets} non-currency metabolites carried by a pathway gene's "
        f"reactions"
    )

    def display_name(orf: str) -> str:
        label = model.genes.get_by_id(orf).name
        return label if label else orf

    seeds: list[tuple[str, str, str, bool]] = []
    for name in sorted(regulator_orfs):
        seeds.append(("regulator", name, regulator_orfs[name], False))
    for name in PATHWAY_GENES:
        seeds.append(("pathway enzyme", name, pathway_orfs[name], True))
    for orf in adjacent_orfs:
        seeds.append(("adjacent enzyme", display_name(orf), orf, True))
    classes_df = pd.DataFrame(
        seeds, columns=["class", "gene", "orf", "in_yeast9"]
    )
    class_sizes = {c: int((classes_df["class"] == c).sum()) for c in CLASSES}
    print(f"class sizes: {class_sizes}")

    # --- the readout locus, as a reaction bitmask --------------------------------------
    rxn_index = {r.id: i for i, r in enumerate(model.reactions)}
    gene_mask: dict[str, int] = {}
    for gid, rxns in reactions_of_gene.items():
        mask = 0
        for rid in rxns:
            mask |= 1 << rxn_index[rid]
        gene_mask[gid] = mask
    locus_rxns = {
        r.id for orf in pathway_orf_set for r in model.genes.get_by_id(orf).reactions
    } | set(READOUT_EXCHANGES)
    locus_mask = 0
    for rid in sorted(locus_rxns):
        locus_mask |= 1 << rxn_index[rid]
    print(
        f"readout locus: {len(locus_rxns)} reactions "
        f"({len(locus_rxns) - len(READOUT_EXCHANGES)} catalyzed by the thirteen pathway "
        f"genes, plus the {len(READOUT_EXCHANGES)} fatty acid exchanges)"
    )

    # --- reach ------------------------------------------------------------------------
    print(f"\nBFS: {len(seeds)} seeds x {len(GRAPH_ORDER)} graphs x {len(DEPTHS)} depths")
    rows = []
    for graph_name in GRAPH_ORDER:
        adj = adjacency[graph_name]
        for klass, gene, orf, _in_y9 in seeds:
            out_degree = len(adj.get(orf, frozenset()))
            levels = bfs_levels(adj, orf, max(DEPTHS))
            for k in DEPTHS:
                reached = levels[k - 1]
                metabolic = reached & model_genes
                mask = 0
                for gid in metabolic:
                    mask |= gene_mask[gid]
                n_touched = mask.bit_count()
                n_off = (mask & ~locus_mask).bit_count()
                rows.append(
                    {
                        "seed_gene": gene,
                        "seed_orf": orf,
                        "class": klass,
                        "graph": graph_name,
                        "k": k,
                        "n_reached": len(reached),
                        "n_metabolic": len(metabolic),
                        "n_on_locus": len(reached & pathway_orf_set),
                        "n_reactions_touched": n_touched,
                        "n_reactions_off_locus": n_off,
                        "share_off_locus": (
                            n_off / n_touched if n_touched > 0 else float("nan")
                        ),
                        "out_degree": out_degree,
                    }
                )
        print(f"  {graph_name} done")
    reach_df = pd.DataFrame(rows)

    # --- summary ----------------------------------------------------------------------
    per_graph: dict[str, dict] = {}
    for graph_name in GRAPH_ORDER:
        per_class: dict[str, dict] = {}
        for klass in CLASSES:
            per_k: dict[str, dict] = {}
            for k in DEPTHS:
                sub = reach_df[
                    (reach_df["graph"] == graph_name)
                    & (reach_df["class"] == klass)
                    & (reach_df["k"] == k)
                ]
                per_k[str(k)] = {
                    "n_reached": quartiles(sub["n_reached"].to_numpy(dtype=float)),
                    "n_reactions_off_locus": quartiles(
                        sub["n_reactions_off_locus"].to_numpy(dtype=float)
                    ),
                    "share_off_locus": quartiles(
                        sub["share_off_locus"].to_numpy(dtype=float)
                    ),
                    "out_degree": quartiles(sub["out_degree"].to_numpy(dtype=float)),
                }
            per_class[klass] = per_k
        per_graph[graph_name] = per_class

    mann_whitney: dict[str, dict] = {}
    for k in (1, 2):
        at_k: dict[str, dict] = {}
        base = reach_df[(reach_df["graph"] == UNION_GRAPH) & (reach_df["k"] == k)]
        x = base[base["class"] == "regulator"]["n_reactions_off_locus"].to_numpy(
            dtype=float
        )
        for other in ("pathway enzyme", "adjacent enzyme"):
            y = base[base["class"] == other]["n_reactions_off_locus"].to_numpy(
                dtype=float
            )
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            at_k[f"regulator vs {other}"] = {
                "u": float(stat),
                "p": float(p),
                "n_regulator": int(x.size),
                f"n_{other.replace(' ', '_')}": int(y.size),
                "median_regulator": float(np.median(x)),
                f"median_{other.replace(' ', '_')}": float(np.median(y)),
            }
        mann_whitney[f"k{k}"] = at_k

    summary = {
        "what_this_is": (
            "A descriptive count of how far a perturbation seed reaches on annotation "
            "graphs, and how many of the Yeast9 reactions in that reach sit off the "
            "measured locus. It is not a titer, a flux, or an interaction score, and it "
            "does not test any mechanism connecting reach to an interaction."
        ),
        "model_path": MODEL_PATH,
        "model_sha256": sha256_of_file(MODEL_PATH),
        "yeast9": {
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
        },
        "class_sizes": class_sizes,
        "adjacent_enzyme_rule": (
            "a Yeast9 gene that is not one of the thirteen pathway genes and whose "
            "reactions share at least one non-currency metabolite with a reaction "
            "catalyzed by a pathway gene"
        ),
        "n_shared_metabolites": int(n_shared_mets),
        "readout_locus": {
            "n_reactions": len(locus_rxns),
            "exchanges": READOUT_EXCHANGES,
        },
        "currency_metabolites": sorted(CURRENCY),
        "graphs": graph_stats,
        "depths": list(DEPTHS),
        "per_graph": per_graph,
        "mann_whitney_n_reactions_off_locus_union": mann_whitney,
        "caveats": CAVEATS,
    }

    reach_df.to_csv(osp.join(RESULTS_DIR, "reach.csv"), index=False)
    classes_df.to_csv(osp.join(RESULTS_DIR, "classes.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nwritten\n  {osp.join(RESULTS_DIR, 'reach.csv')}"
        f"\n  {osp.join(RESULTS_DIR, 'classes.csv')}"
        f"\n  {osp.join(RESULTS_DIR, 'summary.json')}"
    )

    make_figure(reach_df)
    print_summary(summary, reach_df)


def make_figure(reach_df: pd.DataFrame) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(mm_to_in(PANEL_WIDTHS_MM["full"]), mm_to_in(125.0)),
    )
    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.20, top=0.90, wspace=0.22, hspace=0.80
    )

    # (a) genes reached against hop depth on the union graph, median with the IQR band.
    ax = axes[0][0]
    for klass in CLASSES:
        med, lo, hi = [], [], []
        for k in DEPTHS:
            v = reach_df[
                (reach_df["graph"] == UNION_GRAPH)
                & (reach_df["class"] == klass)
                & (reach_df["k"] == k)
            ]["n_reached"].to_numpy(dtype=float)
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        ax.fill_between(
            DEPTHS, lo, hi, color=CLASS_COLOR[klass], alpha=0.25, linewidth=0
        )
        ax.plot(
            DEPTHS,
            med,
            color=CLASS_COLOR[klass],
            linewidth=1.0,
            marker="o",
            markersize=2.0,
            label=klass,
        )
    ax.set_yscale("log")
    ax.set_xticks(list(DEPTHS))
    ax.set_xlabel("hop depth k")
    ax.set_ylabel("genes within k hops")
    ax.set_title("reach on the union graph", fontsize=6)
    ax.legend(fontsize=5, loc="lower right", handlelength=1.2, labelspacing=0.25)

    # (b) off-locus Yeast9 reactions touched at k=1, median per class on every graph.
    ax = axes[0][1]
    width = 0.27
    for i, klass in enumerate(CLASSES):
        meds = [
            np.median(
                reach_df[
                    (reach_df["graph"] == g)
                    & (reach_df["class"] == klass)
                    & (reach_df["k"] == 1)
                ]["n_reactions_off_locus"].to_numpy(dtype=float)
            )
            for g in GRAPH_ORDER
        ]
        ax.bar(
            [j + (i - 1) * width for j in range(len(GRAPH_ORDER))],
            meds,
            width=width,
            color=CLASS_COLOR[klass],
            edgecolor="black",
            linewidth=0.4,
            label=klass,
        )
    ax.set_xticks(range(len(GRAPH_ORDER)))
    ax.set_xticklabels([GRAPH_LABEL[g] for g in GRAPH_ORDER], rotation=90)
    ax.set_ylabel("off-locus reactions at k=1")
    ax.set_title("median off-locus reactions touched, one hop", fontsize=6)
    ax.legend(fontsize=5, loc="upper left", handlelength=1.0, labelspacing=0.25)

    # (c) share of touched reactions that sit off the locus, union graph, k=1 and k=2.
    ax = axes[1][0]
    rng = np.random.default_rng(0)
    positions, labels = [], []
    pos = 0.0
    for k in (1, 2):
        for klass in CLASSES:
            v = reach_df[
                (reach_df["graph"] == UNION_GRAPH)
                & (reach_df["class"] == klass)
                & (reach_df["k"] == k)
            ]["share_off_locus"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            jitter = rng.uniform(-0.16, 0.16, size=v.size)
            ax.scatter(
                np.full(v.size, pos) + jitter,
                v,
                s=1.2,
                color=CLASS_COLOR[klass],
                edgecolors="none",
                zorder=2,
            )
            if v.size:
                ax.plot(
                    [pos - 0.3, pos + 0.3],
                    [np.median(v)] * 2,
                    color="black",
                    linewidth=0.8,
                    zorder=3,
                )
            positions.append(pos)
            labels.append(f"{klass.split()[0]}\nk={k}")
            pos += 1.0
        pos += 0.4
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("share of touched reactions off locus")
    ax.set_title("off-locus share on the union graph", fontsize=6)

    # (d) out-degree of the seed on every graph, median per class.
    ax = axes[1][1]
    for i, klass in enumerate(CLASSES):
        meds = [
            np.median(
                reach_df[
                    (reach_df["graph"] == g)
                    & (reach_df["class"] == klass)
                    & (reach_df["k"] == 1)
                ]["out_degree"].to_numpy(dtype=float)
            )
            for g in GRAPH_ORDER
        ]
        ax.bar(
            [j + (i - 1) * width for j in range(len(GRAPH_ORDER))],
            meds,
            width=width,
            color=CLASS_COLOR[klass],
            edgecolor="black",
            linewidth=0.4,
            label=klass,
        )
    ax.set_xticks(range(len(GRAPH_ORDER)))
    ax.set_xticklabels([GRAPH_LABEL[g] for g in GRAPH_ORDER], rotation=90)
    ax.set_ylabel("out-degree of the seed")
    ax.set_title("median seed out-degree per graph", fontsize=6)
    ax.legend(fontsize=5, loc="upper left", handlelength=1.0, labelspacing=0.25)

    for row in axes:
        for ax in row:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)

    # Panel letters go in FIGURE coordinates, not axes coordinates, so a letter cannot be
    # pushed off the top of the canvas the way an axes-anchored offset can be.
    letters = (("a", axes[0][0]), ("b", axes[0][1]), ("c", axes[1][0]), ("d", axes[1][1]))
    for letter, ax in letters:
        box = ax.get_position()
        fig.text(
            max(box.x0 - 0.048, 0.004),
            min(box.y1 + 0.022, 0.97),
            letter,
            fontsize=8,
            fontweight="bold",
            fontfamily="Arial",
            ha="left",
            va="bottom",
        )

    png = osp.join(IMAGES_DIR, f"{FIG_STEM}.png")
    svg = osp.join(IMAGES_DIR, f"{FIG_STEM}.svg")
    fig.savefig(png, dpi=600)
    savefig_true_size_svg(fig, svg)
    plt.close(fig)
    print(f"\nfigure written\n  {png}\n  {svg}")


def print_summary(summary: dict, reach_df: pd.DataFrame) -> None:
    print("\n" + "=" * 92)
    print("PERTURBATION REACH, A COUNT ON ANNOTATION GRAPHS")
    print("=" * 92)
    print(summary["what_this_is"])

    print(f"\nclass sizes: {summary['class_sizes']}")
    print(
        f"readout locus: {summary['readout_locus']['n_reactions']} Yeast9 reactions; "
        f"adjacent enzyme rule keyed on {summary['n_shared_metabolites']} non-currency "
        f"metabolites"
    )

    print("\nUNION GRAPH, median [q1, q3] per class and hop depth")
    header = (
        f"{'k':>2s} {'class':>16s} {'genes reached':>26s} "
        f"{'off-locus reactions':>26s} {'off-locus share':>22s}"
    )
    print(header)
    print("-" * len(header))
    for k in summary["depths"]:
        for klass in CLASSES:
            s = summary["per_graph"][UNION_GRAPH][klass][str(k)]
            def fmt(d: dict, digits: int = 0) -> str:
                return (
                    f"{d['median']:.{digits}f} [{d['q1']:.{digits}f}, "
                    f"{d['q3']:.{digits}f}]"
                )
            print(
                f"{k:2d} {klass:>16s} {fmt(s['n_reached']):>26s} "
                f"{fmt(s['n_reactions_off_locus']):>26s} "
                f"{fmt(s['share_off_locus'], 3):>22s}"
            )

    print("\nEVERY GRAPH at k=1, median off-locus reactions touched")
    line = f"{'graph':>26s} " + " ".join(f"{c:>16s}" for c in CLASSES)
    print(line)
    print("-" * len(line))
    for graph_name in GRAPH_ORDER:
        vals = [
            summary["per_graph"][graph_name][c]["1"]["n_reactions_off_locus"]["median"]
            for c in CLASSES
        ]
        print(
            f"{graph_name:>26s} " + " ".join(f"{v:>16.1f}" for v in vals)
        )

    print("\nMann-Whitney U, two-sided, n_reactions_off_locus on the union graph")
    for k_label, tests in summary[
        "mann_whitney_n_reactions_off_locus_union"
    ].items():
        for name, t in tests.items():
            other_n = [v for kk, v in t.items() if kk.startswith("n_") and kk != "n_regulator"]
            other_med = [
                v for kk, v in t.items()
                if kk.startswith("median_") and kk != "median_regulator"
            ]
            print(
                f"  {k_label} {name}: U {t['u']:.1f}, p {t['p']:.3g}, "
                f"n {t['n_regulator']} vs {other_n[0]}, medians "
                f"{t['median_regulator']:.1f} vs {other_med[0]:.1f}"
            )

    zero_medians = [
        (g, c, k)
        for g in GRAPH_ORDER
        for c in CLASSES
        for k in summary["depths"]
        if summary["per_graph"][g][c][str(k)]["n_reached"]["median"] == 0.0
    ]
    if zero_medians:
        print(
            f"\n{len(zero_medians)} of the {len(GRAPH_ORDER) * len(CLASSES) * 3} "
            f"graph-class-depth cells have a median reach of zero, so the seeds of that "
            f"class are mostly absent from that graph. First few: {zero_medians[:6]}"
        )

    print("\ncaveats a Methods paragraph would have to state")
    for c in summary["caveats"]:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
