# experiments/010-kuzmin-tmi/scripts/carotenoid_graph_enrichment.py
# [[experiments.010-kuzmin-tmi.scripts.carotenoid_graph_enrichment]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/carotenoid_graph_enrichment
"""Are GEA1 / VPS1 / PCT1 / HXK2 enriched for carotenoid-precursor genes in the 010 graphs?

For each query gene, each of the nine graphs the 010 model trained on (physical,
regulatory, tflink, six STRING v12.0 channels), and each carotenoid-relevant gene set,
report degree, the number of neighbors inside the set, a hypergeometric enrichment
p-value, fold enrichment, and the shortest-path distance to the set. Yeast has no native
carotenoid pathway, so "carotenoid-relevant" means the precursor supply: the mevalonate
pathway through GGPP, the competing squalene branch, and the GO isoprenoid-biosynthesis
annotation set.
"""

import os
import os.path as osp
from itertools import combinations

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import hypergeom

from torchcell.graph import SCerevisiaeGraph
from torchcell.graph.graph import build_gene_multigraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

# The nine graphs in experiments/010-kuzmin-tmi/conf/equivariant_cell_graph_transformer_cabbi_000.yaml
GRAPH_NAMES = [
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

QUERY_GENES = ["GEA1", "VPS1", "PCT1", "HXK2"]

# Mevalonate pathway acetyl-CoA -> FPP -> GGPP: the native precursor supply for any
# heterologous carotenoid pathway (crtE/crtYB/crtI condense GGPP).
MVA_GGPP_BACKBONE = [
    "ERG10",  # acetoacetyl-CoA thiolase
    "ERG13",  # HMG-CoA synthase
    "HMG1",  # HMG-CoA reductase
    "HMG2",  # HMG-CoA reductase
    "ERG12",  # mevalonate kinase
    "ERG8",  # phosphomevalonate kinase
    "MVD1",  # mevalonate diphosphate decarboxylase (ERG19)
    "IDI1",  # IPP isomerase
    "ERG20",  # FPP synthase
    "BTS1",  # GGPP synthase
]
# The FPP-consuming branch that competes with GGPP for precursor.
SQUALENE_BRANCH = ["ERG9", "ERG1"]
GO_ISOPRENOID_BIOSYNTHESIS = "GO:0008299"


class EnrichmentRow(BaseModel):
    """One query gene scored against one gene set in one graph."""

    query_gene: str
    query_systematic: str
    graph: str
    gene_set: str
    n_graph_nodes: int
    set_size_in_graph: int
    degree: int
    neighbors_in_set: int
    expected_in_set: float
    fold_enrichment: float | None
    hypergeom_p: float
    shortest_path_to_set: int | None
    set_neighbors: str
    frac_genes_with_any_set_neighbor: float


def resolve(genome: SCerevisiaeGenome, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        res = genome.resolve_gene_name(name)
        if res.systematic_name is None or res.status.name not in ("CURRENT", "RENAMED"):
            raise ValueError(f"{name} did not resolve to a live gene: {res}")
        out[name] = res.systematic_name
    return out


def go_annotated_genes(graph: SCerevisiaeGraph, go_id: str) -> set[str]:
    dag = graph.genome.go_dag
    terms = {go_id} | set(dag[go_id].get_all_children())
    genes: set[str] = set()
    for gene in graph.genome.gene_set:
        for detail in graph.G_raw.nodes[gene].get("go_details", []):
            if detail["go"]["go_id"] in terms:
                genes.add(gene)
    return genes


def score(
    G: nx.Graph, gene: str, query_name: str, graph_name: str, set_name: str, gene_set: set[str]
) -> EnrichmentRow:
    nodes = set(G.nodes)
    N = len(nodes)
    set_in_graph = gene_set & nodes
    K = len(set_in_graph)
    neighbors = set(G.neighbors(gene)) - {gene} if gene in G else set()
    k = len(neighbors)
    hits = sorted(neighbors & set_in_graph)
    m = len(hits)
    expected = k * K / N if N else 0.0
    fold = (m / expected) if expected > 0 else None
    p = float(hypergeom.sf(m - 1, N, K, k)) if k > 0 else 1.0
    if gene in G and K > 0:
        lengths = nx.single_source_shortest_path_length(G, gene)
        dists = [d for g, d in lengths.items() if g in set_in_graph and g != gene]
        shortest = min(dists) if dists else None
    else:
        shortest = None
    # How common is it, among all connected genes, to touch the set at all?
    connected = [g for g in nodes if G.degree(g) > 0]
    any_hit = sum(1 for g in connected if (set(G.neighbors(g)) - {g}) & set_in_graph)
    frac_any = any_hit / len(connected) if connected else 0.0
    return EnrichmentRow(
        query_gene=query_name,
        query_systematic=gene,
        graph=graph_name,
        gene_set=set_name,
        n_graph_nodes=N,
        set_size_in_graph=K,
        degree=k,
        neighbors_in_set=m,
        expected_in_set=expected,
        fold_enrichment=fold,
        hypergeom_p=p,
        shortest_path_to_set=shortest,
        set_neighbors=";".join(hits),
        frac_genes_with_any_set_neighbor=frac_any,
    )


def main() -> None:
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
    multigraph = build_gene_multigraph(graph=graph, graph_names=GRAPH_NAMES)

    query = resolve(genome, QUERY_GENES)
    backbone = resolve(genome, MVA_GGPP_BACKBONE)
    squalene = resolve(genome, SQUALENE_BRANCH)
    go_iso = go_annotated_genes(graph, GO_ISOPRENOID_BIOSYNTHESIS)
    go_name = genome.go_dag[GO_ISOPRENOID_BIOSYNTHESIS].name
    gene_sets: dict[str, set[str]] = {
        "mva_ggpp_backbone": set(backbone.values()),
        "mva_ggpp_plus_squalene_branch": set(backbone.values()) | set(squalene.values()),
        f"go_{GO_ISOPRENOID_BIOSYNTHESIS.replace(':', '_')}_{go_name.replace(' ', '_')}": go_iso,
    }
    print("query:", query)
    print("backbone:", backbone)
    print("squalene branch:", squalene)
    print(f"{GO_ISOPRENOID_BIOSYNTHESIS} ({go_name}): {len(go_iso)} genes")

    # Adjacency is scored undirected: for the directed regulatory and tflink graphs a
    # query gene that is a TARGET of a set member counts as adjacent to it.
    graphs: dict[str, nx.Graph] = {}
    for name in GRAPH_NAMES:
        G = multigraph[name].graph
        print(f"{name}: directed={G.is_directed()} nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
        graphs[name] = G.to_undirected() if G.is_directed() else G
    union = nx.Graph()
    for G in graphs.values():
        union.add_nodes_from(G.nodes)
        union.add_edges_from(G.edges)
    graphs["union_of_9"] = union

    rows: list[EnrichmentRow] = []
    for gname, G in graphs.items():
        for qname, qsys in query.items():
            for sname, sset in gene_sets.items():
                rows.append(score(G, qsys, qname, gname, sname, sset))
    df = pd.DataFrame([r.model_dump() for r in rows])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = osp.join(RESULTS_DIR, "carotenoid_graph_enrichment.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # Are the four query genes connected to EACH OTHER in any graph?
    pair_rows = []
    for gname, G in graphs.items():
        for a, b in combinations(QUERY_GENES, 2):
            sa, sb = query[a], query[b]
            edge = G.has_edge(sa, sb)
            if sa in G and sb in G and nx.has_path(G, sa, sb):
                d = nx.shortest_path_length(G, sa, sb)
            else:
                d = None
            pair_rows.append({"graph": gname, "gene_a": a, "gene_b": b, "edge": edge, "distance": d})
    pair_df = pd.DataFrame(pair_rows)
    out_pairs = osp.join(RESULTS_DIR, "carotenoid_graph_enrichment_query_pairs.csv")
    pair_df.to_csv(out_pairs, index=False)
    print(f"wrote {out_pairs}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    cols = [
        "query_gene", "graph", "gene_set", "degree", "set_size_in_graph",
        "neighbors_in_set", "expected_in_set", "fold_enrichment", "hypergeom_p",
        "shortest_path_to_set", "frac_genes_with_any_set_neighbor", "set_neighbors",
    ]
    print(df[cols].to_string(index=False))
    print(pair_df[pair_df["edge"]].to_string(index=False))


if __name__ == "__main__":
    main()
