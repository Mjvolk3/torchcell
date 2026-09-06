# experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py
# [[experiments.006-kuzmin-tmi.scripts.dcell_model_go_stats]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats
"""Measured statistics of the Gene Ontology DAG that structures the DCell baseline.

Rebuilds the filtered GO DAG exactly as ``experiments/006-kuzmin-tmi/scripts/dcell.py``
does for the trigenic DCell run (``conf/dcell_kuzmin2018_tmi*.yaml``): the cached
``SCerevisiaeGraph.G_go`` (SGD ``go_details`` annotations over the 6,607-gene S288C
reference, three GO namespaces joined under ``GO:ROOT``), then in order
``filter_go_IGI`` -> ``filter_redundant_terms`` -> ``filter_by_contained_genes(n=4)``.
No date cutoff is applied by that config (``model.go_date_filter`` is absent), so the
optional 2017-07-19 cutoff used in the earlier experiment-005 exploration is reported
as a separate reference row, not as part of the pipeline.

Reported (CSV under results/dcell_model/):
  go_filter_stages.csv        terms, edges, annotations, covered genes, leaves per stage
  go_terms_final.csv          per-term: namespace, stratum, direct + contained genes, widths
  go_strata.csv               terms per stratum (the model's processing order)
  go_genes_final.csv          per-gene: number of terms it is annotated to (0 = uncovered)
  go_edges_final.csv          the 3,208 hierarchy edges of the final DAG (child, parent)
  go_annotations_final.csv    the 59,986 direct (term, gene) annotation rows of the final DAG
  example_triple.csv          the real Kuzmin 2018 trigenic record drawn in the DAG panel
                              (picked by rule from the local 005 build; see pick_example_triple)
  go_evidence_codes.csv       annotation evidence codes retained in the final DAG
  dcell_model_size.csv        parameter count implied by the widths, vs the wandb-logged
                              ``model/params_*`` of the trigenic runs (frozen pull)
  dcell_wandb_model_size.csv  the frozen wandb pull (one row per run)
  dcell_vs_paper.csv          the published DCell ontology and model (Ma et al. 2018, every
                              value a verbatim quote from the sha256-pinned OCR text of the
                              mirror key maUsingDeepLearning2018) against the same quantities
                              measured on the frozen DAG above (see PUBLISHED)

Panels (true-size SVG + PNG under $ASSET_IMAGES_DIR/006-kuzmin-tmi/):
  dcell_model_go_dag (the whole filtered DAG, strata as layers, one triple deletion
  highlighted), dcell_model_terms_per_stratum, dcell_model_genes_per_term,
  dcell_model_terms_per_gene

Tables: paper/nature-biotech/sections/tab-dcell-model-go-filter.tex
        paper/nature-biotech/sections/tab-dcell-model-vs-paper.tex

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py --from-csv
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py --dag-only
        (rebuild the DAG without the wandb pull, check it against the frozen
        go_terms_final.csv, and freeze go_edges_final.csv + go_annotations_final.csv)
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py --pick-triple
        (re-pick the example triple from the local 005 LMDB, freeze example_triple.csv, render)
"""

import argparse
import json
import math
import os
import os.path as osp
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pydantic import BaseModel

from torchcell.data.cell_data import compute_strata
from torchcell.graph import (
    SCerevisiaeGraph,
    filter_by_contained_genes,
    filter_by_date,
    filter_go_IGI,
    filter_redundant_terms,
)
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.utils import PANEL_WIDTHS_MM, PLOT_PALETTE, mm_to_in, savefig_true_size_svg

# Set AFTER the torchcell imports: torchcell.graph applies the repo mplstyle on import.
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

RESULTS_DIR = "experiments/006-kuzmin-tmi/results/dcell_model"
IMG_DIR = osp.join(ASSET_IMAGES_DIR, "006-kuzmin-tmi")
TEX_DIR = "paper/nature-biotech/sections"

# The trigenic DCell configuration (conf/dcell_kuzmin2018_tmi*.yaml + scripts/dcell.py).
MIN_GENES = 4  # scripts/dcell.py: wandb.config.model.get("go_min_genes", 4)
MIN_GENES_WORD = {4: "four", 6: "six"}[MIN_GENES]  # prose form for the comparison table
DATE_FILTER = None  # scripts/dcell.py: wandb.config.model.get("go_date_filter", None)
SUBSYSTEM_MIN = 20  # model.subsystem_output_min
SUBSYSTEM_RATIO = 0.3  # model.subsystem_output_max_mult
REFERENCE_DATE_CUTOFF = "2017-07-19"  # experiment-005 exploration only; not in the 006 run
WANDB_ENTITY = "zhao-group"
WANDB_PROJECTS = ["torchcell_006-kuzmin-tmi_dcell", "torchcell_005-kuzmin2018-tmi_dcell"]
WANDB_FIELDS = [
    "model/num_go_terms",
    "model/num_subsystems",
    "model/params_total",
    "model/params_dcell",
    "model/params_dcell_linear",
    "model/params_subsystems",
]

PANEL_W_MM = PANEL_WIDTHS_MM["third"]
PANEL_H_MM = 44.0
DAG_W_MM = PANEL_WIDTHS_MM["wide"]  # the DAG panel; the equations column fills the rest of 180 mm
# The DAG panel's height sets the height of the equations column beside it in FigS-dcell-model
# (dcell_model_compose_figure.py reads the SVG and sizes the column to it). Its visible
# content is flush with the image: the axes frame at the top edge, the legend at the bottom.
DAG_H_MM = 80.0
DAG_TOP = 0.997  # axes frame 0.24 mm below the image's top edge (the frame stroke)
DAG_BOTTOM_CLEAR = 0.006  # legend's lower edge ~0.5 mm above the image's bottom edge

# --------------------------------------------------------------------------- the published model
# Ma et al. 2018 (Nat. Methods 15, 290-298), mirror key maUsingDeepLearning2018, pulled over
# tc-lit on 2026.09.05 and verified against the manifest: paper.md (MinerU OCR of the paper
# and its Online Methods) sha256 below. Every published value in the comparison table is a
# verbatim quote from that text; what the text does not state is "not reported".
PAPER_KEY = "maUsingDeepLearning2018"
PAPER_MD_SHA256 = "ac837bc358ea4969a72789e66e31380bfdcbd7b8aea98dec47b108ee21d2070c"


class PublishedValue(BaseModel):
    """One row of the published-vs-TorchCell table: the published value with its evidence."""

    row: str
    published: str  # the table cell (LaTeX)
    quote: str | None  # verbatim from paper.md, or None when the paper does not state it
    where: str  # section of the paper the quote is from


PUBLISHED = [
    PublishedValue(
        row="GO release",
        published="not reported; cites the GO Consortium (ref.~9, \\emph{Nucleic Acids Res.} 2016)",
        quote="the Gene Ontology (GO), a literature-curated reference database from which we extracted 2,526 cellular subsystems (intracellular components, processes or functions)9 [...] 9. The Gene Ontology Consortium. Expansion of the Gene Ontology knowledgebase and resources. Nucleic Acids Res. 45, D331–D338 (2016).",
        where="Introduction; reference 9",
    ),
    PublishedValue(
        row="Annotation source",
        published="not reported (``gene-to-term annotations'')",
        quote="a biological ontology consisting of terms representing cellular subsystems, child–parent relations representing containment of one term by another, and gene-to-term annotations",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Genes",
        published="the genes disrupted in the training genotypes; count not reported",
        quote=r"For each sample i, $X _ { i } \in \ : R ^ { M }$ denotes the genotype, represented as a binary vector of states on $M$ genes $\mathbf { \nabla } \cdot \mathbf { 1 } =$ disrupted; $0 =$ wild type) [...] 2. Terms containing fewer than six yeast genes disrupted in the available genotypes",
        where="Online Methods, DCell architecture and training algorithm (OCR of the inline math kept as is); Preparation of ontologies",
    ),
    PublishedValue(
        row="Evidence filter",
        published="terms with evidence code IGI removed",
        quote="1. Terms with the evidence code ‘inferred by genetic interaction’ (IGI), to avoid potential circularity in predicting genetic interactions in the genotype–phenotype samples.",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Redundancy filter",
        published="terms redundant with respect to their children",
        quote="3. Terms that are redundant with respect to their children terms in the ontology.",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Containment threshold",
        published="fewer than six disrupted genes, counted over the term and its descendants",
        quote="2. Terms containing fewer than six yeast genes disrupted in the available genotypes (with ‘containment’ defined as all genes annotated to that term or its descendants).",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Filter order",
        published="IGI, containment, redundancy (as listed)",
        quote="We used the following criteria to filter (remove) terms from GO: 1. Terms with the evidence code ‘inferred by genetic interaction’ (IGI), to avoid potential circularity in predicting genetic interactions in the genotype–phenotype samples. 2. Terms containing fewer than six yeast genes disrupted in the available genotypes (with ‘containment’ defined as all genes annotated to that term or its descendants). 3. Terms that are redundant with respect to their children terms in the ontology.",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Removed-term rewiring",
        published="children connected to all parents",
        quote="When a term was removed, all children were connected directly to all parent terms to maintain the hierarchical structure.",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(
        row="Subsystems",
        published="2,526",
        quote="The remaining 2,526 terms were used to define the hierarchy of DCell subsystems.",
        where="Online Methods, Preparation of ontologies",
    ),
    PublishedValue(row="Hierarchy edges", published="not reported", quote=None, where=""),
    PublishedValue(
        row="Depth",
        published="12 layers",
        quote="The depth of both networks is 12 layers, on par with deep neural networks in other fields7.",
        where="Results, DCell design",
    ),
    PublishedValue(row="Leaves", published="not reported", quote=None, where=""),
    PublishedValue(
        row="Width rule",
        published="$\\max(20,\\lceil 0.3\\times\\text{genes contained by }t\\rceil)$",
        quote=r"L _ { o } ^ { ( t ) } = \operatorname* { m a x } \left( 2 0 , \left\lceil 0 . 3 * \mathrm { n u m b e r ~ o f ~ g e n e s ~ c o n t a i n e d ~ b y ~ } t \right\rceil \right)",
        where="Online Methods, equation (2) (OCR of the display math kept as is)",
    ),
    PublishedValue(
        row="Root readout",
        published="one output neuron; root width not reported",
        quote=r"the output layer, or root, is a single neuron representing cell phenotype [...] Linear in equation (3) denotes linear functions transforming multidimensional vector $O _ { i } ^ { ( t ) }$ into a scalar.",
        where="Results, DCell design; Online Methods, equation (3)",
    ),
    PublishedValue(
        row="Hidden units",
        published="97,181 (20 to 1,075 per subsystem)",
        quote="The use of multiple neurons (ranging from 20 to 1,075 per system; see Online Methods) acknowledges that cellular components are often multifunctional [...] By this design, the VNN embedded in GO includes 97,181 neurons; the corresponding model for CliXO includes 22,167 neurons.",
        where="Results, DCell design",
    ),
    PublishedValue(row="Parameters", published="not reported", quote=None, where=""),
    PublishedValue(
        row="Auxiliary loss",
        published="$\\alpha=0.3$, summed over $t\\neq r$, plus $\\lambda\\lVert W\\rVert_2$ ($\\lambda$ by four-fold CV)",
        quote=r"the parameter $\alpha$ $_ { ( = 0 . 3 ) }$ balances these two contributions. $\lambda$ is an $l _ { 2 }$ norm regularization factor determined by four-fold cross-validation.",
        where="Online Methods, equation (3) (OCR of the inline math kept as is)",
    ),
    PublishedValue(
        row="Training data",
        published="Costanzo 2010 ($\\sim$3 M examples) or Costanzo 2016 ($\\sim$8 M), single and double deletions",
        quote=r"Several forms of the model were employed in this study, trained on either Costanzo et al.16 (\~3 million training examples) or a more recently published update in 2016 ( ${ \sim } 8$ million training examples)15.",
        where="Online Methods, Training genotype-phenotype data (OCR of the inline math kept as is)",
    ),
]

NAMESPACE_COLOR = {
    "biological_process": PLOT_PALETTE[0],  # orange
    "molecular_function": PLOT_PALETTE[2],  # purple
    "cellular_component": PLOT_PALETTE[3],  # yellow
    "super_root": PLOT_PALETTE[5],  # gray
}
NAMESPACE_LABEL = {
    "biological_process": "Biological process",
    "molecular_function": "Molecular function",
    "cellular_component": "Cellular component",
    "super_root": "GO:ROOT",
}
HIGHLIGHT = PLOT_PALETTE[1]  # red: the perturbation and its paths to the root
FEED = PLOT_PALETTE[4]  # blue: the zeroed gene states entering their subsystems (dashed)
NAMESPACE_ROOTS = ("GO:0008150", "GO:0003674", "GO:0005575")

# The example strain drawn in the DAG panel is a real Kuzmin 2018 trigenic record from the
# local experiment-005 build (Kuzmin 2018 only, 91,050 records).
KUZMIN2018_LMDB = "data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/processed/lmdb"
TRIPLE_ANNOTATIONS = (5, 12)  # each gene must carry this many direct annotations, inclusive


# --------------------------------------------------------------------------- graph stats
def go_release(go_root: str) -> str:
    """Return the ``data-version`` line of the cached go.obo (the release used)."""
    with open(osp.join(go_root, "go.obo"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
    raise ValueError("go.obo carries no data-version line")


def annotation_pairs(G: nx.DiGraph, gene_set: set[str]) -> list[tuple[str, str]]:
    """(term, gene) pairs for every direct annotation of a reference gene."""
    return [
        (t, g)
        for t, d in G.nodes(data=True)
        for g in (d.get("gene_set") or [])
        if g in gene_set
    ]


def stage_row(name: str, G: nx.DiGraph, gene_set: set[str]) -> dict:
    pairs = annotation_pairs(G, gene_set)
    ns = Counter(d.get("namespace") for _, d in G.nodes(data=True))
    return {
        "stage": name,
        "terms": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "annotations": len(pairs),
        "genes_covered": len({g for _, g in pairs}),
        "leaves": sum(1 for n in G if G.in_degree(n) == 0),
        "roots": sum(1 for n in G if G.out_degree(n) == 0),
        "biological_process": ns.get("biological_process", 0),
        "molecular_function": ns.get("molecular_function", 0),
        "cellular_component": ns.get("cellular_component", 0),
    }


def contained_genes(G: nx.DiGraph, gene_set: set[str]) -> dict[str, int]:
    """Genes annotated to a term or any descendant (the DCell paper's 'containment')."""
    G_rev = G.reverse(copy=False)  # parent -> child
    out = {}
    for t in G.nodes():
        reach = nx.single_source_shortest_path_length(G_rev, t).keys()
        genes: set[str] = set()
        for n in reach:
            genes.update(g for g in (G.nodes[n].get("gene_set") or []) if g in gene_set)
        out[t] = len(genes)
    return out


def implied_parameters(G: nx.DiGraph, direct: dict[str, int]) -> dict[str, int]:
    """Parameter count of ``torchcell.models.dcell.DCell`` built on ``G``.

    Mirrors ``_calculate_input_dim`` / ``_calculate_output_dim`` / ``DCellSubsystem`` /
    ``linear_heads`` exactly: width = max(min, ceil(ratio * direct genes)); input = sum of
    child widths + max(direct genes, 1); subsystem = Linear(in, out) + BatchNorm1d(out);
    head = Linear(out, 1).
    """
    width = {t: max(SUBSYSTEM_MIN, math.ceil(SUBSYSTEM_RATIO * direct[t])) for t in G}
    subsystem = heads = 0
    for t in G:
        children = list(G.predecessors(t))  # edges are child -> parent
        in_dim = sum(width[c] for c in children) + max(direct[t], 1)
        in_dim = max(in_dim, 1)
        subsystem += in_dim * width[t] + width[t] + 2 * width[t]
        heads += width[t] + 1
    return {
        "params_subsystems": subsystem,
        "params_dcell_linear": heads,
        "params_total": subsystem + heads,
        "neurons": sum(width.values()),
        "width_min": min(width.values()),
        "width_max": max(width.values()),
    }, width


def load_raw_go() -> tuple[nx.DiGraph, set[str], str]:
    """The cached ``SCerevisiaeGraph.G_go`` over the 6,607-gene reference, plus the GO release."""
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    gene_set = set(genome.gene_set)
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )
    release = go_release(osp.join(DATA_ROOT, "data/go"))
    print(f"go.obo release {release}; reference genes {len(gene_set)}")
    return graph.G_go.copy(), gene_set, release


def filter_dag(G0: nx.DiGraph, gene_set: set[str]) -> nx.DiGraph:
    """The three filters in the order scripts/dcell.py applies them (no date cutoff)."""
    G1 = filter_by_date(G0, DATE_FILTER) if DATE_FILTER else G0
    return filter_by_contained_genes(filter_redundant_terms(filter_go_IGI(G1)), n=MIN_GENES, gene_set=gene_set)


def freeze_structure(G4: nx.DiGraph, gene_set: set[str]) -> None:
    """Freeze the final DAG's edges and direct annotation rows (what the DAG panel draws)."""
    pd.DataFrame(list(G4.edges()), columns=["child", "parent"]).sort_values(["child", "parent"]).to_csv(
        osp.join(RESULTS_DIR, "go_edges_final.csv"), index=False
    )
    pd.DataFrame(sorted(annotation_pairs(G4, gene_set)), columns=["term", "gene"]).to_csv(
        osp.join(RESULTS_DIR, "go_annotations_final.csv"), index=False
    )


def build_dag_only() -> None:
    """Rebuild the DAG without touching wandb, check it against the frozen term table, freeze edges."""
    G0, gene_set, release = load_raw_go()
    G4 = filter_dag(G0, gene_set)
    terms = pd.read_csv(osp.join(RESULTS_DIR, "go_terms_final.csv"))
    frozen = set(terms["term"])
    if frozen != set(G4.nodes()):
        raise SystemExit(f"rebuilt DAG differs from go_terms_final.csv: {len(frozen ^ set(G4.nodes()))} terms")
    strata = compute_strata(G4)
    mism = sum(1 for t, s in zip(terms["term"], terms["stratum"]) if strata[t] != s)
    if mism:
        raise SystemExit(f"rebuilt strata differ from go_terms_final.csv on {mism} terms")
    if G4.number_of_edges() != terms["n_parents"].sum():
        raise SystemExit("rebuilt edge count differs from go_terms_final.csv")
    freeze_structure(G4, gene_set)
    print(f"froze {G4.number_of_edges()} edges and {len(annotation_pairs(G4, gene_set))} annotation rows (GO {release})")
    pick_example_triple(pd.read_csv(osp.join(RESULTS_DIR, "go_genes_final.csv")))


def build_and_measure() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    G0, gene_set, release = load_raw_go()

    # The pipeline, in the order scripts/dcell.py applies it.
    rows = [stage_row("raw", G0, gene_set)]
    G1 = filter_by_date(G0, DATE_FILTER) if DATE_FILTER else G0
    if DATE_FILTER:
        rows.append(stage_row(f"date<={DATE_FILTER}", G1, gene_set))
    G2 = filter_go_IGI(G1)
    rows.append(stage_row("drop IGI annotations", G2, gene_set))
    G3 = filter_redundant_terms(G2)
    rows.append(stage_row("drop redundant terms", G3, gene_set))
    G4 = filter_by_contained_genes(G3, n=MIN_GENES, gene_set=gene_set)
    rows.append(stage_row(f"contained genes >= {MIN_GENES}", G4, gene_set))
    # Reference row: the experiment-005 exploration's date cutoff, same downstream filters.
    Gd = filter_by_contained_genes(
        filter_redundant_terms(filter_go_IGI(filter_by_date(G0, REFERENCE_DATE_CUTOFF))),
        n=MIN_GENES,
        gene_set=gene_set,
    )
    rows.append(stage_row(f"reference: date<={REFERENCE_DATE_CUTOFF} then same filters", Gd, gene_set))
    stages = pd.DataFrame(rows)
    stages.insert(0, "go_release", release)
    stages.to_csv(osp.join(RESULTS_DIR, "go_filter_stages.csv"), index=False)
    print(stages.to_string())

    # Final DAG: strata (the model's processing order), widths, containment.
    strata = compute_strata(G4)
    direct = {t: len([g for g in (G4.nodes[t].get("gene_set") or []) if g in gene_set]) for t in G4}
    contained = contained_genes(G4, gene_set)
    sizes, width = implied_parameters(G4, direct)
    terms = pd.DataFrame(
        {
            "term": list(G4.nodes()),
            "name": [G4.nodes[t].get("name") for t in G4],
            "namespace": [G4.nodes[t].get("namespace") for t in G4],
            "go_level": [G4.nodes[t].get("level") for t in G4],
            "stratum": [strata[t] for t in G4],
            "direct_genes": [direct[t] for t in G4],
            "contained_genes": [contained[t] for t in G4],
            "n_children": [G4.in_degree(t) for t in G4],
            "n_parents": [G4.out_degree(t) for t in G4],
            "width": [width[t] for t in G4],
        }
    ).sort_values(["stratum", "term"])
    terms.to_csv(osp.join(RESULTS_DIR, "go_terms_final.csv"), index=False)
    strata_df = terms.groupby("stratum").size().rename("terms").reset_index()
    strata_df["direct_genes_sum"] = terms.groupby("stratum")["direct_genes"].sum().values
    strata_df.to_csv(osp.join(RESULTS_DIR, "go_strata.csv"), index=False)

    per_gene = Counter(g for _, g in annotation_pairs(G4, gene_set))
    genes = pd.DataFrame({"gene": sorted(gene_set)})
    genes["n_terms"] = [per_gene.get(g, 0) for g in genes["gene"]]
    genes.to_csv(osp.join(RESULTS_DIR, "go_genes_final.csv"), index=False)
    freeze_structure(G4, gene_set)
    pick_example_triple(genes)

    ev = Counter()
    dates = []
    gene_codes: dict[str, set[str]] = {}
    for t, d in G4.nodes(data=True):
        for g, rec in (d.get("genes") or {}).items():
            if g not in gene_set:
                continue
            code = rec["go_details"]["experiment"]["display_name"]
            ev[code] += 1
            dates.append(rec["go_details"]["date_created"])
            gene_codes.setdefault(g, set()).add(code)
    # Genes whose only retained annotations are ND ("no biological data"), i.e. genes held in
    # the DAG solely by the unknown-function annotations on the three namespace roots.
    genes_only_nd = sum(1 for g, codes in gene_codes.items() if codes == {"ND"})
    nd_on_roots = sum(
        1
        for t in ("GO:0008150", "GO:0003674", "GO:0005575")
        for g, rec in (G4.nodes[t].get("genes") or {}).items()
        if g in gene_set and rec["go_details"]["experiment"]["display_name"] == "ND"
    )
    pd.DataFrame(sorted(ev.items(), key=lambda kv: -kv[1]), columns=["evidence_code", "annotations"]).to_csv(
        osp.join(RESULTS_DIR, "go_evidence_codes.csv"), index=False
    )

    wandb_df = pull_wandb()
    logged = wandb_df.dropna(subset=["model/params_total"])
    logged = logged[logged["model/params_total"] > 0]
    size = {
        "go_release": release,
        "subsystems": G4.number_of_nodes(),
        "hierarchy_edges": G4.number_of_edges(),
        "strata": int(terms["stratum"].max()) + 1,
        "leaves": int((terms["n_children"] == 0).sum()),
        "annotations": int(terms["direct_genes"].sum()),
        "genes_covered": int((genes["n_terms"] > 0).sum()),
        "genes_uncovered": int((genes["n_terms"] == 0).sum()),
        "annotation_date_min": min(dates),
        "annotation_date_max": max(dates),
        "annotations_nd": ev.get("ND", 0),
        "annotations_nd_on_namespace_roots": nd_on_roots,
        "genes_only_nd": genes_only_nd,
        "direct_genes_median": float(terms["direct_genes"].median()),
        "direct_genes_max": int(terms["direct_genes"].max()),
        "contained_genes_median": float(terms["contained_genes"].median()),
        "terms_per_gene_median": float(genes.loc[genes["n_terms"] > 0, "n_terms"].median()),
        **sizes,
        "wandb_num_subsystems": int(logged["model/num_subsystems"].iloc[0]) if len(logged) else None,
        "wandb_params_total": int(logged["model/params_total"].iloc[0]) if len(logged) else None,
        "wandb_params_subsystems": int(logged["model/params_subsystems"].iloc[0]) if len(logged) else None,
        "wandb_params_dcell_linear": int(logged["model/params_dcell_linear"].iloc[0]) if len(logged) else None,
        "wandb_runs_agreeing": int(logged["model/params_total"].nunique() == 1) if len(logged) else None,
        "wandb_n_runs": len(logged),
    }
    pd.DataFrame([size]).to_csv(osp.join(RESULTS_DIR, "dcell_model_size.csv"), index=False)
    print(pd.Series(size).to_string())


def pull_wandb() -> pd.DataFrame:
    """Freeze the ``model/*`` summary fields of every DCell run to CSV."""
    import wandb

    api = wandb.Api()
    rows = []
    for project in WANDB_PROJECTS:
        for run in api.runs(f"{WANDB_ENTITY}/{project}"):
            s = run.summary
            rows.append(
                {
                    "project": project,
                    "run_id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    **{k: s.get(k) for k in WANDB_FIELDS},
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(osp.join(RESULTS_DIR, "dcell_wandb_model_size.csv"), index=False)
    return df


# --------------------------------------------------------------------------- panels
def new_panel():
    fig, ax = plt.subplots(figsize=(mm_to_in(PANEL_W_MM), mm_to_in(PANEL_H_MM)))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.2, top=0.95)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, pad=1.5)
    return fig, ax


def save(fig, name: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    savefig_true_size_svg(fig, osp.join(IMG_DIR, f"{name}.svg"))
    fig.savefig(osp.join(IMG_DIR, f"{name}.png"), dpi=300)
    plt.close(fig)


def pick_example_triple(genes: pd.DataFrame) -> pd.DataFrame:
    """The strain drawn in the DAG panel: a real Kuzmin 2018 trigenic record, chosen by rule.

    Walk the records of the local experiment-005 build (the Kuzmin 2018-only LMDB, keys
    ``"0"`` .. ``"n-1"`` in ascending order) and take the FIRST record whose three
    perturbations are all deletions (no allele or temperature-sensitive allele, so every
    gene is a nuclear deletion target) and whose three genes each carry between 5 and 12
    direct annotations in the final DAG (typical genes, neither the ND-only tail nor the
    hubs). The record index, genes, annotation counts, and measured interaction are frozen
    to ``example_triple.csv`` so ``--from-csv`` renders without the LMDB.
    """
    import lmdb

    n_terms = dict(zip(genes["gene"], genes["n_terms"]))
    lo, hi = TRIPLE_ANNOTATIONS
    env = lmdb.open(osp.join(DATA_ROOT, KUZMIN2018_LMDB), readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n_records = txn.stat()["entries"]
        for i in range(n_records):
            rec = json.loads(txn.get(str(i).encode()))[0]["experiment"]
            perts = rec["genotype"]["perturbations"]
            if len(perts) != 3 or any(p["perturbation_type"] != "deletion" for p in perts):
                continue
            gs = [p["systematic_gene_name"] for p in perts]
            if all(lo <= n_terms.get(g, 0) <= hi for g in gs):
                break
        else:
            raise SystemExit("no all-deletion triple with the required annotation counts")
    env.close()
    df = pd.DataFrame(
        {
            "record_index": i,
            "n_records": n_records,
            "dataset_name": rec["dataset_name"],
            "gene": gs,
            "perturbed_gene_name": [p["perturbed_gene_name"] for p in perts],
            "perturbation_type": [p["perturbation_type"] for p in perts],
            "n_terms": [n_terms[g] for g in gs],
            "gene_interaction": rec["phenotype"]["gene_interaction"],
            "gene_interaction_p_value": rec["phenotype"]["gene_interaction_p_value"],
            "rule": f"first record (ascending LMDB key) with three deletions, each gene {lo}-{hi} direct annotations",
        }
    )
    df.to_csv(osp.join(RESULTS_DIR, "example_triple.csv"), index=False)
    print(f"example triple: record {i} of {n_records}: {gs} (annotations {df['n_terms'].tolist()})")
    return df


def layout_dag(terms: pd.DataFrame, edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Layered positions in [0, 1] x {stratum}: x by parent barycenter, strata as rows.

    Strata are longest-path depths from GO:ROOT (``compute_strata``), so every parent of a
    term sits in a shallower stratum and a single top-down pass fixes each term's x from
    the mean x of its parents. Large strata are spread evenly in barycenter order (which
    keeps each namespace's subtree together); small ones keep their barycenter x, pushed
    apart only enough not to overlap.
    """
    parents: dict[str, list[str]] = {}
    for c, p in zip(edges["child"], edges["parent"]):
        parents.setdefault(c, []).append(p)
    pos: dict[str, tuple[float, float]] = {}
    ns_rank = {"super_root": 0, "biological_process": 1, "molecular_function": 2, "cellular_component": 3}
    for s, grp in terms.groupby("stratum", sort=True):
        names = list(grp["term"])
        if s == 0:
            pos[names[0]] = (0.5, 0)
            continue
        if s == 1:
            # The three namespace roots, in a fixed order across the width.
            ordered = sorted(names, key=lambda t: ns_rank[grp.set_index("term").loc[t, "namespace"]])
            for i, t in enumerate(ordered):
                pos[t] = ((i + 0.5) / len(ordered), s)
            continue
        bary = {t: float(np.mean([pos[p][0] for p in parents[t]])) for t in names}
        ordered = sorted(names, key=lambda t: (bary[t], t))
        n = len(ordered)
        if n >= 40:
            for i, t in enumerate(ordered):
                pos[t] = ((i + 0.5) / n, s)
        else:
            xs = np.array([bary[t] for t in ordered])
            gap = 0.012
            for i in range(1, n):  # push right neighbours apart to the minimum gap
                xs[i] = max(xs[i], xs[i - 1] + gap)
            xs = np.clip(xs - max(0.0, xs[-1] - (1 - gap / 2)), gap / 2, 1 - gap / 2)
            for t, x in zip(ordered, xs):
                pos[t] = (float(x), s)
    return pos


def panel_go_dag(terms: pd.DataFrame, edges: pd.DataFrame, ann: pd.DataFrame, genes: pd.DataFrame, triple: list[str]) -> None:
    """The whole filtered DAG with one real triple deletion propagating to the root."""
    pos = layout_dag(terms, edges)
    ns = dict(zip(terms["term"], terms["namespace"]))
    n_strata = int(terms["stratum"].max()) + 1
    gene_index = {g: i for i, g in enumerate(sorted(genes["gene"]))}
    if any(g not in gene_index for g in triple):
        raise SystemExit(f"example triple {triple} is not in the 6,607-gene reference")
    hit_terms = sorted(set(ann.loc[ann["gene"].isin(triple), "term"]))
    # Every hierarchy edge on a path from a hit term up to the root: what the zeroed rows
    # touch, since each subsystem feeds all of its parents.
    parents: dict[str, list[str]] = {}
    for c, p in zip(edges["child"], edges["parent"]):
        parents.setdefault(c, []).append(p)
    hot_edges: set[tuple[str, str]] = set()
    stack = list(hit_terms)
    seen: set[str] = set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        for p in parents.get(t, []):
            hot_edges.add((t, p))
            stack.append(p)

    fig, ax = plt.subplots(figsize=(mm_to_in(DAG_W_MM), mm_to_in(DAG_H_MM)))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.135, top=DAG_TOP)
    gene_row = n_strata + 0.9  # the strain row sits one layer below the deepest stratum

    def xy(t: str) -> tuple[float, float]:
        x, s = pos[t]
        return x, s

    cold = [(xy(c), xy(p)) for c, p in zip(edges["child"], edges["parent"]) if (c, p) not in hot_edges]
    ax.add_collection(LineCollection(cold, colors="#C8C8C8", linewidths=0.12, zorder=1))
    hot = [(xy(c), xy(p)) for c, p in hot_edges]
    ax.add_collection(LineCollection(hot, colors=HIGHLIGHT, linewidths=0.45, alpha=0.85, zorder=3))
    # Gene states of the deleted genes entering the subsystems that annotate them (blue,
    # dashed) so they read apart from the red solid paths to the root.
    feed = [((gene_index[g] / len(gene_index), gene_row), xy(t)) for t, g in zip(ann["term"], ann["gene"]) if g in triple]
    ax.add_collection(LineCollection(feed, colors=FEED, linewidths=0.4, linestyles=(0, (2, 1.5)), zorder=3))

    for space, color in NAMESPACE_COLOR.items():
        sel = [t for t in terms["term"] if ns[t] == space and t not in hit_terms]
        if not sel:
            continue
        xs, ys = zip(*(xy(t) for t in sel))
        ax.scatter(xs, ys, s=1.6 if space != "super_root" else 9, color=color, linewidths=0, zorder=2)
    xs, ys = zip(*(xy(t) for t in hit_terms))
    ax.scatter(xs, ys, s=7, facecolor="white", edgecolor=HIGHLIGHT, linewidths=0.5, zorder=4)

    # The strain row: 6,607 gene states, present in gray, the deleted three in red.
    ax.plot([0, 1], [gene_row, gene_row], color="#9A9A9A", lw=2.2, solid_capstyle="butt", zorder=2)
    # Gene labels "<gene> = 0" under their ticks; a label is about LABEL_W of the axis wide,
    # so neighbors closer than that are pushed to opposite sides of their ticks.
    LABEL_W = 0.115
    xs_genes = sorted((gene_index[g] / len(gene_index), g) for g in triple)
    prev_right = -1.0
    for i, (x, g) in enumerate(xs_genes):
        ax.plot([x, x], [gene_row - 0.32, gene_row + 0.32], color=HIGHLIGHT, lw=0.9, zorder=4)
        next_close = i + 1 < len(xs_genes) and xs_genes[i + 1][0] - x < LABEL_W
        if next_close and x - LABEL_W - 0.006 > prev_right:
            ha, x_text = "right", x - 0.006
        elif x - LABEL_W / 2 > prev_right and x + LABEL_W / 2 < 1:
            ha, x_text = "center", x
        else:
            ha, x_text = "left", x + 0.006
        ax.text(x_text, gene_row + 0.42, f"{g} = 0", ha=ha, va="top", color=HIGHLIGHT, fontsize=6)
        prev_right = {"right": x - 0.006, "center": x + LABEL_W / 2, "left": x + 0.006 + LABEL_W}[ha]
    ax.text(0.5, gene_row + 1.0, "gene-state vector s over the 6,607 genes: present = 1, deleted = 0",
            ha="center", va="top", fontsize=6)
    # Namespace and root labels sit ABOVE their node on a white box, clear of every edge:
    # the MF root is directly under GO:ROOT, so its label steps right of that vertical edge.
    label_box = dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none")
    for t, name in zip(NAMESPACE_ROOTS, ("BP", "MF", "CC")):
        x, s = pos[t]
        dx, ha = (0.012, "left") if name == "MF" else (0.0, "center")
        ax.text(x + dx, s - 0.42, name, ha=ha, va="center", fontsize=6, bbox=label_box, zorder=6)
    ax.text(0.5, -0.42, "GO:ROOT", ha="center", va="center", fontsize=6, bbox=label_box, zorder=6)

    ax.set_xlim(-0.005, 1.005)
    ax.set_ylim(gene_row + 1.8, -0.7)  # root at the top, strain row at the bottom
    ax.set_yticks(list(range(n_strata)) + [gene_row])
    ax.set_yticklabels([str(s) for s in range(n_strata)] + ["s"])
    ax.set_ylabel("Stratum")
    ax.set_xticks([])
    ax.tick_params(axis="y", width=0.5, length=2, pad=1.5)
    handles = [
        Line2D([], [], marker="o", ls="", ms=3, color=NAMESPACE_COLOR[k], label=NAMESPACE_LABEL[k])
        for k in ("biological_process", "molecular_function", "cellular_component")
    ] + [
        Line2D([], [], marker="o", ls="", ms=3, markerfacecolor="white", markeredgecolor=HIGHLIGHT, label="Annotated to a deleted gene"),
        Line2D([], [], color=HIGHLIGHT, lw=0.8, label="Path to GO:ROOT"),
        Line2D([], [], color=FEED, lw=0.8, ls=(0, (2, 1.5)), label="Gene state into subsystem"),
    ]
    legend_gap = 0.005  # legend hangs this fraction of the axes height below the frame
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -legend_gap), frameon=False, ncol=3,
                    handlelength=1.4, handletextpad=0.5, borderpad=0.0, labelspacing=0.25, columnspacing=1.2)
    ax.text(0.995, 0.985, f"{len(terms):,} subsystems, {len(edges):,} edges, {n_strata} strata",
            transform=ax.transAxes, ha="right", va="top", fontsize=6)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    # Flush bottom: measure the legend and set the bottom margin so its lower edge sits
    # DAG_BOTTOM_CLEAR above the image's bottom edge (the figure is composed edge to edge).
    fig.canvas.draw()
    h_leg = leg.get_window_extent().transformed(fig.transFigure.inverted()).height
    bottom = h_leg + DAG_BOTTOM_CLEAR + legend_gap * (DAG_TOP - 0.135)
    fig.subplots_adjust(bottom=bottom)
    save(fig, "dcell_model_go_dag")


def panel_terms_per_stratum(strata_df: pd.DataFrame) -> None:
    fig, ax = new_panel()
    ax.bar(strata_df["stratum"], strata_df["terms"], color=PLOT_PALETTE[0], edgecolor="black", linewidth=0.4, width=0.8)
    ax.set_xlabel("Stratum (0 = root, longest path from root)")
    ax.set_ylabel("Subsystems")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.7, strata_df["stratum"].max() + 0.7)
    ax.yaxis.grid(True, linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    save(fig, "dcell_model_terms_per_stratum")


def panel_genes_per_term(terms: pd.DataFrame) -> None:
    fig, ax = new_panel()
    hi = max(terms["contained_genes"].max(), terms["direct_genes"].max())
    bins = np.logspace(0, np.log10(hi) + 0.05, 24)
    direct = terms["direct_genes"].clip(lower=1)
    ax.hist(terms["contained_genes"], bins=bins, color=PLOT_PALETTE[1], edgecolor="black", linewidth=0.4, label="Contained (term or descendants)")
    ax.hist(direct, bins=bins, color=PLOT_PALETTE[3], edgecolor="black", linewidth=0.4, label="Direct annotation (sets width)")
    ax.set_xscale("log")
    ax.set_xlabel("Genes per subsystem")
    ax.set_ylabel("Subsystems")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # headroom so the legend clears the mode
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.5)
    ax.yaxis.grid(True, linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    save(fig, "dcell_model_genes_per_term")


def panel_terms_per_gene(genes: pd.DataFrame) -> None:
    fig, ax = new_panel()
    hi = genes["n_terms"].max()
    bins = np.arange(-0.5, hi + 1.5, 1.0) if hi <= 60 else np.linspace(-0.5, hi + 0.5, 61)
    ax.hist(genes["n_terms"], bins=bins, color=PLOT_PALETTE[2], edgecolor="black", linewidth=0.4)
    n0 = int((genes["n_terms"] == 0).sum())
    ax.set_xlabel("Subsystems per gene (direct annotations)")
    ax.set_ylabel("Genes")
    ax.text(
        0.97, 0.93,
        f"{len(genes) - n0:,} of {len(genes):,} genes covered\n"
        f"min {int(genes['n_terms'].min())}, median {int(genes['n_terms'].median())}, max {int(genes['n_terms'].max())}",
        transform=ax.transAxes, ha="right", va="top",
    )
    ax.yaxis.grid(True, linewidth=0.3, color="0.85")
    ax.set_axisbelow(True)
    save(fig, "dcell_model_terms_per_gene")


def write_table(stages: pd.DataFrame) -> None:
    label = {
        "raw": r"SGD annotations on GO, three namespaces under \texttt{GO:ROOT}",
        "drop IGI annotations": "drop IGI-evidence annotations (empty terms removed)",
        "drop redundant terms": "drop terms whose gene set equals a parent's",
        f"contained genes >= {MIN_GENES}": rf"drop terms containing $<{MIN_GENES}$ genes (the trigenic run)",
        f"reference: date<={REFERENCE_DATE_CUTOFF} then same filters": rf"reference only: annotations dated $\le$ {REFERENCE_DATE_CUTOFF}, then the same three filters",
    }
    lines = [
        "%% AUTO-GENERATED -- do not hand-edit.",
        "%% SOURCE: experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py",
        "%%         reads experiments/006-kuzmin-tmi/results/dcell_model/go_filter_stages.csv",
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Stage & Terms & Edges & Annotations & Genes covered & Leaves \\",
        r"\midrule",
    ]
    for _, r in stages.iterrows():
        lines.append(
            f"{label[r['stage']]} & {r['terms']:,} & {r['edges']:,} & {r['annotations']:,} & {r['genes_covered']:,} & {r['leaves']:,} \\\\"
        )
    release = stages["go_release"].iloc[0].replace("releases/", "")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{The GO DAG at each filtering stage of the DCell baseline. Filters are applied in the order listed"
        r" (\texttt{torchcell.graph.filter\_go\_IGI}, \texttt{filter\_redundant\_terms}, \texttt{filter\_by\_contained\_genes}),"
        f" on GO release {release} with SGD gene annotations; a removed term's children are reconnected to its parents."
        r" Annotations are direct (term, gene) pairs over the 6,607-gene reference; genes covered are those with at least one such pair;"
        r" leaves are terms with no child term. The last row is not part of the trigenic run.}",
        r"\label{tab:dcell-go-filter}",
        r"\end{table}",
    ]
    with open(osp.join(TEX_DIR, "tab-dcell-model-go-filter.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def torchcell_values(stages: pd.DataFrame, terms: pd.DataFrame, strata_df: pd.DataFrame, genes: pd.DataFrame,
                     edges: pd.DataFrame, size: pd.Series) -> dict[str, str]:
    """The TorchCell column of the comparison, every value read off the frozen DAG."""
    st = stages.set_index("stage")
    raw, igi, red, fin = st.loc["raw"], st.loc["drop IGI annotations"], st.loc["drop redundant terms"], st.loc[f"contained genes >= {MIN_GENES}"]
    root = terms.set_index("term").loc["GO:ROOT"]
    n_strata = int(strata_df["stratum"].max()) + 1
    if len(edges) != int(fin["edges"]) or len(terms) != int(fin["terms"]):
        raise SystemExit("go_edges_final.csv / go_terms_final.csv disagree with go_filter_stages.csv")
    if int(terms["width"].sum()) != int(size["neurons"]):
        raise SystemExit("hidden units in go_terms_final.csv disagree with dcell_model_size.csv")
    release = raw["go_release"].replace("releases/", "")
    return {
        "GO release": f"{release} (\\texttt{{go.obo}} data-version)",
        "Annotation source": "SGD \\texttt{go\\_details} over the 6,607-gene S288C reference",
        "Genes": f"{len(genes):,} reference genes, all covered",
        "Evidence filter": f"IGI annotations removed ({int(raw['annotations'] - igi['annotations']):,}); a term goes only when emptied ({int(raw['terms'] - igi['terms'])} terms)",
        "Redundancy filter": f"terms whose gene set equals a parent's ({int(igi['terms'] - red['terms'])} terms)",
        "Containment threshold": f"fewer than {MIN_GENES_WORD} of the 6,607 reference genes, counted over the term and its descendants ({int(red['terms'] - fin['terms']):,} terms)",
        "Filter order": "IGI, redundancy, containment",
        "Removed-term rewiring": "children connected to all parents",
        "Subsystems": f"{int(fin['terms']):,}",
        "Hierarchy edges": f"{int(fin['edges']):,}",
        "Depth": f"{n_strata} strata (\\texttt{{GO:ROOT}} plus {n_strata - 1})",
        "Leaves": f"{int(fin['leaves']):,}",
        "Width rule": "$\\max(20,\\lceil 0.3\\,\\lvert\\mathrm{genes}(t)\\rvert\\rceil)$, direct annotations",
        "Root readout": f"one output; \\texttt{{GO:ROOT}} has {int(root['direct_genes'])} direct genes, so $L_r={int(root['width'])}$",
        "Hidden units": f"{int(size['neurons']):,} ({int(size['width_min'])} to {int(size['width_max']):,} per subsystem; {int((terms['width'] == SUBSYSTEM_MIN).sum()):,} subsystems at the floor of {SUBSYSTEM_MIN})",
        "Parameters": f"{int(size['params_total']):,}",
        "Auxiliary loss": "$\\alpha=0.3$, averaged over $t\\neq r$; AdamW weight decay",
        "Training data": "the trigenic dataset of the CGT experiment (\\suppnoteref{note:dcell-training})",
    }


def write_vs_paper_table(stages: pd.DataFrame, terms: pd.DataFrame, strata_df: pd.DataFrame, genes: pd.DataFrame,
                         edges: pd.DataFrame, size: pd.Series) -> None:
    """Published DCell (verbatim from the paper) against the TorchCell rebuild, CSV + LaTeX."""
    tc = torchcell_values(stages, terms, strata_df, genes, edges, size)
    if set(tc) != {p.row for p in PUBLISHED}:
        raise SystemExit(f"row mismatch between PUBLISHED and torchcell_values: {set(tc) ^ {p.row for p in PUBLISHED}}")
    rows = [
        {"row": p.row, "published": p.published, "torchcell": tc[p.row], "quote": p.quote or "", "where": p.where,
         "citation_key": PAPER_KEY, "paper_md_sha256": PAPER_MD_SHA256}
        for p in PUBLISHED
    ]
    pd.DataFrame(rows).to_csv(osp.join(RESULTS_DIR, "dcell_vs_paper.csv"), index=False)
    lines = [
        "%% AUTO-GENERATED -- do not hand-edit.",
        "%% SOURCE: experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py",
        "%%         published column: verbatim quotes from the OCR text of the mirror key",
        f"%%         {PAPER_KEY} (paper.md, sha256 {PAPER_MD_SHA256}), listed in",
        "%%         results/dcell_model/dcell_vs_paper.csv; TorchCell column: go_filter_stages.csv,",
        "%%         go_terms_final.csv, go_strata.csv, go_genes_final.csv, go_edges_final.csv, dcell_model_size.csv.",
    ]
    for p in PUBLISHED:
        lines.append(f"%% {p.row}: " + (f"[{p.where}] {p.quote}" if p.quote else "not stated in the paper"))
    lines += [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\begin{tabular}{@{}p{0.17\linewidth}p{0.38\linewidth}p{0.39\linewidth}@{}}",
        r"\toprule",
        r"& \textbf{Published DCell} & \textbf{TorchCell DCell} \\",
        r"\midrule",
    ]
    for p in PUBLISHED:
        lines.append(f"{p.row} & {p.published} & {tc[p.row]} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{The published DCell ontology and model (Ma et al.\ 2018) against the TorchCell rebuild."
        r" Published values are quoted from the paper's text and Online Methods (the OCR text of the mirrored"
        r" paper, sha256-pinned in the generating script); ``not reported'' marks a quantity the paper does not"
        r" state. TorchCell values are measured on the frozen DAG of \supptab{tab:dcell-go-filter}; counts in"
        r" parentheses are the terms or annotations each filter removed. Depth counts strata of the longest path"
        r" from \texttt{GO:ROOT}; the paper's ``12 layers'' does not say whether the root is counted."
        r" Hidden units are the sum of subsystem widths $L_t$.}",
        r"\label{tab:dcell-vs-paper}",
        r"\end{table}",
    ]
    with open(osp.join(TEX_DIR, "tab-dcell-model-vs-paper.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def render() -> None:
    stages = pd.read_csv(osp.join(RESULTS_DIR, "go_filter_stages.csv"))
    terms = pd.read_csv(osp.join(RESULTS_DIR, "go_terms_final.csv"))
    strata_df = pd.read_csv(osp.join(RESULTS_DIR, "go_strata.csv"))
    genes = pd.read_csv(osp.join(RESULTS_DIR, "go_genes_final.csv"))
    edges = pd.read_csv(osp.join(RESULTS_DIR, "go_edges_final.csv"))
    ann = pd.read_csv(osp.join(RESULTS_DIR, "go_annotations_final.csv"))
    ex = pd.read_csv(osp.join(RESULTS_DIR, "example_triple.csv"))
    size = pd.read_csv(osp.join(RESULTS_DIR, "dcell_model_size.csv")).iloc[0]
    triple = list(ex["gene"])
    panel_go_dag(terms, edges, ann, genes, triple)
    print(f"highlighted triple deletion: record {int(ex['record_index'].iloc[0])}: {triple}")
    panel_terms_per_stratum(strata_df)
    panel_genes_per_term(terms)
    panel_terms_per_gene(genes)
    write_table(stages)
    write_vs_paper_table(stages, terms, strata_df, genes, edges, size)
    print(f"panels -> {IMG_DIR}/dcell_model_*.svg; tables -> {TEX_DIR}/tab-dcell-model-go-filter.tex, tab-dcell-model-vs-paper.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--from-csv", action="store_true", help="re-render panels and table from the frozen CSVs")
    g.add_argument("--dag-only", action="store_true",
                   help="rebuild the DAG (no wandb), verify it against go_terms_final.csv, freeze edges + annotations, render")
    g.add_argument("--pick-triple", action="store_true",
                   help="re-pick the example triple from the local 005 LMDB, freeze example_triple.csv, render")
    args = ap.parse_args()
    if args.dag_only:
        build_dag_only()
    elif args.pick_triple:
        pick_example_triple(pd.read_csv(osp.join(RESULTS_DIR, "go_genes_final.csv")))
    elif not args.from_csv:
        build_and_measure()
    render()


if __name__ == "__main__":
    main()
