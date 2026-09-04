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
  go_evidence_codes.csv       annotation evidence codes retained in the final DAG
  dcell_model_size.csv        parameter count implied by the widths, vs the wandb-logged
                              ``model/params_*`` of the trigenic runs (frozen pull)
  dcell_wandb_model_size.csv  the frozen wandb pull (one row per run)

Panels (true-size SVG + PNG under $ASSET_IMAGES_DIR/006-kuzmin-tmi/):
  dcell_model_terms_per_stratum, dcell_model_genes_per_term, dcell_model_terms_per_gene

Table: paper/nature-biotech/sections/tab-dcell-model-go-filter.tex

Run from the repo root:
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py
    python experiments/006-kuzmin-tmi/scripts/dcell_model_go_stats.py --from-csv
"""

import argparse
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
from matplotlib.ticker import MaxNLocator

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


def build_and_measure() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
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
    G0 = graph.G_go.copy()
    release = go_release(osp.join(DATA_ROOT, "data/go"))
    print(f"go.obo release {release}; reference genes {len(gene_set)}")

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


def render() -> None:
    stages = pd.read_csv(osp.join(RESULTS_DIR, "go_filter_stages.csv"))
    terms = pd.read_csv(osp.join(RESULTS_DIR, "go_terms_final.csv"))
    strata_df = pd.read_csv(osp.join(RESULTS_DIR, "go_strata.csv"))
    genes = pd.read_csv(osp.join(RESULTS_DIR, "go_genes_final.csv"))
    panel_terms_per_stratum(strata_df)
    panel_genes_per_term(terms)
    panel_terms_per_gene(genes)
    write_table(stages)
    print(f"panels -> {IMG_DIR}/dcell_model_*.svg; table -> {TEX_DIR}/tab-dcell-model-go-filter.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-csv", action="store_true", help="re-render panels and table from the frozen CSVs")
    args = ap.parse_args()
    if not args.from_csv:
        build_and_measure()
    render()


if __name__ == "__main__":
    main()
