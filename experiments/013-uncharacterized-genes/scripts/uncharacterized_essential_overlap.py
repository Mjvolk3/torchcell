# experiments/013-uncharacterized-genes/scripts/uncharacterized_essential_overlap.py
# [[experiments.013-uncharacterized-genes.scripts.uncharacterized_essential_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/013-uncharacterized-genes/scripts/uncharacterized_essential_overlap.py

import json
import os
import os.path as osp
from typing import Dict, Set

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from upsetplot import UpSet, from_contents

from torchcell.graph import SCerevisiaeGraph
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

if DATA_ROOT is None:
    raise ValueError("DATA_ROOT environment variable not set")
if EXPERIMENT_ROOT is None:
    raise ValueError("EXPERIMENT_ROOT environment variable not set")
if ASSET_IMAGES_DIR is None:
    raise ValueError("ASSET_IMAGES_DIR environment variable not set")

# Custom colors from torchcell.mplstyle
CUSTOM_BLUE = "#7191A9"
CUSTOM_GREEN = "#6B8D3A"
CUSTOM_RED = "#A97171"
CUSTOM_ORANGE = "#D4A373"


def get_essential_genes_from_graph(graph: SCerevisiaeGraph) -> Set[str]:
    """
    Extract essential genes from graph based on inviable phenotype.

    This uses the same logic as GeneEssentialitySgdDataset in
    torchcell/datasets/scerevisiae/sgd.py

    Args:
        graph: SCerevisiaeGraph instance

    Returns:
        Set of essential gene IDs
    """
    essential_genes = set()

    print("Extracting essential genes from graph...")
    for gene in tqdm(graph.G_raw.nodes(), desc="Processing genes"):
        node_data = graph.G_raw.nodes[gene]

        # Check for inviable phenotypes (same logic as sgd.py:171-179)
        inviable_phenotypes = [
            i
            for i in node_data.get("phenotype_details", [])
            if (
                i["mutant_type"] == "null"
                and i["strain"]["display_name"] == "S288C"
                and i["phenotype"]["display_name"] == "inviable"
            )
        ]

        if inviable_phenotypes:
            essential_genes.add(gene)

    print(f"  Found {len(essential_genes)} essential genes")
    return essential_genes


def get_uncharacterized_from_genome(genome: SCerevisiaeGenome) -> Set[str]:
    """
    Extract uncharacterized genes from genome using orf_classification field.

    Args:
        genome: SCerevisiaeGenome instance

    Returns:
        Set of uncharacterized gene IDs from genome
    """
    uncharacterized_genes = set()

    print("Extracting uncharacterized genes from genome...")
    for gene_id in tqdm(genome.gene_set, desc="Processing genes"):
        gene = genome[gene_id]

        if gene is None:
            continue

        # Check orf_classification for "Uncharacterized"
        if gene.orf_classification and any(
            "Uncharacterized" in classification
            for classification in gene.orf_classification
        ):
            uncharacterized_genes.add(gene_id)

    print(f"  Found {len(uncharacterized_genes)} uncharacterized genes")
    return uncharacterized_genes


def get_uncharacterized_from_graph(graph: SCerevisiaeGraph) -> Set[str]:
    """
    Extract uncharacterized genes from graph using qualifier field.

    Args:
        graph: SCerevisiaeGraph instance

    Returns:
        Set of uncharacterized gene IDs from graph
    """
    uncharacterized_genes = set()

    print("Extracting uncharacterized genes from graph...")
    for gene in tqdm(graph.G_raw.nodes(), desc="Processing genes"):
        node_data = graph.G_raw.nodes[gene]

        # Check locus qualifier for "Uncharacterized"
        locus_data = node_data.get("locus", {})
        qualifier = locus_data.get("qualifier", "")

        if qualifier == "Uncharacterized":
            uncharacterized_genes.add(gene)

    print(f"  Found {len(uncharacterized_genes)} uncharacterized genes")
    return uncharacterized_genes


def create_upset_plot(
    gene_sets: Dict[str, Set[str]],
    output_path: str,
    title: str = "Gene Set Overlaps"
):
    """
    Create an UpSet plot showing overlaps between gene sets.

    Args:
        gene_sets: Dictionary mapping set names to gene ID sets
        output_path: Path to save the plot
        title: Title for the plot
    """
    # Create UpSet data from sets
    upset_data = from_contents(gene_sets)

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    upset = UpSet(
        upset_data,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        element_size=40,
    )
    upset.plot(fig=fig)

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved UpSet plot: {output_path}")


def analyze_agreement(
    genome_unchar: Set[str],
    graph_unchar: Set[str],
    essential: Set[str]
) -> tuple[Dict, Dict[str, Set[str]]]:
    """
    Analyze agreement between different data sources.

    Args:
        genome_unchar: Uncharacterized genes from genome
        graph_unchar: Uncharacterized genes from graph
        essential: Essential genes from graph

    Returns:
        Tuple of (analysis_dict, gene_sets_dict)
    """
    # Basic set operations
    unchar_and = genome_unchar & graph_unchar  # Both sources agree
    unchar_or = genome_unchar | graph_unchar   # Either source says uncharacterized

    # Overlaps with essential
    essential_and_genome_unchar = essential & genome_unchar
    essential_and_graph_unchar = essential & graph_unchar
    essential_and_agreed_unchar = essential & unchar_and
    essential_and_any_unchar = essential & unchar_or

    # Only in one source
    only_genome_unchar = genome_unchar - graph_unchar
    only_graph_unchar = graph_unchar - genome_unchar

    analysis = {
        "total_genome_uncharacterized": len(genome_unchar),
        "total_graph_uncharacterized": len(graph_unchar),
        "total_essential": len(essential),
        "both_sources_agree_uncharacterized": len(unchar_and),
        "either_source_uncharacterized": len(unchar_or),
        "only_genome_uncharacterized": len(only_genome_unchar),
        "only_graph_uncharacterized": len(only_graph_unchar),
        "essential_and_genome_uncharacterized": len(essential_and_genome_unchar),
        "essential_and_graph_uncharacterized": len(essential_and_graph_unchar),
        "essential_and_agreed_uncharacterized": len(essential_and_agreed_unchar),
        "essential_and_any_uncharacterized": len(essential_and_any_unchar),
        "agreement_percentage": len(unchar_and) / len(unchar_or) * 100 if len(unchar_or) > 0 else 0,
    }

    return analysis, {
        "unchar_and": unchar_and,
        "unchar_or": unchar_or,
        "only_genome_unchar": only_genome_unchar,
        "only_graph_unchar": only_graph_unchar,
        "essential_and_genome_unchar": essential_and_genome_unchar,
        "essential_and_graph_unchar": essential_and_graph_unchar,
        "essential_and_agreed_unchar": essential_and_agreed_unchar,
        "essential_and_any_unchar": essential_and_any_unchar,
    }


def print_analysis_summary(analysis: Dict, gene_sets: Dict[str, Set[str]], genome: SCerevisiaeGenome, graph: SCerevisiaeGraph):
    """Print formatted analysis summary with gene details."""
    print("\n" + "=" * 70)
    print("GENE SET ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Data Source Totals ---")
    print(f"Essential genes (graph):                    {analysis['total_essential']:>6,}")
    print(f"Uncharacterized genes (genome):             {analysis['total_genome_uncharacterized']:>6,}")
    print(f"Uncharacterized genes (graph):              {analysis['total_graph_uncharacterized']:>6,}")

    print("\n--- Uncharacterized Source Agreement ---")
    print(f"Both sources agree (∩):                     {analysis['both_sources_agree_uncharacterized']:>6,}")
    print(f"Either source says uncharacterized (∪):     {analysis['either_source_uncharacterized']:>6,}")
    print(f"Only in genome:                             {analysis['only_genome_uncharacterized']:>6,}")
    print(f"Only in graph:                              {analysis['only_graph_uncharacterized']:>6,}")
    print(f"Agreement rate:                             {analysis['agreement_percentage']:>6.2f}%")

    print("\n--- Essential ∩ Uncharacterized Overlaps ---")
    print(f"Essential ∩ Genome-Uncharacterized:         {analysis['essential_and_genome_uncharacterized']:>6,}")
    print(f"Essential ∩ Graph-Uncharacterized:          {analysis['essential_and_graph_uncharacterized']:>6,}")
    print(f"Essential ∩ Agreed-Uncharacterized (∩):     {analysis['essential_and_agreed_uncharacterized']:>6,}")
    print(f"Essential ∩ Any-Uncharacterized (∪):        {analysis['essential_and_any_uncharacterized']:>6,}")

    # Calculate percentages
    if analysis['total_essential'] > 0:
        pct_genome = analysis['essential_and_genome_uncharacterized'] / analysis['total_essential'] * 100
        pct_graph = analysis['essential_and_graph_uncharacterized'] / analysis['total_essential'] * 100
        pct_agreed = analysis['essential_and_agreed_uncharacterized'] / analysis['total_essential'] * 100
        pct_any = analysis['essential_and_any_uncharacterized'] / analysis['total_essential'] * 100

        print("\n--- Percentage of Essential Genes ---")
        print(f"That are genome-uncharacterized:            {pct_genome:>6.2f}%")
        print(f"That are graph-uncharacterized:             {pct_graph:>6.2f}%")
        print(f"That are agreed-uncharacterized:            {pct_agreed:>6.2f}%")
        print(f"That are any-uncharacterized:               {pct_any:>6.2f}%")

    # Print detailed gene information for Essential ∩ Agreed-Uncharacterized
    essential_and_agreed = gene_sets["essential_and_agreed_unchar"]
    if essential_and_agreed:
        print("\n" + "=" * 70)
        print(f"GENES THAT ARE BOTH ESSENTIAL AND UNCHARACTERIZED (n={len(essential_and_agreed)})")
        print("(Both genome and graph agree these are uncharacterized)")
        print("=" * 70)

        for gene_id in sorted(essential_and_agreed):
            print(f"\n{gene_id}:")

            # Get genome data
            gene = genome[gene_id]
            if gene:
                print(f"  Genome Classification: {gene.orf_classification}")
                print(f"  Location: Chr {gene.chromosome}:{gene.start}-{gene.end} ({gene.strand})")
                if gene.note:
                    print(f"  Genome Note: {gene.note[0] if gene.note else 'N/A'}")

            # Get graph data
            if gene_id in graph.G_raw.nodes:
                node_data = graph.G_raw.nodes[gene_id]
                locus_data = node_data.get("locus", {})
                print(f"  Graph Qualifier: {locus_data.get('qualifier', 'N/A')}")

                # Print full description (not truncated)
                description = locus_data.get('description', 'N/A')
                print(f"  Graph Description: {description}")

                # Print inviable phenotypes
                inviable = [
                    p for p in node_data.get("phenotype_details", [])
                    if (p["mutant_type"] == "null"
                        and p["strain"]["display_name"] == "S288C"
                        and p["phenotype"]["display_name"] == "inviable")
                ]
                if inviable:
                    print(f"  Essential evidence: {len(inviable)} inviable phenotype record(s)")

    # Print genes only in genome but not in graph
    only_genome = gene_sets["only_genome_unchar"]
    if only_genome:
        print("\n" + "=" * 70)
        print(f"GENES UNCHARACTERIZED IN GENOME ONLY (n={len(only_genome)})")
        print("(Not marked as uncharacterized in graph)")
        print("=" * 70)

        for gene_id in sorted(only_genome):
            print(f"\n{gene_id}:")

            # Get genome data
            gene = genome[gene_id]
            if gene:
                print(f"  Genome Classification: {gene.orf_classification}")
                print(f"  Location: Chr {gene.chromosome}:{gene.start}-{gene.end} ({gene.strand})")
                if gene.note:
                    print(f"  Genome Note: {gene.note[0] if gene.note else 'N/A'}")

            # Get graph data
            if gene_id in graph.G_raw.nodes:
                node_data = graph.G_raw.nodes[gene_id]
                locus_data = node_data.get("locus", {})
                print(f"  Graph Qualifier: {locus_data.get('qualifier', 'N/A')}")

                # Print full description (not truncated)
                description = locus_data.get('description', 'N/A')
                print(f"  Graph Description: {description}")
            else:
                print(f"  Graph: Gene not found in graph")

    # Print genes only in graph but not in genome
    only_graph = gene_sets["only_graph_unchar"]
    if only_graph:
        print("\n" + "=" * 70)
        print(f"GENES UNCHARACTERIZED IN GRAPH ONLY (n={len(only_graph)})")
        print("(Not marked as uncharacterized in genome)")
        print("=" * 70)

        for gene_id in sorted(only_graph):
            print(f"\n{gene_id}:")

            # Get genome data
            gene = genome[gene_id]
            if gene:
                print(f"  Genome Classification: {gene.orf_classification}")
                print(f"  Location: Chr {gene.chromosome}:{gene.start}-{gene.end} ({gene.strand})")
                if gene.note:
                    print(f"  Genome Note: {gene.note[0] if gene.note else 'N/A'}")
            else:
                print(f"  Genome: Gene not found in genome")

            # Get graph data
            if gene_id in graph.G_raw.nodes:
                node_data = graph.G_raw.nodes[gene_id]
                locus_data = node_data.get("locus", {})
                print(f"  Graph Qualifier: {locus_data.get('qualifier', 'N/A')}")

                # Print full description (not truncated)
                description = locus_data.get('description', 'N/A')
                print(f"  Graph Description: {description}")

    print("=" * 70)


def main():
    """Main analysis function."""
    # Type assertions
    assert DATA_ROOT is not None, "DATA_ROOT must be set"
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"
    assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set"

    print("Initializing genome and graph...")

    # Initialize genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    # Initialize graph
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    print(f"\nGenome gene set size: {len(genome.gene_set)}")
    print(f"Graph node count: {graph.G_raw.number_of_nodes()}")

    # Extract gene sets
    print("\n" + "=" * 70)
    print("EXTRACTING GENE SETS")
    print("=" * 70 + "\n")

    essential_genes = get_essential_genes_from_graph(graph)
    genome_uncharacterized = get_uncharacterized_from_genome(genome)
    graph_uncharacterized = get_uncharacterized_from_graph(graph)

    # Analyze agreement
    analysis, gene_sets = analyze_agreement(
        genome_uncharacterized,
        graph_uncharacterized,
        essential_genes
    )

    # Print summary with gene details
    print_analysis_summary(analysis, gene_sets, genome, graph)

    # Create output directories
    results_dir = osp.join(EXPERIMENT_ROOT, "013-uncharacterized-genes", "results")
    images_dir = osp.join(ASSET_IMAGES_DIR, "013-uncharacterized-genes")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Save analysis results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    analysis_file = osp.join(results_dir, "essential_uncharacterized_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Saved analysis: {analysis_file}")

    # Save gene sets
    gene_sets_file = osp.join(results_dir, "essential_uncharacterized_gene_sets.json")
    gene_sets_serializable = {
        key: sorted(list(value)) for key, value in gene_sets.items()
    }
    with open(gene_sets_file, "w") as f:
        json.dump(gene_sets_serializable, f, indent=2)
    print(f"  Saved gene sets: {gene_sets_file}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    print(f"  Images will be saved to: {images_dir}")

    # Main UpSet plot showing all three gene sets
    upset_path = osp.join(
        images_dir,
        "upset_essential_uncharacterized_all.png"
    )
    create_upset_plot(
        {
            "Essential (Graph)": essential_genes,
            "Unchar (Genome)": genome_uncharacterized,
            "Unchar (Graph)": graph_uncharacterized,
        },
        upset_path,
        title="Essential (inviable) ∩ Uncharacterized (Genome & Graph)"
    )

    # Create a summary dataframe
    summary_df = pd.DataFrame([
        {"Category": "Essential genes", "Count": analysis["total_essential"]},
        {"Category": "Genome uncharacterized", "Count": analysis["total_genome_uncharacterized"]},
        {"Category": "Graph uncharacterized", "Count": analysis["total_graph_uncharacterized"]},
        {"Category": "Both sources agree uncharacterized", "Count": analysis["both_sources_agree_uncharacterized"]},
        {"Category": "Either source uncharacterized", "Count": analysis["either_source_uncharacterized"]},
        {"Category": "Essential ∩ Genome-Unchar", "Count": analysis["essential_and_genome_uncharacterized"]},
        {"Category": "Essential ∩ Graph-Unchar", "Count": analysis["essential_and_graph_uncharacterized"]},
        {"Category": "Essential ∩ Agreed-Unchar", "Count": analysis["essential_and_agreed_uncharacterized"]},
        {"Category": "Essential ∩ Any-Unchar", "Count": analysis["essential_and_any_uncharacterized"]},
    ])

    summary_csv = osp.join(results_dir, "essential_uncharacterized_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Saved summary CSV: {summary_csv}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return analysis, gene_sets


if __name__ == "__main__":
    analysis, gene_sets = main()
