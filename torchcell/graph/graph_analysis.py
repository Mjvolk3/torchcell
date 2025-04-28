import glob
import requests
import gzip
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import Set

import gffutils
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import pandas as pd
from attrs import define, field
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from gffutils import FeatureDB
from gffutils.feature import Feature
from matplotlib.patches import Patch
from sortedcontainers import SortedDict, SortedSet
from torch_geometric.data import download_url
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from torchcell.sequence import GeneSet, Genome, ParsedGenome
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
import torchcell

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def plot_go_graph(G):
    # Define a color map for namespaces.
    namespace_colors = {
        "super_root": "#122A4B",
        "biological_process": "#FF552E",
        "molecular_function": "#8286C2",
        "cellular_component": "#D23943",
    }

    # Identify nodes without 'namespace'
    nodes_missing_namespace = [
        node for node, attrs in G.nodes(data=True) if "namespace" not in attrs
    ]
    print(f"num nodes_missing_namespace: {len(nodes_missing_namespace)}")

    # Generate the color list for nodes based on their namespace
    node_colors = [
        namespace_colors[G.nodes[node].get("namespace", "missing")]
        for node in G.nodes()
    ]

    max_level = max([data["level"] for node, data in G.nodes(data=True)])
    min_level = min([data["level"] for node, data in G.nodes(data=True)])

    # Calculate node positions based on levels
    level_spacing = 10.0
    horizontal_spacing = 20.0

    new_pos = {}
    for node, data in G.nodes(data=True):
        level = data.get("level")
        nodes_in_level = [n for n, d in G.nodes(data=True) if d.get("level") == level]

        # Calculate horizontal position
        idx_in_level = nodes_in_level.index(node)
        total_nodes_in_level = len(nodes_in_level)
        new_x = (idx_in_level - total_nodes_in_level / 2) * horizontal_spacing

        # Adjust vertical position
        new_y = (max_level - level) * level_spacing - min_level * level_spacing

        new_pos[node] = (new_x, new_y)

    # Plotting
    plt.figure(figsize=(20, 10))
    nx.draw(
        G,
        new_pos,
        with_labels=False,
        node_size=20,
        node_color=node_colors,
        alpha=0.6,
        linewidths=0.5,
        width=0.25,
        edge_color="lightgray",
        arrowsize=5,
    )

    # Add legend
    legend_elements = [
        Patch(
            facecolor=namespace_colors[key],
            edgecolor="black",
            label=key.replace("_", " ").title(),
        )
        for key in namespace_colors
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper right",
        title="Namespaces",
        fontsize=28,
        title_fontsize=28,
    )

    # Annotate levels on the left side
    unique_levels = sorted(list({data["level"] for node, data in G.nodes(data=True)}))

    # Determine the minimum x-coordinate and set the margin
    min_x_coord = min([x for x, y in new_pos.values()])
    margin = 120.0 * horizontal_spacing  # Adjust the multiplier as needed

    for level in unique_levels:
        y_pos = (max_level - level) * level_spacing - min_level * level_spacing
        plt.text(
            x=min_x_coord - margin,  # Use the adjusted x-coordinate
            y=y_pos,
            s=f"level: {level}",
            ha="left",
            va="center",
            fontsize=28,
        )

    plt.title("GO DAG with Forwarded Isolated Nodes and Super Node", fontsize=28)
    plt.tight_layout()
    plt.savefig(
        "./notes/assets/images/dcell-gene-ontology-dag-no-isoloated-with-super-node.png",
        dpi=300,
    )
    plt.close()
    # plt.show()


# TODO add a function that will remove genes not in dmf costanzo and smf costanzo



def plot_annotation_dates_by_month(G_go: nx.DiGraph):
    # Function to convert date strings to datetime objects
    def convert_to_datetime(date_str):
        return datetime.strptime(date_str, "%Y-%m-%d")

    # Collect all the annotation dates and group by month and year
    dates = defaultdict(int)
    for go_term in G_go.nodes():
        if "genes" in G_go.nodes[go_term]:
            for gene, details in G_go.nodes[go_term]["genes"].items():
                date_obj = convert_to_datetime(details["go_details"]["date_created"])
                month_year = date_obj.strftime("%Y-%m")  # Format as "YYYY-MM"
                dates[month_year] += 1

    # Sort the dates chronologically
    sorted_dates = sorted(dates.items(), key=lambda x: datetime.strptime(x[0], "%Y-%m"))

    # Extract x and y values
    x_vals, y_vals = zip(*sorted_dates)

    # Extract unique years and their first month occurrence for labeling purposes
    year_ticks = [date for idx, date in enumerate(x_vals) if date.endswith("-01")]

    # Plot the histogram
    plt.bar(x_vals, y_vals, edgecolor="k", alpha=0.7)
    plt.title("Gene Ontology Annotation Dates (Grouped by Month)", fontsize=24)
    plt.xlabel("Date (Year)", labelpad=10, fontsize=24)
    plt.ylabel("Number of Annotations", labelpad=10, fontsize=24)

    # Display only the labels for the years using their first month
    plt.xticks(year_ticks, rotation=45, fontsize=16)
    plt.yticks(fontsize=16)

    # Make year ticks longer for distinction
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text() in year_ticks:
            tick.tick1line.set_markersize(10)
            tick.label1.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(
        "./notes/assets/images/gene-ontology-annotation-dates-grouped-by-month.png",
        dpi=300,
    )
    # plt.show()
    plt.close()


def plot_histogram_of_gene_counts(G_go: nx.DiGraph):
    # Extracting the gene sets for each GO term
    gene_sets = [G_go.nodes[node].get("gene_set", []) for node in G_go.nodes()]

    # Getting the counts of genes for each GO term
    gene_counts = [len(gene_set) for gene_set in gene_sets]

    # Plotting the histogram
    plt.hist(gene_counts, bins=100, alpha=0.7)
    plt.title("Histogram of Gene Counts per GO Term")
    plt.xlabel("Number of Genes")
    plt.ylabel("Number of GO Terms")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_histogram_of_contained_gene_counts(
    G_go: nx.DiGraph, gene_set: set = None, show_min_max: bool = False
):
    def compute_containment(go_term):
        """Function to compute containment of a given term with its subsequent terms."""

        # Reverse the graph to travel in the opposite direction
        G_reverse = G_go.reverse(copy=False)

        # Find all reachable nodes from the current term in the reversed graph
        reachable_nodes = nx.single_source_shortest_path_length(
            G_reverse, go_term
        ).keys()

        # Get gene sets of all these nodes
        gene_sets = [
            set(G_go.nodes[node].get("gene_set", [])) for node in reachable_nodes
        ]

        # Compute the union of all gene sets
        contained_genes = set.union(*gene_sets)
        if gene_set is not None:
            contained_genes = contained_genes.intersection(gene_set)
        return contained_genes

    # Construct containment_dictionary
    containment_dictionary = {node: compute_containment(node) for node in G_go.nodes()}

    # Get counts for histogram
    contained_gene_counts = [len(genes) for genes in containment_dictionary.values()]

    # Plotting the histogram
    plt.hist(contained_gene_counts, bins=100, alpha=0.7)
    plt.title("Histogram of Contained Gene Counts per GO Term")
    plt.xlabel("Number of Contained Genes")
    plt.ylabel("Number of GO Terms")

    # Optional vertical dashed lines for min and max values
    if show_min_max:
        plt.axvline(min(contained_gene_counts), color="red", linestyle="--", alpha=0.7)
        plt.axvline(max(contained_gene_counts), color="red", linestyle="--", alpha=0.7)

    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def compare_go_terms(G_raw, G):
    matching_nodes = 0
    total_nodes = len(G_raw.nodes())
    total_new_terms = (
        0  # This will accumulate the total new terms added to G_raw from G
    )

    for node in G_raw.nodes():
        # Extract GO terms from the complex graph G_raw
        go_details_G_raw = G_raw.nodes[node]["go_details"]
        go_terms_G_raw = {
            detail["go"]["go_id"] for detail in go_details_G_raw if "go" in detail
        }

        # Check if the node exists in the simpler graph G
        if node in G.nodes():
            # Extract GO terms from the simple graph G
            go_terms_G = {rec["GO_ID"] for rec in G.nodes[node]["annotations"]}

            # Check if the terms in G are exactly the same as in G_raw
            if go_terms_G == go_terms_G_raw:
                matching_nodes += 1
            else:
                # Calculate the new terms from G that aren't in G_raw
                new_terms = go_terms_G - go_terms_G_raw
                total_new_terms += len(new_terms)
                # Optionally, report the node and the number of new terms
                print(
                    f"Node {node}: {len(new_terms)} new terms from G would be added to G_raw."
                )
        else:
            # Handle the case where a node in G_raw does not exist in G
            print(f"Node {node} from G_raw does not exist in G.")

    print(f"{matching_nodes} out of {total_nodes} nodes have matching GO term sets.")
    print(f"A total of {total_new_terms} new GO terms would be added to G_raw from G.")


def go_gaf_investigation():
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from Bio.UniProt.GOA import gafiterator
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    genome.drop_chrmt()
    genome.drop_empty_go()
    print(f"Gene Set Length: {len(genome.gene_set)}")
    # Initialize the graph
    G = nx.Graph()

    # Add nodes to the graph
    for gene_name in genome.gene_set:
        G.add_node(gene_name, annotations=[])

    # Define your filters here
    synonym_filter = {"Synonym": genome.gene_set}

    # Function to check if a record matches the given filters
    def record_has(rec, filters):
        for field, values in filters.items():
            if field in rec and not set(rec[field]).isdisjoint(values):
                return True
        return False

    # Read the GAF file and apply filters
    gaf_path = osp.join(DATA_ROOT, "data/sgd.gaf")
    with open(gaf_path) as handle:
        for record in gafiterator(handle):
            # Apply filters
            for synonym in record.get("Synonym", []):
                if synonym in genome.gene_set:
                    G.nodes[synonym]["annotations"].append(record)
                    break  # Once matched, no need to check other synonyms

    # use this len(G.nodes[[node for node in G.nodes][100]]["annotations"])
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    graph.G_go
    print()
    compare_go_terms(graph.G_raw, G)


import networkx as nx
import plotly.graph_objects as go


def plotly_go_graph(G):
    # Define color map for namespaces
    namespace_colors = {
        "super_root": "#122A4B",
        "biological_process": "#FF552E",
        "molecular_function": "#8286C2",
        "cellular_component": "#D23943",
        "missing": "#AAAAAA",  # Color for missing namespace
    }

    # Determine levels and layout
    max_level = max(data.get("level", 0) for node, data in G.nodes(data=True))
    min_level = min(data.get("level", 0) for node, data in G.nodes(data=True))
    level_spacing = 2.0  # Increased spacing for levels
    horizontal_spacing = 2.0  # Increased spacing horizontally

    new_pos = {}
    for node, data in G.nodes(data=True):
        level = data.get("level", 0)
        nodes_in_level = [n for n, d in G.nodes(data=True) if d.get("level") == level]
        idx_in_level = nodes_in_level.index(node)
        total_nodes_in_level = len(nodes_in_level)
        new_x = (idx_in_level - total_nodes_in_level / 2) * horizontal_spacing
        new_y = (max_level - level) * level_spacing - min_level * level_spacing
        new_pos[node] = (new_x, new_y)

    # Creating edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = new_pos[edge[0]]
        x1, y1 = new_pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Creating node traces
    node_x = []
    node_y = []
    node_info = []
    node_color = []
    for node in G.nodes():
        x, y = new_pos[node]
        node_x.append(x)
        node_y.append(y)
        node_data = G.nodes[node]
        name = node_data.get("name", "none")
        level = node_data.get("level", "none")
        namespace = node_data.get("namespace", "missing")
        info_str = f"{node}: name={name}, level={level}"
        node_info.append(info_str)
        node_color.append(namespace_colors.get(namespace, "missing"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_info,
        marker=dict(showscale=False, size=7, color=node_color, line=None),
    )
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Gene Ontology Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    )

    fig.show()


def plot_go_node_types(G):
    # Define a color map for namespaces
    namespace_colors = {
        "super_root": "#122A4B",
        "biological_process": "#FF552E",
        "molecular_function": "#8286C2",
        "cellular_component": "#D23943",
        "missing": "#AAAAAA",  # Color for missing namespace
    }

    # Count the occurrences of each namespace type
    namespace_counts = {
        "super_root": 0,
        "biological_process": 0,
        "molecular_function": 0,
        "cellular_component": 0,
        "missing": 0,
    }

    # Traverse the nodes and count their namespace type
    for node in G.nodes(data=True):
        namespace = node[1].get("namespace", "missing")
        if namespace in namespace_counts:
            namespace_counts[namespace] += 1
        else:
            namespace_counts["missing"] += 1

    # Prepare data for Plotly bar chart
    namespaces = list(namespace_counts.keys())
    counts = list(namespace_counts.values())
    colors = [namespace_colors[ns] for ns in namespaces]

    # Create a bar chart using Plotly
    fig = go.Figure(
        data=[go.Bar(x=namespaces, y=counts, marker_color=colors)],
        layout=go.Layout(
            title="GO Node Types Distribution",
            xaxis_title="GO Node Type",
            yaxis_title="Number of Nodes",
        ),
    )
    fig.show()


def plot_pathway_annotations(G_gene: nx.Graph):
    # Create a list to store the number of pathway annotations for each gene
    pathway_counts = []
    for node_name, node_data in G_gene.nodes(data=True):
        if "pathways" in node_data and node_data["pathways"] is not None:
            pathway_counts.append(len(node_data["pathways"]))
        else:
            pathway_counts.append(0)

    # Calculate the fraction of genes with at least one annotation
    total_genes = len(pathway_counts)
    genes_with_annotation = sum(count > 0 for count in pathway_counts)
    fraction_with_annotation = genes_with_annotation / total_genes

    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(
        pathway_counts,
        bins=range(max(pathway_counts) + 2),
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Number of Pathway Annotations")
    plt.ylabel("Number of Genes")
    plt.title("Histogram of Pathway Annotations per Gene")
    plt.xticks(range(max(pathway_counts) + 1))

    # Add labels for the count of genes in each bin
    for i in range(max(pathway_counts) + 1):
        count = sum(count == i for count in pathway_counts)
        plt.text(i, count + 0.1, str(count), ha="center", va="bottom")

    # Add text displaying the fraction of genes with at least one annotation
    plt.text(
        0.95,
        0.95,
        f"Fraction of genes with annotation: {fraction_with_annotation:.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
    )
    plt.savefig("./notes/assets/images/pathway-annotations-per-gene.png", dpi=300)
    plt.tight_layout()
    plt.show()


def old_main() -> None:
    import os
    import random

    import matplotlib.pyplot as plt
    import pandas as pd
    from dotenv import load_dotenv

    from torchcell.datasets.scerevisiae import (
        DmfCostanzo2016Dataset,
        SmfCostanzo2016Dataset,
    )

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=True,
    )
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        genome=genome,
    )
    # graph.read_raw()

    print("-----------")
    print(
        f"G_string9_1_neighborhood: {set(graph.G_string9_1_neighborhood.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_fusion: {set(graph.G_string9_1_fusion.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_cooccurence: {set(graph.G_string9_1_cooccurence.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_coexpression: {set(graph.G_string9_1_coexpression.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_experimental: {set(graph.G_string9_1_experimental.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_database: {set(graph.G_string9_1_database.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string9_1_combined: {set(graph.G_string9_1_combined.nodes()) - genome.gene_set}"
    )
    print("-----------")
    print(
        f"G_string12_0_neighborhood: {set(graph.G_string12_0_neighborhood.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_fusion: {set(graph.G_string12_0_fusion.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_cooccurence: {set(graph.G_string12_0_cooccurence.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_coexpression: {set(graph.G_string12_0_coexpression.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_experimental: {set(graph.G_string12_0_experimental.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_database: {set(graph.G_string12_0_database.nodes()) - genome.gene_set}"
    )
    print(
        f"G_string12_0_combined: {set(graph.G_string12_0_combined.nodes()) - genome.gene_set}"
    )
    print("-----------")

    print()
    #
    # dmf_dataset = DmfCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_1e5"),
    #     preprocess={"duplicate_resolution": "low_dmf_std"},
    #     # subset_n=100,
    # )
    # smf_dataset = SmfCostanzo2016Dataset(
    #     root=osp.join(DATA_ROOT, "data/scerevisiae/costanzo2016_smf"),
    #     preprocess={"duplicate_resolution": "low_std_both"},
    #     skip_process_file_exist_check=True,
    # )
    gene_set = genome.gene_set
    #
    print(graph.G_go.number_of_nodes())
    G = graph.G_go.copy()

    ##### Filtering
    # plot_histogram_of_gene_counts(G)
    # plot_histogram_of_contained_gene_counts(G, gene_set=gene_set, show_min_max=True)
    # G = filter_by_date(G, "2018-02-01")
    G = filter_by_date(G, "2017-07-19")
    print(f"After date filter: {G.number_of_nodes()}")
    G = filter_go_IGI(G)
    print(f"After IGI filter: {G.number_of_nodes()}")
    G = filter_redundant_terms(G)
    print(f"After redundant filter: {G.number_of_nodes()}")
    # (filter_by_contained_genes(G, 1, gene_set=gene_set)).number_of_nodes()
    G = filter_by_contained_genes(G, n=6, gene_set=gene_set)
    print(f"After containment filter: {G.number_of_nodes()}")
    # print()
    # plot_go_graph(G)
    print()

    # plot_annotation_dates_by_month(graph.G_go)
    # plot_go_graph(graph.G_go)
    plotly_go_graph(G)
    plot_go_node_types(G)


def main():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    genome = SCerevisiaeGenome(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), overwrite=True
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    print()
    # Plot the pathway annotations per gene
    plot_pathway_annotations(graph.G_gene)


if __name__ == "__main__":
    old_main()
    # main_go()

    # # TODO maybe remove
    # @staticmethod
    # def parse_genome(genome) -> ParsedGenome:
    #     # BUG we have to do this black magic because when you merge datasets with +
    #     # the genome is None
    #     if genome is None:
    #         return None
    #     else:
    #         data = {}
    #         data["gene_set"] = genome.gene_set
    #         return ParsedGenome(**data)
    #         data["gene_set"] = genome.gene_set
    #         return ParsedGenome(**data)
