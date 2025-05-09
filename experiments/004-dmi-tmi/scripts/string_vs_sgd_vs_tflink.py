# experiments/004-dmi-tmi/scripts/string_vs_sgd_vs_tflink
# [[experiments.004-dmi-tmi.scripts.string_vs_sgd_vs_tflink]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/004-dmi-tmi/scripts/string_vs_sgd_vs_tflink
# Test file: experiments/004-dmi-tmi/scripts/test_string_vs_sgd_vs_tflink.py


import os
import os.path as osp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.patches import Patch
from itertools import combinations
from sortedcontainers import SortedDict
import logging
import torchcell
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph, GeneGraph, GeneMultiGraph

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def setup_environment():
    """Initialize environment variables and create necessary directories"""
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Create output directory if it doesn't exist
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    return DATA_ROOT, ASSET_IMAGES_DIR


def load_networks(DATA_ROOT):
    """Initialize the SCerevisiaeGraph and load all required networks"""
    # Initialize genome and graph objects
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,  # Use existing data
    )

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Dictionary to store all networks (already GeneGraph objects)
    networks = {
        # Non-STRING networks
        "Regulatory": graph.G_regulatory,
        "Physical": graph.G_physical,
        "TFLink": graph.G_tflink,
        # STRING v12.0 networks
        "STRING12_Neighborhood": graph.G_string12_0_neighborhood,
        "STRING12_Fusion": graph.G_string12_0_fusion,
        "STRING12_Cooccurence": graph.G_string12_0_cooccurence,
        "STRING12_Coexpression": graph.G_string12_0_coexpression,
        "STRING12_Experimental": graph.G_string12_0_experimental,
        "STRING12_Database": graph.G_string12_0_database,
    }

    # Print basic statistics for each network
    print("\n===== Network Statistics =====")
    for name, network in networks.items():
        print(
            f"{name}: {network.graph.number_of_nodes()} nodes, {network.graph.number_of_edges()} edges"
        )

    return networks, genome


def convert_to_undirected_edge_sets(networks):
    """Convert networks to sets of undirected edges for comparison"""
    edge_sets = {}

    for name, network in networks.items():
        # Create an edge set - for directed graphs, we ignore the direction
        edges = set()
        for u, v in network.graph.edges():
            # Sort the nodes to handle undirected edges uniformly
            edge = tuple(sorted([u, v]))
            edges.add(edge)

        edge_sets[name] = edges

    return edge_sets


def compute_pairwise_comparisons(edge_sets):
    """Compute pairwise comparisons between all networks"""
    network_names = list(edge_sets.keys())
    comparisons = {}

    for i, name1 in enumerate(network_names):
        for j, name2 in enumerate(network_names):
            if i < j:  # Only compute for unique pairs
                set1 = edge_sets[name1]
                set2 = edge_sets[name2]

                # Compute overlap and Jaccard similarity
                intersection = set1.intersection(set2)
                union = set1.union(set2)

                comparisons[(name1, name2)] = {
                    "network1_edges": len(set1),
                    "network2_edges": len(set2),
                    "shared_edges": len(intersection),
                    "union_edges": len(union),
                    "jaccard_similarity": (
                        len(intersection) / len(union) if union else 0
                    ),
                    "overlap_coefficient1": (
                        len(intersection) / len(set1) if set1 else 0
                    ),
                    "overlap_coefficient2": (
                        len(intersection) / len(set2) if set2 else 0
                    ),
                }

    return comparisons


def visualize_network_sizes(networks, ASSET_IMAGES_DIR):
    """Create a bar chart comparing network sizes with nodes on secondary y-axis"""
    networks_data = [
        (name, G.graph.number_of_nodes(), G.graph.number_of_edges())
        for name, G in networks.items()
    ]

    # Sort by edge count for better visualization
    networks_data.sort(key=lambda x: x[2], reverse=True)

    names = [data[0] for data in networks_data]
    nodes = [data[1] for data in networks_data]
    edges = [data[2] for data in networks_data]

    x = np.arange(len(names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot edges on primary y-axis
    bars1 = ax1.bar(
        x - width / 2,
        edges,
        width,
        label="Edges",
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
    )
    ax1.set_xlabel("Network", fontsize=14)
    ax1.set_ylabel(
        "Edge Count",
        fontsize=14,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
    )
    ax1.tick_params(
        axis="y", labelcolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    )

    # Create secondary y-axis and plot nodes
    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        x + width / 2,
        nodes,
        width,
        label="Nodes",
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
    )
    ax2.set_ylabel(
        "Node Count",
        fontsize=14,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
    )
    ax2.tick_params(
        axis="y", labelcolor=plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    )

    # Set x-axis ticks and title
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_title("Network Sizes Comparison", fontsize=16)

    # Add legend with both bars
    fig.legend(
        [bars1, bars2], ["Edges", "Nodes"], loc="upper right", bbox_to_anchor=(0.9, 0.9)
    )

    fig.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "network_sizes_comparison.png"))
    plt.close()


def visualize_jaccard_similarity_heatmap(comparisons, ASSET_IMAGES_DIR):
    """Create a heatmap of Jaccard similarity between networks"""
    # Extract all unique network names
    network_names = set()
    for name1, name2 in comparisons.keys():
        network_names.add(name1)
        network_names.add(name2)

    network_names = sorted(list(network_names))
    n = len(network_names)

    # Create a matrix for the heatmap
    similarity_matrix = np.zeros((n, n))

    # Fill the matrix with Jaccard similarity values
    for i, name1 in enumerate(network_names):
        for j, name2 in enumerate(network_names):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Perfect similarity with itself
            elif (name1, name2) in comparisons:
                similarity_matrix[i, j] = comparisons[(name1, name2)][
                    "jaccard_similarity"
                ]
            elif (name2, name1) in comparisons:
                similarity_matrix[i, j] = comparisons[(name2, name1)][
                    "jaccard_similarity"
                ]

    # Create the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=network_names,
        yticklabels=network_names,
    )
    plt.title("Jaccard Similarity Between Networks", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "jaccard_similarity_heatmap.png"))
    plt.close()


def visualize_overlap_matrix(comparisons, ASSET_IMAGES_DIR):
    """Create a matrix visualization of overlap between networks"""
    # Extract all unique network names
    network_names = set()
    for name1, name2 in comparisons.keys():
        network_names.add(name1)
        network_names.add(name2)

    network_names = sorted(list(network_names))
    n = len(network_names)

    # Create matrices for shared edges and overlap coefficients
    shared_edges_matrix = np.zeros((n, n))
    overlap_coef_matrix = np.zeros((n, n))

    # Fill the matrices
    for i, name1 in enumerate(network_names):
        for j, name2 in enumerate(network_names):
            if i == j:
                shared_edges_matrix[i, j] = 0  # No overlap with itself displayed
                overlap_coef_matrix[i, j] = 1.0  # Perfect overlap with itself
            elif (name1, name2) in comparisons:
                shared_edges_matrix[i, j] = comparisons[(name1, name2)]["shared_edges"]
                # Use the minimum of the two overlap coefficients
                overlap_coef_matrix[i, j] = min(
                    comparisons[(name1, name2)]["overlap_coefficient1"],
                    comparisons[(name1, name2)]["overlap_coefficient2"],
                )
            elif (name2, name1) in comparisons:
                shared_edges_matrix[i, j] = comparisons[(name2, name1)]["shared_edges"]
                overlap_coef_matrix[i, j] = min(
                    comparisons[(name2, name1)]["overlap_coefficient1"],
                    comparisons[(name2, name1)]["overlap_coefficient2"],
                )

    # Create the shared edges heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        shared_edges_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        xticklabels=network_names,
        yticklabels=network_names,
    )
    plt.title("Number of Shared Edges Between Networks", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "shared_edges_matrix.png"))
    plt.close()

    # Create the overlap coefficient heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(
        np.ones_like(overlap_coef_matrix, dtype=bool)
    )  # Mask for the upper triangle
    sns.heatmap(
        overlap_coef_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        mask=mask,
        xticklabels=network_names,
        yticklabels=network_names,
    )
    plt.title("Overlap Coefficient Between Networks", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "overlap_coefficient_matrix.png"))
    plt.close()


def create_network_overlap_charts(edge_sets, ASSET_IMAGES_DIR):
    """Create charts showing overlap between specific pairs of networks"""
    # Focus on comparing regulatory, physical, and TFLink networks
    key_networks = ["Regulatory", "Physical", "TFLink"]

    for key_network in key_networks:
        if key_network not in edge_sets:
            continue

        # Prepare data for visualization
        comparison_data = []
        for name, edges in edge_sets.items():
            if name != key_network and name in edge_sets:
                intersection = edge_sets[key_network].intersection(edges)
                comparison_data.append(
                    {
                        "network": name,
                        "unique_to_key": len(edge_sets[key_network] - edges),
                        "shared": len(intersection),
                        "unique_to_other": len(edges - edge_sets[key_network]),
                        "jaccard": (
                            len(intersection) / len(edge_sets[key_network].union(edges))
                            if edges
                            else 0
                        ),
                    }
                )

        # Sort by Jaccard similarity
        comparison_data.sort(key=lambda x: x["jaccard"], reverse=True)

        # Create stacked bar chart
        networks = [item["network"] for item in comparison_data]
        unique_key = [item["unique_to_key"] for item in comparison_data]
        shared = [item["shared"] for item in comparison_data]
        unique_other = [item["unique_to_other"] for item in comparison_data]

        plt.figure(figsize=(14, 8))

        # Plot stacked bars
        plt.bar(networks, unique_key, label=f"Unique to {key_network}")
        plt.bar(networks, shared, bottom=unique_key, label="Shared")
        plt.bar(
            networks,
            unique_other,
            bottom=[unique_key[i] + shared[i] for i in range(len(shared))],
            label="Unique to other network",
        )

        plt.title(
            f"Edge Overlap Between {key_network} Network and Other Networks",
            fontsize=16,
        )
        plt.xlabel("Network", fontsize=14)
        plt.ylabel("Number of Edges", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        plt.savefig(osp.join(ASSET_IMAGES_DIR, f"{key_network}_vs_others_overlap.png"))
        plt.close()


def visualize_tf_regulatory_comparison(networks, ASSET_IMAGES_DIR):
    """Create a special visualization comparing TFLink and Regulatory networks"""
    if "TFLink" not in networks or "Regulatory" not in networks:
        print("Cannot compare TFLink and Regulatory networks - one or both are missing")
        return

    # Get the networks
    tflink = networks["TFLink"].graph
    regulatory = networks["Regulatory"].graph

    # Get TFs (nodes with outgoing edges) in each network
    tflink_tfs = {u for u, _ in tflink.out_edges()}
    regulatory_tfs = {u for u, _ in regulatory.out_edges()}

    # Calculate overlap of TFs
    common_tfs = tflink_tfs.intersection(regulatory_tfs)
    only_tflink_tfs = tflink_tfs - regulatory_tfs
    only_regulatory_tfs = regulatory_tfs - tflink_tfs

    # Get colors from the current style's color cycle
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create pie chart for TF overlap
    plt.figure(figsize=(10, 10))
    sizes = [len(only_regulatory_tfs), len(common_tfs), len(only_tflink_tfs)]
    labels = [
        f"Only in Regulatory\n({sizes[0]} TFs)",
        f"Common\n({sizes[1]} TFs)",
        f"Only in TFLink\n({sizes[2]} TFs)",
    ]

    # Use first three colors from the color cycle
    colors = [color_cycle[1], color_cycle[2], color_cycle[3]]

    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")
    plt.title("Transcription Factors Distribution Between Networks", fontsize=16)
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "tf_distribution_pie.png"))
    plt.close()

    # Create a dataframe of top TFs by number of targets
    tf_data = []
    for tf in tflink_tfs.union(regulatory_tfs):
        tflink_targets = (
            list(v for _, v in tflink.out_edges(tf)) if tf in tflink else []
        )
        reg_targets = (
            list(v for _, v in regulatory.out_edges(tf)) if tf in regulatory else []
        )

        # Get common targets
        common_targets = set(tflink_targets).intersection(set(reg_targets))

        tf_data.append(
            {
                "TF": tf,
                "TFLink_targets": len(tflink_targets),
                "Regulatory_targets": len(reg_targets),
                "Common_targets": len(common_targets),
            }
        )

    # Convert to dataframe and sort by total targets
    tf_df = pd.DataFrame(tf_data)
    tf_df["Total_targets"] = (
        tf_df["TFLink_targets"] + tf_df["Regulatory_targets"] - tf_df["Common_targets"]
    )
    tf_df = tf_df.sort_values(by="Total_targets", ascending=False).head(15)

    # Create bar chart for top TFs
    plt.figure(figsize=(14, 8))
    bar_width = 0.3
    indices = np.arange(len(tf_df))

    plt.bar(
        indices - bar_width,
        tf_df["TFLink_targets"],
        bar_width,
        label="TFLink Targets",
        color=color_cycle[3],
        alpha=0.7,
    )
    plt.bar(
        indices,
        tf_df["Common_targets"],
        bar_width,
        label="Common Targets",
        color=color_cycle[2],
        alpha=0.7,
    )
    plt.bar(
        indices + bar_width,
        tf_df["Regulatory_targets"],
        bar_width,
        label="Regulatory Targets",
        color=color_cycle[1],
        alpha=0.7,
    )

    plt.xlabel("Transcription Factor", fontsize=14)
    plt.ylabel("Number of Target Genes", fontsize=14)
    plt.title("Top Transcription Factors Comparison", fontsize=16)
    plt.xticks(indices, tf_df["TF"], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "top_tfs_comparison.png"))
    plt.close()

    # Compute target overlap for common TFs
    if common_tfs:
        target_overlaps = []
        for tf in common_tfs:
            tflink_targets = set(v for _, v in tflink.out_edges(tf))
            reg_targets = set(v for _, v in regulatory.out_edges(tf))

            overlap = len(tflink_targets.intersection(reg_targets))
            jaccard = (
                overlap / len(tflink_targets.union(reg_targets))
                if tflink_targets or reg_targets
                else 0
            )

            target_overlaps.append(
                {
                    "TF": tf,
                    "TFLink_targets": len(tflink_targets),
                    "Regulatory_targets": len(reg_targets),
                    "Overlap": overlap,
                    "Jaccard": jaccard,
                }
            )

        # Sort by Jaccard similarity
        target_overlaps.sort(key=lambda x: x["Jaccard"], reverse=True)

        # Create a plot for the top 10 TFs with highest target overlap
        top_n = min(10, len(target_overlaps))
        top_tfs = target_overlaps[:top_n]

        plt.figure(figsize=(14, 8))

        x = np.arange(len(top_tfs))
        width = 0.25

        plt.bar(
            x - width,
            [tf["TFLink_targets"] for tf in top_tfs],
            width,
            label="TFLink Targets",
            color=color_cycle[3],
            alpha=0.7,
        )
        plt.bar(
            x,
            [tf["Overlap"] for tf in top_tfs],
            width,
            label="Common Targets",
            color=color_cycle[2],
            alpha=0.7,
        )
        plt.bar(
            x + width,
            [tf["Regulatory_targets"] for tf in top_tfs],
            width,
            label="Regulatory Targets",
            color=color_cycle[1],
            alpha=0.7,
        )

        plt.title(
            "TFs with Highest Target Overlap Between TFLink and Regulatory Networks",
            fontsize=16,
        )
        plt.xlabel("Transcription Factor", fontsize=14)
        plt.ylabel("Number of Target Genes", fontsize=14)
        plt.xticks(x, [tf["TF"] for tf in top_tfs], rotation=45, ha="right")

        # Add Jaccard similarity as text above bars
        for i, tf in enumerate(top_tfs):
            plt.text(
                i,
                max(tf["TFLink_targets"], tf["Regulatory_targets"]) + 5,
                f"J={tf['Jaccard']:.2f}",
                ha="center",
            )

        plt.legend()
        plt.tight_layout()
        plt.savefig(osp.join(ASSET_IMAGES_DIR, "tfs_with_highest_target_overlap.png"))
        plt.close()


def create_multigraph_example(networks, ASSET_IMAGES_DIR):
    """Create an example GeneMultiGraph and visualize connectivity stats"""
    # Create a GeneMultiGraph with selected networks
    selected_networks = ["Regulatory", "Physical", "TFLink"]
    graphs_dict = SortedDict()

    for name in selected_networks:
        if name in networks:
            graphs_dict[name] = networks[name]

    # Create the GeneMultiGraph
    multigraph = GeneMultiGraph(graphs=graphs_dict)

    # Print info about the multigraph
    print("\n===== GeneMultiGraph Example =====")
    print(f"Created multigraph with {len(multigraph)} graphs")
    print(multigraph)

    # Create a visualization of graph sizes in the multigraph
    plt.figure(figsize=(10, 6))
    graph_names = list(multigraph.keys())
    node_counts = [g.graph.number_of_nodes() for g in multigraph.values()]
    edge_counts = [g.graph.number_of_edges() for g in multigraph.values()]

    x = np.arange(len(graph_names))
    width = 0.35

    plt.bar(x - width / 2, node_counts, width, label="Nodes")
    plt.bar(x + width / 2, edge_counts, width, label="Edges")

    plt.xlabel("Network")
    plt.ylabel("Count")
    plt.title("GeneMultiGraph Component Networks")
    plt.xticks(x, graph_names)
    plt.legend()

    plt.tight_layout()
    plt.savefig(osp.join(ASSET_IMAGES_DIR, "gene_multigraph_example.png"))
    plt.close()


def main():
    """Main function to run the network comparison analysis"""
    # Setup environment and paths
    DATA_ROOT, ASSET_IMAGES_DIR = setup_environment()
    print(f"Asset images will be saved to: {ASSET_IMAGES_DIR}")

    # Load all networks
    networks, genome = load_networks(DATA_ROOT)

    # Convert networks to edge sets for comparison
    edge_sets = convert_to_undirected_edge_sets(networks)

    # Compute pairwise comparisons
    comparisons = compute_pairwise_comparisons(edge_sets)

    # Print some comparison results
    print("\n===== Pairwise Comparison Results =====")
    for (name1, name2), results in sorted(comparisons.items()):
        print(f"\n{name1} vs {name2}:")
        print(f"  Network 1 edges: {results['network1_edges']}")
        print(f"  Network 2 edges: {results['network2_edges']}")
        print(f"  Shared edges: {results['shared_edges']}")
        print(f"  Jaccard similarity: {results['jaccard_similarity']:.4f}")

    # Generate visualizations
    print("\n===== Generating Visualizations =====")

    visualize_network_sizes(networks, ASSET_IMAGES_DIR)
    print("✅ Network sizes chart created")

    visualize_jaccard_similarity_heatmap(comparisons, ASSET_IMAGES_DIR)
    print("✅ Jaccard similarity heatmap created")

    visualize_overlap_matrix(comparisons, ASSET_IMAGES_DIR)
    print("✅ Edge overlap matrices created")

    create_network_overlap_charts(edge_sets, ASSET_IMAGES_DIR)
    print("✅ Network overlap charts created")

    visualize_tf_regulatory_comparison(networks, ASSET_IMAGES_DIR)
    print("✅ TF regulatory comparison visualizations created")

    # Create an example GeneMultiGraph
    create_multigraph_example(networks, ASSET_IMAGES_DIR)
    print("✅ GeneMultiGraph example created")

    print(f"\nAll visualizations saved to {ASSET_IMAGES_DIR}")


if __name__ == "__main__":
    main()
