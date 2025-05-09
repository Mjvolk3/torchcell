# experiments/004-dmi-tmi/scripts/dango_ppi_vs_sgd_ppi
# [[experiments.004-dmi-tmi.scripts.dango_ppi_vs_sgd_ppi]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/004-dmi-tmi/scripts/dango_ppi_vs_sgd_ppi
# Test file: experiments/004-dmi-tmi/scripts/test_dango_ppi_vs_sgd_ppi.py


import os
import os.path as osp
import gzip
import requests
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph import SCerevisiaeGraph

load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Create directory for STRING data
string_dir = os.path.join(DATA_ROOT, "data/string9.1/")
os.makedirs(string_dir, exist_ok=True)

# Create directory for STRING v12 data
string_v12_dir = os.path.join(DATA_ROOT, "data/string12.0/")
os.makedirs(string_v12_dir, exist_ok=True)

# Create directory for asset images if it doesn't exist
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)


def download_string_data(output_path, version="9.1"):
    """
    Download the STRING protein interaction file for S. cerevisiae

    Args:
        output_path: Path to save the downloaded file
        version: STRING database version ("9.1" or "12.0")
    """
    if version == "9.1":
        url = "http://string91.embl.de/newstring_download/protein.links.detailed.v9.1/4932.protein.links.detailed.v9.1.txt.gz"
    elif version == "12.0":
        url = "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/4932.protein.links.detailed.v12.0.txt.gz"
    else:
        raise ValueError(f"Unsupported STRING version: {version}")

    if os.path.exists(output_path):
        print(
            f"STRING v{version} data already exists at {output_path}, skipping download"
        )
        return

    print(f"Downloading STRING v{version} data to {output_path}...")
    response = requests.get(url, stream=True)
    with open(output_path, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            f.write(chunk)
    print(f"STRING v{version} download complete!")


def load_string_data(file_path, version="9.1"):
    """
    Load the STRING interaction data from the gzipped file

    Args:
        file_path: Path to the STRING data file
        version: STRING database version ("9.1" or "12.0")

    Returns:
        Dictionary of NetworkX graphs for each network type
    """
    # Load the data
    df = pd.read_csv(file_path, sep=" ", compression="gzip")

    print(f"\nColumns in the STRING v{version} dataframe:")
    print(df.columns.tolist())

    print(f"\nSample data from STRING v{version}:")
    print(df.head())

    # Network types we're interested in
    network_types = [
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
    ]

    networks = {}

    # Create a combined graph to include ALL interactions from all network types
    G_combined = nx.Graph()

    for network_type in network_types:
        # Create a graph for each network type
        G = nx.Graph()

        # Filter interactions where the network type score > 0
        type_df = df[df[network_type] > 0][["protein1", "protein2", network_type]]

        # Add edges to the individual graph
        for _, row in type_df.iterrows():
            G.add_edge(row["protein1"], row["protein2"], weight=row[network_type])

            # Also add to the combined graph
            G_combined.add_edge(row["protein1"], row["protein2"])

        networks[network_type] = G
        print(
            f"STRING v{version} {network_type} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )

    # Add the combined graph to the networks dictionary
    networks["combined"] = G_combined
    print(
        f"STRING v{version} combined network: {G_combined.number_of_nodes()} nodes, {G_combined.number_of_edges()} edges"
    )

    return networks


def strip_string_prefix(protein_id):
    """Strip the '4932.' prefix from STRING protein IDs"""
    if protein_id.startswith("4932."):
        return protein_id[5:]
    return protein_id


def compare_networks(torchcell_graph, string_networks, graph_type, string_version):
    """
    Compare a torchcell graph (physical or regulatory) to the STRING networks

    Args:
        torchcell_graph: NetworkX graph from SCerevisiaeGraph (physical or regulatory)
        string_networks: Dictionary of STRING networks
        graph_type: String indicating the type of torchcell graph ('physical' or 'regulatory')
        string_version: STRING database version ("9.1" or "12.0")

    Returns:
        Dictionary with comparison results
    """
    # Convert torchcell graph to a set of edges (without direction)
    torchcell_edges = set()
    for edge in torchcell_graph.edges():
        # Sort the nodes to handle undirected edges
        edge = tuple(sorted(edge))
        torchcell_edges.add(edge)

    print(
        f"{graph_type.capitalize()} graph has {len(torchcell_edges)} edges (comparing with STRING v{string_version})"
    )

    # Create a dictionary to store comparison results
    comparison = {
        "torchcell_graph_edges": len(torchcell_edges),
        "overlap": {},
        "version": string_version,
    }

    for network_name, string_graph in string_networks.items():
        # Convert STRING graph edges to a set, removing the '4932.' prefix
        string_edges = set()
        for u, v in string_graph.edges():
            u_stripped = strip_string_prefix(u)
            v_stripped = strip_string_prefix(v)
            # Sort the nodes to handle undirected edges
            edge = tuple(sorted([u_stripped, v_stripped]))
            string_edges.add(edge)

        # Compute overlap
        overlap = torchcell_edges.intersection(string_edges)

        comparison["overlap"][network_name] = {
            "string_edges": len(string_edges),
            "shared_edges": len(overlap),
            "jaccard_similarity": (
                len(overlap) / len(torchcell_edges.union(string_edges))
                if torchcell_edges.union(string_edges)
                else 0
            ),
        }

    return comparison


def analyze_network_overlaps(string_networks):
    """
    Analyze the overlap between different STRING network types

    Args:
        string_networks: Dictionary of STRING networks

    Returns:
        overlap_counts: Dictionary with counts of edges in 1, 2, ... network types
        network_pairs_overlap: Matrix of pairwise overlaps between network types
    """
    network_types = [nt for nt in string_networks.keys() if nt != "combined"]

    # Convert each network to a set of edges
    network_edge_sets = {}
    for network_type in network_types:
        edge_set = set()
        for u, v in string_networks[network_type].edges():
            # Standardize the edge order
            edge = tuple(sorted([u, v]))
            edge_set.add(edge)
        network_edge_sets[network_type] = edge_set

    # Create a dictionary to count how many networks each edge appears in
    edge_network_count = {}
    for network_type, edge_set in network_edge_sets.items():
        for edge in edge_set:
            if edge not in edge_network_count:
                edge_network_count[edge] = 0
            edge_network_count[edge] += 1

    # Count how many edges appear in 1, 2, ... network types
    overlap_counts = {}
    for i in range(1, len(network_types) + 1):
        overlap_counts[i] = len(
            [e for e, count in edge_network_count.items() if count == i]
        )

    # Calculate pairwise overlaps between network types
    network_pairs_overlap = {}
    for i, type1 in enumerate(network_types):
        for j, type2 in enumerate(network_types):
            if i < j:  # Only compute for unique pairs
                overlap = len(
                    network_edge_sets[type1].intersection(network_edge_sets[type2])
                )
                network_pairs_overlap[(type1, type2)] = overlap

    return overlap_counts, network_pairs_overlap


def visualize_comparison(comparison, graph_type):
    """
    Create visualizations of the network comparison

    Args:
        comparison: Dictionary with comparison results
        graph_type: String indicating the type of torchcell graph ('physical' or 'regulatory')
    """
    string_version = comparison["version"]
    
    # Create a bar chart for edge counts
    plt.figure(figsize=(14, 7))

    # Prepare data - exclude combined for individual network comparison
    network_types = [nt for nt in comparison["overlap"].keys() if nt != "combined"]
    torchcell_counts = [comparison["torchcell_graph_edges"]] * len(network_types)
    string_counts = [comparison["overlap"][nt]["string_edges"] for nt in network_types]
    overlap_counts = [comparison["overlap"][nt]["shared_edges"] for nt in network_types]

    # Set up bar chart
    x = range(len(network_types))
    width = 0.25

    plt.bar(
        [i - width for i in x],
        torchcell_counts,
        width,
        label=f"Torchcell {graph_type.capitalize()} Graph",
    )
    plt.bar(x, string_counts, width, label=f"STRING v{string_version} Network")
    plt.bar([i + width for i in x], overlap_counts, width, label="Overlap")

    plt.xlabel("Network Type")
    plt.ylabel("Number of Interactions")
    plt.title(
        f"Comparison of {graph_type.capitalize()} Graph vs STRING v{string_version} Networks"
    )
    plt.xticks(x, network_types, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(
            ASSET_IMAGES_DIR, f"{graph_type}_network_comparison_v{string_version}.png"
        )
    )

    # Create a Jaccard similarity chart
    plt.figure(figsize=(12, 6))
    all_network_types = list(comparison["overlap"].keys())  # Include combined
    jaccard_values = [
        comparison["overlap"][nt]["jaccard_similarity"] for nt in all_network_types
    ]
    plt.bar(all_network_types, jaccard_values)
    plt.xlabel("Network Type")
    plt.ylabel("Jaccard Similarity")
    plt.title(
        f"Jaccard Similarity Between {graph_type.capitalize()} Graph and STRING v{string_version} Networks"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(
            ASSET_IMAGES_DIR, f"{graph_type}_jaccard_similarity_v{string_version}.png"
        )
    )

    # Create a separate visualization for combined network vs torchcell graph
    plt.figure(figsize=(10, 6))

    # Data for combined network comparison
    combined_data = [
        comparison["torchcell_graph_edges"],
        comparison["overlap"]["combined"]["string_edges"],
        comparison["overlap"]["combined"]["shared_edges"],
    ]
    labels = [
        f"Torchcell {graph_type.capitalize()}",
        f"STRING v{string_version} Combined",
        "Overlap",
    ]

    plt.bar(labels, combined_data)
    plt.ylabel("Number of Interactions")
    plt.title(
        f"Comparison of {graph_type.capitalize()} Graph vs Combined STRING v{string_version} Network"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        os.path.join(
            ASSET_IMAGES_DIR, f"{graph_type}_combined_comparison_v{string_version}.png"
        )
    )


def main():
    # Download STRING v9.1 data
    string_v91_file = os.path.join(
        string_dir, "4932.protein.links.detailed.v9.1.txt.gz"
    )
    download_string_data(string_v91_file, version="9.1")

    # Download STRING v12.0 data
    string_v12_file = os.path.join(
        string_v12_dir, "4932.protein.links.detailed.v12.0.txt.gz"
    )
    download_string_data(string_v12_file, version="12.0")

    # Load STRING v9.1 data
    string_v91_networks = load_string_data(string_v91_file, version="9.1")

    # Load STRING v12.0 data
    string_v12_networks = load_string_data(string_v12_file, version="12.0")

    # Load torchcell graphs
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,  # Set to False to use existing data
    )
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Access the physical and regulatory graphs
    physical_graph = graph.G_physical
    regulatory_graph = graph.G_regulatory

    print(
        f"Physical graph: {physical_graph.number_of_nodes()} nodes, {physical_graph.number_of_edges()} edges"
    )
    print(
        f"Regulatory graph: {regulatory_graph.number_of_nodes()} nodes, {regulatory_graph.number_of_edges()} edges"
    )

    # Compare with STRING v9.1
    print("\n=== Comparisons with STRING v9.1 ===")

    # Compare physical network with STRING v9.1
    physical_v91_comparison = compare_networks(
        physical_graph, string_v91_networks, "physical", "9.1"
    )

    # Print physical comparison results
    print("\nPhysical Graph vs STRING v9.1 Comparison Results:")
    print(f"Physical Graph: {physical_v91_comparison['torchcell_graph_edges']} edges")
    print("\nOverlap with STRING v9.1 networks:")
    for network, stats in physical_v91_comparison["overlap"].items():
        print(
            f"{network}: {stats['string_edges']} edges, {stats['shared_edges']} shared edges"
        )
        print(f"Jaccard similarity: {stats['jaccard_similarity']:.4f}")

    # Compare regulatory network with STRING v9.1
    regulatory_v91_comparison = compare_networks(
        regulatory_graph, string_v91_networks, "regulatory", "9.1"
    )

    # Print regulatory comparison results
    print("\nRegulatory Graph vs STRING v9.1 Comparison Results:")
    print(
        f"Regulatory Graph: {regulatory_v91_comparison['torchcell_graph_edges']} edges"
    )
    print("\nOverlap with STRING v9.1 networks:")
    for network, stats in regulatory_v91_comparison["overlap"].items():
        print(
            f"{network}: {stats['string_edges']} edges, {stats['shared_edges']} shared edges"
        )
        print(f"Jaccard similarity: {stats['jaccard_similarity']:.4f}")

    # Compare with STRING v12.0
    print("\n=== Comparisons with STRING v12.0 ===")

    # Compare physical network with STRING v12.0
    physical_v12_comparison = compare_networks(
        physical_graph, string_v12_networks, "physical", "12.0"
    )

    # Print physical comparison results
    print("\nPhysical Graph vs STRING v12.0 Comparison Results:")
    print(f"Physical Graph: {physical_v12_comparison['torchcell_graph_edges']} edges")
    print("\nOverlap with STRING v12.0 networks:")
    for network, stats in physical_v12_comparison["overlap"].items():
        print(
            f"{network}: {stats['string_edges']} edges, {stats['shared_edges']} shared edges"
        )
        print(f"Jaccard similarity: {stats['jaccard_similarity']:.4f}")

    # Compare regulatory network with STRING v12.0
    regulatory_v12_comparison = compare_networks(
        regulatory_graph, string_v12_networks, "regulatory", "12.0"
    )

    # Print regulatory comparison results
    print("\nRegulatory Graph vs STRING v12.0 Comparison Results:")
    print(
        f"Regulatory Graph: {regulatory_v12_comparison['torchcell_graph_edges']} edges"
    )
    print("\nOverlap with STRING v12.0 networks:")
    for network, stats in regulatory_v12_comparison["overlap"].items():
        print(
            f"{network}: {stats['string_edges']} edges, {stats['shared_edges']} shared edges"
        )
        print(f"Jaccard similarity: {stats['jaccard_similarity']:.4f}")

    # Visualize the comparisons
    print("\nGenerating visualizations...")
    visualize_comparison(physical_v91_comparison, "physical")
    visualize_comparison(regulatory_v91_comparison, "regulatory")
    visualize_comparison(physical_v12_comparison, "physical")
    visualize_comparison(regulatory_v12_comparison, "regulatory")
    print("Visualizations complete!")

    # Analyze network overlaps
    overlap_counts_v91, network_pairs_v91 = analyze_network_overlaps(
        string_v91_networks
    )
    overlap_counts_v12, network_pairs_v12 = analyze_network_overlaps(
        string_v12_networks
    )

    # Print results
    print("\nSTRING v9.1 network overlap analysis:")
    print("Number of edges appearing in N network types:")
    for count, num_edges in sorted(overlap_counts_v91.items()):
        print(f"In {count} networks: {num_edges} edges")

    print("\nPairwise overlaps between network types:")
    for (type1, type2), overlap in sorted(network_pairs_v91.items()):
        print(f"{type1} ∩ {type2}: {overlap} edges")

    print("\nSTRING v12.0 network overlap analysis:")
    print("Number of edges appearing in N network types:")
    for count, num_edges in sorted(overlap_counts_v12.items()):
        print(f"In {count} networks: {num_edges} edges")

    print("\nPairwise overlaps between network types:")
    for (type1, type2), overlap in sorted(network_pairs_v12.items()):
        print(f"{type1} ∩ {type2}: {overlap} edges")


if __name__ == "__main__":
    main()
