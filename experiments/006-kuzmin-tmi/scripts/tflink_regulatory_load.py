#!/usr/bin/env python
# tflink_regulatory_load.py
# Simple script to load TFLink and Regulatory graphs for S. cerevisiae
# Only requires: pandas, networkx, requests

import os
import os.path as osp
import json
import tempfile
import pandas as pd
import networkx as nx
import requests

# Box direct download URLs (pre-configured)
BOX_URLS = {
    "regulatory_edges.csv": (
        "https://uofi.box.com/shared/static/8v3brw6yz62v9m9dvfffpe7rhx2zspjw.csv"
    ),
    "tflink_edges.csv": (
        "https://uofi.box.com/shared/static/8eip7umror9j3eyxy1bikc6hkgr53ku9.csv"
    ),
    "graph_metadata.json": (
        "https://uofi.box.com/shared/static/vhe6n0qvr94t92sdf8dhw3l8npo0qi9a.json"
    ),
    "regulatory_nodes.csv": (
        "https://uofi.box.com/shared/static/zq5q200sm5ii4a42569gny8rcnqs94jf.csv"
    ),
    "tflink_nodes.csv": (
        "https://uofi.box.com/shared/static/16737p916gglmpa7g9q7vryb7n6n9f6t.csv"
    ),
}


def download_file(url, local_path):
    """Download a file from URL to local path."""
    response = requests.get(url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def load_graphs():
    """
    Load TFLink and Regulatory graphs from Box URLs.

    Returns:
        dict: Dictionary with 'tflink' and 'regulatory' networkx graphs
    """
    # Create temporary directory for downloads
    temp_dir = tempfile.mkdtemp()
    print(f"Downloading files to temporary directory...")

    # Download all files
    local_files = {}
    for filename, url in BOX_URLS.items():
        local_path = osp.join(temp_dir, filename)
        print(f"  Downloading {filename}...")
        download_file(url, local_path)
        local_files[filename] = local_path

    # Load metadata
    with open(local_files["graph_metadata.json"], "r") as f:
        metadata = json.load(f)

    graphs = {}

    # Load TFLink graph
    print("\nBuilding TFLink graph...")
    if metadata["tflink_graph"]["is_directed"]:
        G_tflink = nx.DiGraph()
    else:
        G_tflink = nx.Graph()

    # Add nodes
    tflink_nodes_df = pd.read_csv(local_files["tflink_nodes.csv"])
    for _, row in tflink_nodes_df.iterrows():
        node = row["gene"]
        attrs = row.drop("gene").to_dict()
        attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G_tflink.add_node(node, **attrs)

    # Add edges
    tflink_edges_df = pd.read_csv(local_files["tflink_edges.csv"])
    for _, row in tflink_edges_df.iterrows():
        source = row["source_gene"]
        target = row["target_gene"]
        attrs = row.drop(["source_gene", "target_gene"]).to_dict()
        attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G_tflink.add_edge(source, target, **attrs)

    graphs["tflink"] = G_tflink
    print(
        f"  TFLink: {G_tflink.number_of_nodes()} genes, "
        f"{G_tflink.number_of_edges()} interactions"
    )

    # Load Regulatory graph
    print("\nBuilding Regulatory graph...")
    if metadata["regulatory_graph"]["is_directed"]:
        G_regulatory = nx.DiGraph()
    else:
        G_regulatory = nx.Graph()

    # Add nodes
    regulatory_nodes_df = pd.read_csv(local_files["regulatory_nodes.csv"])
    for _, row in regulatory_nodes_df.iterrows():
        node = row["gene"]
        attrs = row.drop("gene").to_dict()
        attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G_regulatory.add_node(node, **attrs)

    # Add edges
    regulatory_edges_df = pd.read_csv(local_files["regulatory_edges.csv"])
    for _, row in regulatory_edges_df.iterrows():
        source = row["regulator_gene"]
        target = row["target_gene"]
        attrs = row.drop(["regulator_gene", "target_gene"]).to_dict()
        attrs = {k: v for k, v in attrs.items() if pd.notna(v)}
        G_regulatory.add_edge(source, target, **attrs)

    graphs["regulatory"] = G_regulatory
    print(
        f"  Regulatory: {G_regulatory.number_of_nodes()} genes, "
        f"{G_regulatory.number_of_edges()} interactions"
    )

    return graphs


def show_basic_stats(graphs):
    """
    Show basic statistics about the graphs.

    Args:
        graphs: Dictionary with networkx graphs
    """
    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    for name, G in graphs.items():
        print(f"\n{name.upper()} Network:")
        print(f"  • {G.number_of_nodes():,} genes")
        print(f"  • {G.number_of_edges():,} interactions")
        print(f"  • {'Directed' if G.is_directed() else 'Undirected'} graph")

        # Show top connected genes
        if G.is_directed():
            out_degrees = dict(G.out_degree())
            if out_degrees:
                top_regulators = sorted(
                    out_degrees.items(), key=lambda x: x[1], reverse=True
                )[:3]
                print(
                    f"  • Top regulators: {', '.join([f'{g} ({d})' for g, d in top_regulators])}"
                )
        else:
            degrees = dict(G.degree())
            if degrees:
                top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
                print(
                    f"  • Top hub genes: {', '.join([f'{g} ({d})' for g, d in top_hubs])}"
                )


def save_for_later(graphs):
    """
    Save graphs as pickle files for faster loading next time.

    Args:
        graphs: Dictionary with networkx graphs
    """
    import pickle

    save_dir = "./graph_data"
    os.makedirs(save_dir, exist_ok=True)

    for name, G in graphs.items():
        output_file = osp.join(save_dir, f"{name}_graph.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(G, f)

    print(f"\nGraphs saved to '{save_dir}' folder for faster loading next time.")


def load_saved_graphs():
    """
    Try to load previously saved graphs.

    Returns:
        dict or None: Graphs if found, None otherwise
    """
    import pickle

    save_dir = "./graph_data"
    if not osp.exists(save_dir):
        return None

    graphs = {}
    for name in ["tflink", "regulatory"]:
        filepath = osp.join(save_dir, f"{name}_graph.pkl")
        if osp.exists(filepath):
            with open(filepath, "rb") as f:
                graphs[name] = pickle.load(f)

    if len(graphs) == 2:
        print("Loading from saved files...")
        for name, G in graphs.items():
            print(
                f"  {name}: {G.number_of_nodes()} genes, "
                f"{G.number_of_edges()} interactions"
            )
        return graphs

    return None


# Simple usage example at the bottom
def main():
    """Main function - simple workflow."""
    print("\n" + "=" * 60)
    print("S. cerevisiae Regulatory Network Loader")
    print("=" * 60)

    # Check for saved files first
    saved_graphs = load_saved_graphs()

    if saved_graphs:
        response = input("\nFound saved graphs. Use these? (y/n): ")
        if response.lower() == "y":
            graphs = saved_graphs
        else:
            print("\nDownloading fresh data from Box...")
            graphs = load_graphs()
    else:
        print("\nDownloading data from Box (first time setup)...")
        graphs = load_graphs()

    # Show statistics
    show_basic_stats(graphs)

    # Offer to save for next time
    if not saved_graphs:
        response = input("\nSave graphs for faster loading next time? (y/n): ")
        if response.lower() == "y":
            save_for_later(graphs)

    print("\n✓ Graphs loaded successfully!")
    print("\nTo use in Python:")
    print("  tflink_graph = graphs['tflink']")
    print("  regulatory_graph = graphs['regulatory']")

    return graphs


# Example of how to use the graphs
def example_usage():
    """Show how to use the loaded graphs."""
    # Load the graphs
    graphs = main()

    # Example: Find all targets of a transcription factor
    tf_name = "YAP1"  # Example TF
    if tf_name in graphs["tflink"]:
        targets = list(graphs["tflink"].successors(tf_name))
        print(f"\n{tf_name} regulates {len(targets)} genes")
        print(f"First 5 targets: {targets[:5]}")

    # Example: Find regulatory interactions
    reg_edges = list(graphs["regulatory"].edges())[:5]
    print(f"\nFirst 5 regulatory interactions:")
    for source, target in reg_edges:
        print(f"  {source} → {target}")


if __name__ == "__main__":
    graphs = main()
    # Uncomment to see usage examples:
    # example_usage()
