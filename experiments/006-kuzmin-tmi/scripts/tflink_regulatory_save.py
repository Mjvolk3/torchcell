#!/usr/bin/env python
# tflink_regulatory_save.py
# Script to save TFLink and Regulatory graphs to human-readable CSV format

import os
import os.path as osp
import pandas as pd
import networkx as nx
import json
from dotenv import load_dotenv
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph.graph import SCerevisiaeGraph
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def save_graphs_to_csv(
    output_dir="/Users/michaelvolk/Library/CloudStorage/Box-Box/AAA Lab Research/MV-JC",
):
    """
    Save TFLink and Regulatory graphs to CSV files for easy sharing.
    """
    # Load environment variables
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")

    # Create genome and graph
    log.info("Loading genome and graph data...")
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,
    )

    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save TFLink graph
    log.info("Processing TFLink graph...")
    tflink_graph = graph.G_tflink.graph

    # Extract edges with attributes
    tflink_edges = []
    for source, target, attrs in tflink_graph.edges(data=True):
        edge_data = {"source_gene": source, "target_gene": target}
        # Add all edge attributes
        edge_data.update(attrs)
        tflink_edges.append(edge_data)

    tflink_df = pd.DataFrame(tflink_edges)
    tflink_file = osp.join(output_dir, "tflink_edges.csv")
    tflink_df.to_csv(tflink_file, index=False)
    log.info(f"Saved TFLink graph: {len(tflink_df)} edges to {tflink_file}")

    # Save node information for TFLink
    tflink_nodes = []
    for node, attrs in tflink_graph.nodes(data=True):
        node_data = {"gene": node}
        node_data.update(attrs)
        tflink_nodes.append(node_data)

    if tflink_nodes:
        tflink_nodes_df = pd.DataFrame(tflink_nodes)
        tflink_nodes_file = osp.join(output_dir, "tflink_nodes.csv")
        tflink_nodes_df.to_csv(tflink_nodes_file, index=False)
        log.info(
            f"Saved TFLink nodes: {len(tflink_nodes_df)} nodes to {tflink_nodes_file}"
        )

    # Save Regulatory graph
    log.info("Processing Regulatory graph...")
    regulatory_graph = graph.G_regulatory.graph

    # Extract edges with attributes
    regulatory_edges = []
    for source, target, attrs in regulatory_graph.edges(data=True):
        edge_data = {"regulator_gene": source, "target_gene": target}
        # Add all edge attributes
        edge_data.update(attrs)
        regulatory_edges.append(edge_data)

    regulatory_df = pd.DataFrame(regulatory_edges)
    regulatory_file = osp.join(output_dir, "regulatory_edges.csv")
    regulatory_df.to_csv(regulatory_file, index=False)
    log.info(f"Saved Regulatory graph: {len(regulatory_df)} edges to {regulatory_file}")

    # Save node information for Regulatory
    regulatory_nodes = []
    for node, attrs in regulatory_graph.nodes(data=True):
        node_data = {"gene": node}
        node_data.update(attrs)
        regulatory_nodes.append(node_data)

    if regulatory_nodes:
        regulatory_nodes_df = pd.DataFrame(regulatory_nodes)
        regulatory_nodes_file = osp.join(output_dir, "regulatory_nodes.csv")
        regulatory_nodes_df.to_csv(regulatory_nodes_file, index=False)
        log.info(
            f"Saved Regulatory nodes: {len(regulatory_nodes_df)} nodes to {regulatory_nodes_file}"
        )

    # Create metadata file
    metadata = {
        "tflink_graph": {
            "num_nodes": tflink_graph.number_of_nodes(),
            "num_edges": tflink_graph.number_of_edges(),
            "is_directed": tflink_graph.is_directed(),
            "edge_file": "tflink_edges.csv",
            "node_file": "tflink_nodes.csv",
        },
        "regulatory_graph": {
            "num_nodes": regulatory_graph.number_of_nodes(),
            "num_edges": regulatory_graph.number_of_edges(),
            "is_directed": regulatory_graph.is_directed(),
            "edge_file": "regulatory_edges.csv",
            "node_file": "regulatory_nodes.csv",
        },
    }

    metadata_file = osp.join(output_dir, "graph_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved metadata to {metadata_file}")

    print("\nFiles created:")
    print(f"  - {tflink_file}")
    print(f"  - {osp.join(output_dir, 'tflink_nodes.csv')}")
    print(f"  - {regulatory_file}")
    print(f"  - {osp.join(output_dir, 'regulatory_nodes.csv')}")
    print(f"  - {metadata_file}")

    return output_dir




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Save TFLink and Regulatory graphs to CSV format"
    )
    parser.add_argument(
        "--dir",
        default="/Users/michaelvolk/Library/CloudStorage/Box-Box/AAA Lab Research/MV-JC",
        help="Directory for saving files (default: Box cloud storage)",
    )

    args = parser.parse_args()

    output_dir = save_graphs_to_csv(args.dir)
    print(f"\nGraphs saved to: {output_dir}")
    print("\nFiles created:")
    print("  - tflink_edges.csv")
    print("  - tflink_nodes.csv") 
    print("  - regulatory_edges.csv")
    print("  - regulatory_nodes.csv")
    print("  - graph_metadata.json")
    print("\nTo share these files:")
    print("1. Upload to Box and get direct download links")
    print("2. Share the tflink_regulatory_load.py script with colleagues")
    print("3. They can load the graphs using the direct Box URLs")
