# torchcell/experiments/003-fit-int/scripts/plot_ppi_reg_rcm
# [[torchcell.experiments.003-fit-int.scripts.plot_ppi_reg_rcm]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/experiments/003-fit-int/scripts/plot_ppi_reg_rcm
# Test file: tests/torchcell/experiments/003-fit-int/scripts/test_plot_ppi_reg_rcm.py

import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, contains_isolated_nodes
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
import time
import os
import os.path as osp
from dotenv import load_dotenv
from torch_geometric.utils import to_undirected, add_remaining_self_loops


def load_dataset():
    """Load just the dataset with cell_graph."""
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.data import MeanExperimentDeduplicator, GenotypeAggregator
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.data import Neo4jCellDataset
    from torchcell.data.neo4j_cell import SubgraphRepresentation
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    print(f"DATA_ROOT: {DATA_ROOT}")

    # Simple setup
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )

    # Load codon frequency embedding (just for completeness)
    node_embeddings = {
        "codon_frequency": CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )
    }

    # Setup dataset with Neo4jCellDataset
    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))

    # Create dataset
    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": gem.reaction_map},
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
        add_remaining_gene_self_loops=False,
    )

    return dataset


def calculate_graph_statistics(edge_index, num_nodes, is_directed=False, name=""):
    """Calculate graph statistics focusing on PyG representations."""
    print(f"Calculating statistics for {name} network...")
    start_time = time.time()

    # PyG representation stats
    stats = {}
    stats["num_nodes"] = num_nodes
    stats["pyg_num_edges"] = edge_index.size(1)  # Original edge count in PyG

    # Create original with self-loops using PyG
    self_loops_edge_index, _ = add_remaining_self_loops(
        edge_index.clone(), num_nodes=num_nodes
    )
    stats["pyg_self_loops_num_edges"] = self_loops_edge_index.size(1)

    # Create undirected version using PyG
    undirected_edge_index = to_undirected(edge_index.clone())
    stats["pyg_undirected_num_edges"] = undirected_edge_index.size(1)

    # Create undirected version with self-loops using PyG
    undirected_self_loops_edge_index, _ = add_remaining_self_loops(
        to_undirected(edge_index.clone()), num_nodes=num_nodes
    )
    stats["pyg_undirected_self_loops_num_edges"] = (
        undirected_self_loops_edge_index.size(1)
    )

    # Quick check for isolated nodes using PyG function
    has_isolated = contains_isolated_nodes(edge_index, num_nodes)
    print(f"  Network contains isolated nodes: {has_isolated}")

    # Create networkx graph for other statistics
    G = nx.DiGraph() if is_directed else nx.Graph()

    # Add all nodes and edges
    for i in range(num_nodes):
        G.add_node(i)

    edges = edge_index.t().numpy()
    for src, dst in edges:
        G.add_edge(int(src), int(dst))

    # Calculate basic statistics from NetworkX
    stats["nx_num_edges"] = G.number_of_edges()
    stats["density"] = nx.density(G)

    # Directed graph specific metrics
    if is_directed:
        # Create undirected version
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(G.nodes())
        G_undirected.add_edges_from(G.edges())
        undirected_edges = G_undirected.number_of_edges()
        stats["nx_undirected_edges"] = undirected_edges

        # Count bidirectional edges
        bidirectional_count = G.number_of_edges() - undirected_edges
        stats["bidirectional_edges"] = bidirectional_count
        stats["bidirectional_pct"] = (
            (bidirectional_count / undirected_edges) * 100
            if undirected_edges > 0
            else 0
        )

        # In-degree and out-degree statistics
        in_degrees = [d for _, d in G.in_degree()]
        out_degrees = [d for _, d in G.out_degree()]

        stats["avg_in_degree"] = np.mean(in_degrees)
        stats["max_in_degree"] = max(in_degrees)
        stats["avg_out_degree"] = np.mean(out_degrees)
        stats["max_out_degree"] = max(out_degrees)

        # Reciprocity (fraction of edges that are bidirectional)
        stats["reciprocity"] = nx.reciprocity(G)

    # Degree statistics
    degrees = [d for _, d in G.degree()]
    stats["avg_degree"] = np.mean(degrees)
    stats["max_degree"] = max(degrees)
    stats["min_degree"] = min(degrees)

    # Isolated nodes
    isolated_nodes = sum(1 for d in degrees if d == 0)
    stats["isolated_nodes"] = isolated_nodes
    stats["isolated_nodes_pct"] = (isolated_nodes / num_nodes) * 100

    # Connected components
    if is_directed:
        stats["num_connected_components"] = nx.number_weakly_connected_components(G)
        largest_cc = max(nx.weakly_connected_components(G), key=len)
    else:
        stats["num_connected_components"] = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len)

    stats["largest_component_size"] = len(largest_cc)
    stats["largest_component_pct"] = len(largest_cc) / num_nodes * 100

    return stats


def plot_adjacency_matrix(adj_matrix, title, stats, filename=None):
    """Plot adjacency matrix with PyG-focused statistics."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(1, 20)
    ax_main = fig.add_subplot(gs[0, :16])

    # Plot matrix
    ax_main.imshow(adj_matrix, cmap="Greys", interpolation="none")
    ax_main.set_title(title, fontsize=16, pad=20)

    # Add statistics
    ax_stats = fig.add_subplot(gs[0, 16:])
    ax_stats.axis("off")

    # Create statistics text
    stat_lines = [
        f"Nodes: {stats['num_nodes']:,}",
        f"",
        f"PyG Edge Counts:",
        f"Original: {stats['pyg_num_edges']:,}",
        f"Original+SelfLoops: {stats['pyg_self_loops_num_edges']:,}",
        f"Undirected: {stats['pyg_undirected_num_edges']:,}",
        f"Undirected+SelfLoops: {stats['pyg_undirected_self_loops_num_edges']:,}",
        f"",
        f"NetworkX Edge Counts:",
        f"Original: {stats['nx_num_edges']:,}",
    ]

    # Add directed graph specific stats
    if "nx_undirected_edges" in stats:
        stat_lines.extend(
            [
                f"Undirected: {stats['nx_undirected_edges']:,}",
                f"Bidirectional: {stats['bidirectional_edges']} ({stats['bidirectional_pct']:.1f}%)",
                f"Reciprocity: {stats['reciprocity']:.4f}",
            ]
        )

    # Add empty line
    stat_lines.append(f"")

    # Network properties
    stat_lines.extend(
        [
            f"Network Properties:",
            f"Density: {stats['density']:.5f}",
            f"Avg Degree: {stats['avg_degree']:.2f}",
            f"Max Degree: {stats['max_degree']}",
            f"Isolated Nodes: {stats['isolated_nodes']:,} ({stats['isolated_nodes_pct']:.1f}%)",
            f"Connected Components: {stats['num_connected_components']:,}",
            f"Largest Component: {stats['largest_component_size']:,} ({stats['largest_component_pct']:.1f}%)",
        ]
    )

    # Add directed graph degrees if available
    if "avg_in_degree" in stats:
        stat_lines.extend(
            [
                f"Avg In-Degree: {stats['avg_in_degree']:.2f}",
                f"Max In-Degree: {stats['max_in_degree']}",
                f"Avg Out-Degree: {stats['avg_out_degree']:.2f}",
                f"Max Out-Degree: {stats['max_out_degree']}",
            ]
        )

    stat_text = "Network Statistics:\n" + "\n".join(stat_lines)

    ax_stats.text(
        0.05,
        0.95,
        stat_text,
        transform=ax_stats.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
    )

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()


def process_network(
    edge_index, max_num_nodes, name, is_directed=False, output_dir=None
):
    """Process a network: calculate stats and plot original and reordered matrices."""
    print(f"\n{'='*80}\nProcessing {name} network\n{'='*80}")
    process_start = time.time()

    # If output_dir not specified, use current directory
    if output_dir is None:
        output_dir = ""

    print(f"Converting {name} network to dense adjacency matrix...")
    # Convert to dense adjacency matrix
    dense_adj = to_dense_adj(edge_index, max_num_nodes=max_num_nodes)[0]
    print(f"Dense adjacency matrix shape: {dense_adj.shape}")

    # Calculate statistics
    stats = calculate_graph_statistics(edge_index, max_num_nodes, is_directed, name)

    print(f"Preparing for RCM reordering...")
    # Convert to scipy sparse matrix for RCM reordering
    sparse_adj = csr_matrix(dense_adj.numpy())

    print(f"Computing RCM ordering for {name} network...")
    rcm_start = time.time()
    # Compute RCM ordering
    perm = torch.tensor(reverse_cuthill_mckee(sparse_adj).copy(), dtype=torch.long)
    print(f"RCM ordering complete. Time: {time.time() - rcm_start:.2f} seconds")

    print(f"Applying permutation to {name} adjacency matrix...")
    # Apply permutation
    reordered_adj = dense_adj[perm][:, perm]

    # Convert to numpy for plotting
    adj_original = dense_adj.numpy()
    adj_reordered = reordered_adj.numpy()

    # Plot original and reordered matrices
    original_filename = osp.join(output_dir, f"original_{name.lower()}_matrix.png")
    reordered_filename = osp.join(output_dir, f"reordered_{name.lower()}_matrix.png")

    plot_adjacency_matrix(
        adj_original,
        f"Original {name} Adjacency Matrix ({'Directed' if is_directed else 'Undirected'})",
        stats,
        original_filename,
    )

    plot_adjacency_matrix(
        adj_reordered,
        f"RCM Reordered {name} Adjacency Matrix ({'Directed' if is_directed else 'Undirected'})",
        stats,
        reordered_filename,
    )

    print(
        f"{name} network processing complete. Total time: {time.time() - process_start:.2f} seconds"
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import os.path as osp

    # Load environment variables
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    if not ASSET_IMAGES_DIR:
        print(
            "Warning: ASSET_IMAGES_DIR environment variable not set. Using current directory."
        )
        ASSET_IMAGES_DIR = "."

    # Create directory if it doesn't exist
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    start_time = time.time()
    print(f"Starting graph analysis... {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load dataset (only need the dataset with cell_graph)
    dataset = load_dataset()
    max_num_nodes = len(dataset.gene_set)

    print(f"\nData loaded successfully:")
    print(f"- Number of nodes: {max_num_nodes:,}")

    # Process physical interaction network
    ppi_edge_index = dataset.cell_graph[
        ("gene", "physical_interaction", "gene")
    ].edge_index
    process_network(
        ppi_edge_index,
        max_num_nodes,
        "PPI",
        is_directed=False,
        output_dir=ASSET_IMAGES_DIR,
    )

    # Process regulatory interaction network
    reg_edge_index = dataset.cell_graph[
        ("gene", "regulatory_interaction", "gene")
    ].edge_index
    process_network(
        reg_edge_index,
        max_num_nodes,
        "REG",
        is_directed=True,
        output_dir=ASSET_IMAGES_DIR,
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"Analysis complete! Total time: {(time.time() - start_time)/60:.2f} minutes")
    print(f"PPI: {ppi_edge_index.shape[1]:,} edges among {max_num_nodes:,} nodes")
    print(f"REG: {reg_edge_index.shape[1]:,} edges among {max_num_nodes:,} nodes")
    print(f"Images saved to: {ASSET_IMAGES_DIR}")
    print(f"{'='*80}")
