import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix
import multiprocessing
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, degree
from scipy.sparse.csgraph import reverse_cuthill_mckee, connected_components, dijkstra
from scipy.sparse import csr_matrix
import multiprocessing
import numpy as np
import networkx as nx
import time


# Fix multiprocessing issue with proper main guard
def load_sample_data_batch():
    import os
    import os.path as osp
    from dotenv import load_dotenv
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.datamodules import CellDataModule
    from torchcell.datamodels.fitness_composite_conversion import (
        CompositeFitnessConverter,
    )
    from torchcell.datasets import CodonFrequencyDataset
    from torchcell.data import MeanExperimentDeduplicator
    from torchcell.data import GenotypeAggregator
    from torchcell.datamodules.perturbation_subset import PerturbationSubsetDataModule
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
    from torchcell.data import Neo4jCellDataset
    from torchcell.data.neo4j_cell import SubgraphRepresentation
    from torchcell.metabolism.yeast_GEM import YeastGEM

    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    print(f"DATA_ROOT: {DATA_ROOT}")

    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        data_root=osp.join(DATA_ROOT, "data/sgd/genome"), genome=genome
    )
    selected_node_embeddings = ["codon_frequency"]
    node_embeddings = {}
    if "codon_frequency" in selected_node_embeddings:
        node_embeddings["codon_frequency"] = CodonFrequencyDataset(
            root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
            genome=genome,
        )

    with open("experiments/003-fit-int/queries/001-small-build.cql", "r") as f:
        query = f.read()
    dataset_root = osp.join(
        DATA_ROOT, "data/torchcell/experiments/003-fit-int/001-small-build"
    )
    gem = YeastGEM(root=osp.join(DATA_ROOT, "data/torchcell/yeast_gem"))
    reaction_map = gem.reaction_map

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        graphs={"physical": graph.G_physical, "regulatory": graph.G_regulatory},
        incidence_graphs={"metabolism": reaction_map},
        node_embeddings=node_embeddings,
        converter=CompositeFitnessConverter,
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,
        graph_processor=SubgraphRepresentation(),
    )
    seed = 42

    # Base Module
    cell_data_module = CellDataModule(
        dataset=dataset,
        cache_dir=osp.join(dataset_root, "data_module_cache"),
        split_indices=["phenotype_label_index", "perturbation_count_index"],
        batch_size=2,
        random_seed=seed,
        num_workers=0,  # Changed from 8 to 0 to avoid multiprocessing issues
        pin_memory=False,
    )
    cell_data_module.setup()

    # 1e4 Module
    size = 5e4
    perturbation_subset_data_module = PerturbationSubsetDataModule(
        cell_data_module=cell_data_module,
        size=int(size),
        batch_size=2,
        num_workers=0,  # Changed from 8 to 0 to avoid multiprocessing issues
        pin_memory=True,
        prefetch=False,
        seed=seed,
    )
    perturbation_subset_data_module.setup()
    max_num_nodes = len(dataset.gene_set)

    # Get first batch without tqdm to avoid multiprocessing issues
    batch = next(iter(perturbation_subset_data_module.train_dataloader()))

    input_channels = dataset.cell_graph["gene"].x.size()[-1]
    return dataset, batch, input_channels, max_num_nodes


def calculate_graph_statistics(edge_index, num_nodes, is_directed=False, name=""):
    """Calculate various graph statistics with progress updates."""
    print(f"Calculating statistics for {name} network...")
    start_time = time.time()

    print(f"  Creating NetworkX graph...")
    # Create networkx graph
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes and edges
    print(f"  Adding {num_nodes} nodes to graph...")
    for i in range(num_nodes):
        G.add_node(i)

    print(f"  Adding {edge_index.shape[1]} edges to graph...")
    edges = edge_index.t().numpy()
    for src, dst in edges:
        G.add_edge(int(src), int(dst))

    print(
        f"  Graph construction complete. Time: {time.time() - start_time:.2f} seconds"
    )

    # Calculate statistics
    stats = {}
    stats["num_nodes"] = num_nodes
    stats["num_edges"] = len(edges)

    # Node degree statistics
    print(f"  Calculating degree statistics...")
    degrees = [d for _, d in G.degree()]
    stats["avg_degree"] = np.mean(degrees)
    stats["max_degree"] = max(degrees)
    stats["min_degree"] = min(degrees)

    # Connected components
    print(f"  Finding connected components...")
    if is_directed:
        stats["num_connected_components"] = nx.number_weakly_connected_components(G)
        print(
            f"  Found {stats['num_connected_components']} weakly connected components"
        )
    else:
        stats["num_connected_components"] = nx.number_connected_components(G)
        print(f"  Found {stats['num_connected_components']} connected components")

    # Try to calculate diameter and avg path length for largest component
    print(f"  Finding largest connected component...")
    try:
        if is_directed:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            print(f"  Largest component has {len(largest_cc)} nodes")
            stats["largest_component_size"] = len(largest_cc)
            stats["largest_component_pct"] = len(largest_cc) / num_nodes * 100

            # For directed networks, we don't calculate these metrics by default
            # since they're computationally expensive and often not meaningful
            print(f"  Skipping diameter and path length for directed network")
            # Don't set diameter or avg_path_length keys for directed networks

        else:
            # Code for undirected networks
            largest_cc = max(nx.connected_components(G), key=len)
            print(f"  Largest component has {len(largest_cc)} nodes")
            stats["largest_component_size"] = len(largest_cc)
            stats["largest_component_pct"] = len(largest_cc) / num_nodes * 100

            largest_subgraph = G.subgraph(largest_cc)

            print(
                f"  Calculating diameter (this may take a while for large networks)..."
            )
            diameter_start = time.time()
            # Limit diameter calculation time for very large graphs
            if len(largest_cc) < 10000:  # Adjust threshold as needed
                stats["diameter"] = nx.diameter(largest_subgraph)
                print(
                    f"  Diameter: {stats['diameter']} (took {time.time() - diameter_start:.2f} seconds)"
                )
            else:
                print(
                    f"  Network too large for exact diameter calculation, using approximation..."
                )
                # Consider using an approximation algorithm for large networks
                stats["diameter"] = "Large network (>10k nodes), calculation skipped"

            print(f"  Calculating average path length...")
            path_start = time.time()
            if len(largest_cc) < 10000:  # Adjust threshold as needed
                stats["avg_path_length"] = nx.average_shortest_path_length(
                    largest_subgraph
                )
                print(
                    f"  Average path length: {stats['avg_path_length']:.2f} (took {time.time() - path_start:.2f} seconds)"
                )
            else:
                stats["avg_path_length"] = (
                    "Large network (>10k nodes), calculation skipped"
                )

            # Clustering coefficient (only for undirected)
            print(f"  Calculating clustering coefficient...")
            cluster_start = time.time()
            if num_nodes < 10000:  # Adjust threshold as needed
                stats["avg_clustering"] = nx.average_clustering(G)
                print(
                    f"  Average clustering: {stats['avg_clustering']:.4f} (took {time.time() - cluster_start:.2f} seconds)"
                )
            else:
                print(
                    f"  Network too large for clustering calculation, using approximation..."
                )
                # Approximate clustering by sampling nodes
                sampled_nodes = np.random.choice(
                    list(G.nodes()), size=min(1000, num_nodes), replace=False
                )
                stats["avg_clustering"] = nx.average_clustering(G, nodes=sampled_nodes)
                print(
                    f"  Approximate avg clustering (1000 sampled nodes): {stats['avg_clustering']:.4f}"
                )
    except Exception as e:
        print(f"  Error calculating path-based metrics: {str(e)}")
        stats["largest_component_size"] = "Error"
        # Don't set diameter or avg_path_length keys if there was an error

    print(
        f"Statistics calculation complete for {name}. Total time: {time.time() - start_time:.2f} seconds"
    )
    return stats


def plot_adjacency_matrix(adj_matrix, title, stats=None, filename=None):
    """Plot a single adjacency matrix with statistics box properly contained within figure."""
    print(f"Creating plot for: {title}")
    plot_start = time.time()

    # Create figure with adjusted layout to leave room for stats
    fig = plt.figure(figsize=(14, 10))  # Wider figure to accommodate stats box

    # Create a gridspec for layout control
    gs = fig.add_gridspec(1, 20)  # 20 columns for fine control

    # Create main axis for the matrix plot (using 16 of 20 columns)
    ax_main = fig.add_subplot(gs[0, :16])

    print("  Rendering adjacency matrix image...")
    ax_main.imshow(adj_matrix, cmap="Greys", interpolation="none")
    ax_main.set_title(title, fontsize=16, pad=20)

    # Add statistics text in a separate axis
    if stats:
        print("  Adding statistics to plot...")
        # Create an invisible axis for the stats box
        ax_stats = fig.add_subplot(gs[0, 16:])
        ax_stats.axis("off")  # Hide axis

        # Create list of statistics, checking if each key exists
        stat_lines = [
            f"Nodes: {stats['num_nodes']:,}",
            f"Edges: {stats['num_edges']:,}",
            f"Avg degree: {stats['avg_degree']:.2f}",
        ]

        # Add max degree if available
        if "max_degree" in stats:
            stat_lines.append(f"Max degree: {stats['max_degree']}")

        # Add connected components if available
        if "num_connected_components" in stats:
            stat_lines.append(
                f"Connected components: {stats['num_connected_components']}"
            )

        # Add largest component info if available
        if "largest_component_size" in stats:
            if isinstance(stats["largest_component_size"], int):
                stat_lines.append(
                    f"Largest component: {stats['largest_component_size']} nodes "
                    f"({stats['largest_component_pct']:.1f}%)"
                )
            else:
                stat_lines.append(
                    f"Largest component: {stats['largest_component_size']}"
                )

        # Add diameter if available
        if "diameter" in stats:
            stat_lines.append(f"Diameter: {stats['diameter']}")

        # Add avg path length if available
        if (
            "avg_path_length" in stats
            and stats["avg_path_length"] != "N/A"
            and stats["avg_path_length"] != "Error"
        ):
            if isinstance(stats["avg_path_length"], float):
                stat_lines.append(f"Avg path length: {stats['avg_path_length']:.2f}")
            else:
                stat_lines.append(f"Avg path length: {stats['avg_path_length']}")

        # Add clustering if available (typically only for undirected)
        if "avg_clustering" in stats:
            if isinstance(stats["avg_clustering"], float):
                stat_lines.append(f"Avg clustering: {stats['avg_clustering']:.4f}")
            else:
                stat_lines.append(f"Avg clustering: {stats['avg_clustering']}")

        # Join all lines with newlines
        stat_text = "Network Statistics:\n" + "\n".join(stat_lines)

        # Add the text in the stats axis, properly contained
        ax_stats.text(
            0.05,
            0.95,
            stat_text,
            transform=ax_stats.transAxes,  # Use axis coordinates
            verticalalignment="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
        )

    plt.tight_layout()

    if filename:
        print(f"  Saving plot to {filename}...")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  Plot saved successfully to {filename}")

    # Close the figure to free memory without displaying
    plt.close()

    print(f"  Plot processing complete. Time: {time.time() - plot_start:.2f} seconds")


def process_network(edge_index, max_num_nodes, name, is_directed=False):
    """Process a network: calculate stats and plot original and reordered matrices."""
    print(f"\n{'='*80}\nProcessing {name} network\n{'='*80}")
    process_start = time.time()

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
    plot_adjacency_matrix(
        adj_original,
        f"Original {name} Adjacency Matrix ({'Directed' if is_directed else 'Undirected'})",
        stats,
        f"original_{name.lower()}_matrix.png",
    )

    plot_adjacency_matrix(
        adj_reordered,
        f"RCM Reordered {name} Adjacency Matrix ({'Directed' if is_directed else 'Undirected'})",
        stats,
        f"reordered_{name.lower()}_matrix.png",
    )

    print(
        f"{name} network processing complete. Total time: {time.time() - process_start:.2f} seconds"
    )


if __name__ == "__main__":

    # Proper multiprocessing guard
    multiprocessing.freeze_support()

    overall_start = time.time()
    print(f"Starting graph analysis... {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    dataset, batch, input_channels, max_num_nodes = load_sample_data_batch()

    print(f"\nData loaded successfully:")
    print(f"- Number of nodes: {max_num_nodes:,}")
    print(f"- Input feature dimensions: {input_channels}")

    # Process PPI network (physical interactions)
    ppi_edge_index = batch[("gene", "physical_interaction", "gene")].edge_index
    process_network(ppi_edge_index, max_num_nodes, "PPI", is_directed=False)

    # Process REG network (regulatory interactions)
    reg_edge_index = batch[("gene", "regulatory_interaction", "gene")].edge_index
    process_network(reg_edge_index, max_num_nodes, "REG", is_directed=True)

    # Print summary
    print(f"\n{'='*80}")
    print(
        f"Analysis complete! Total execution time: {(time.time() - overall_start)/60:.2f} minutes"
    )
    print(f"PPI: {ppi_edge_index.shape[1]:,} edges among {max_num_nodes:,} nodes")
    print(f"REG: {reg_edge_index.shape[1]:,} edges among {max_num_nodes:,} nodes")
    print(f"{'='*80}")
