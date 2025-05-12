# experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination_string11_0_to_string12_0
# [[experiments.005-kuzmin2018-tmi.scripts.dango_lambda_determination_string11_0_to_string12_0]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination_string11_0_to_string12_0
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_dango_lambda_determination_string11_0_to_string12_0.py



"""
Apply the original DANGO lambda determination strategy to STRING v11.0 to v12.0 transition.

This script calculates the lambda values for STRING v11.0 networks using the original
strategy from the DANGO paper, applied to the v11.0 to v12.0 transition.
"""

import os
import os.path as osp
import logging
import numpy as np
import matplotlib.pyplot as plt
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.graph.graph import SCerevisiaeGraph
from torchcell.timestamp import timestamp
import torchcell

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = os.environ.get(
    "DATA_ROOT", osp.expanduser("~/Documents/projects/torchcell")
)
ASSET_IMAGES_DIR = os.environ.get("ASSET_IMAGES_DIR", "assets/images")

# Apply torchcell matplotlib style
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def load_gene_graphs():
    """Load the SCerevisiaeGraph with STRING graphs loaded."""
    # Create genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        go_root=osp.join(DATA_ROOT, "data/go"),
        overwrite=False,  # Use existing data to avoid rebuilding
    )

    # Create graph with STRING versions
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(DATA_ROOT, "data/sgd/genome"),
        string_root=osp.join(DATA_ROOT, "data/string"),
        tflink_root=osp.join(DATA_ROOT, "data/tflink"),
        genome=genome,
    )

    return graph


def calculate_zero_decrease_percentage(graph, network_type):
    """
    Calculate the percentage of decreased zeros from STRING v11.0 to v12.0 for a specific network type.

    The calculation is based on the formula in the DANGO paper:
    Percentage of decreased zeros = (Number of new edges in v12.0) / (Number of zeros in v11.0) * 100

    Args:
        graph: SCerevisiaeGraph instance
        network_type: One of the network types (neighborhood, fusion, cooccurence, coexpression, experimental, database)

    Returns:
        dict: Dictionary with all calculated metrics
    """
    # Get the graphs for both versions
    graph_v11 = getattr(graph, f"G_string11_0_{network_type}")
    graph_v12 = getattr(graph, f"G_string12_0_{network_type}")

    # Get the edges for both versions
    edges_v11 = set(graph_v11.graph.edges())
    edges_v12 = set(graph_v12.graph.edges())

    # Calculate the number of nodes - use the intersection of nodes that are in both networks
    nodes_v11 = set(graph_v11.graph.nodes())
    nodes_v12 = set(graph_v12.graph.nodes())
    common_nodes = nodes_v11.intersection(nodes_v12)

    # Filter edges to only include those between common nodes
    filtered_edges_v11 = {
        (u, v) for u, v in edges_v11 if u in common_nodes and v in common_nodes
    }
    filtered_edges_v12 = {
        (u, v) for u, v in edges_v12 if u in common_nodes and v in common_nodes
    }

    num_nodes = len(common_nodes)

    # Calculate the number of possible edges (excluding self-loops)
    num_possible_edges = (num_nodes * (num_nodes - 1)) // 2

    # Calculate the number of edges in each version
    num_edges_v11 = len(filtered_edges_v11)
    num_edges_v12 = len(filtered_edges_v12)

    # Calculate new edges in v12 (edges in v12 but not in v11)
    new_edges_v12 = filtered_edges_v12 - filtered_edges_v11
    num_new_edges = len(new_edges_v12)

    # Calculate the number of zeros in v11 (possible edges - actual edges)
    num_zeros_v11 = num_possible_edges - num_edges_v11

    # Calculate the percentage of decreased zeros (new interactions that were previously zeros)
    if num_zeros_v11 == 0:
        zero_decrease_pct = 0.0
    else:
        zero_decrease_pct = (num_new_edges / num_zeros_v11) * 100

    # Log details
    log.info(f"Network type: {network_type}")
    log.info(f"Common nodes across versions: {num_nodes}")
    log.info(f"Possible edges: {num_possible_edges}")
    log.info(f"Filtered edges in v11.0: {num_edges_v11}")
    log.info(f"Filtered edges in v12.0: {num_edges_v12}")
    log.info(f"New edges in v12.0: {num_new_edges}")
    log.info(f"Zero edges in v11.0: {num_zeros_v11}")
    log.info(f"Percentage of decreased zeros: {zero_decrease_pct:.4f}%")

    # Return all metrics as a dictionary for later use
    return {
        "network_type": network_type,
        "num_nodes": num_nodes,
        "num_possible_edges": num_possible_edges,
        "num_edges_v11": num_edges_v11,
        "num_edges_v12": num_edges_v12,
        "num_new_edges": num_new_edges,
        "num_zeros_v11": num_zeros_v11,
        "zero_decrease_pct": zero_decrease_pct,
    }


def plot_string_comparison(metrics_list):
    """
    Plot a comparison of STRING v11.0 and v12.0 networks.
    Shows edge counts, percentage of decreased zeros, and determined lambda values.

    Args:
        metrics_list: List of dictionaries containing metrics for each network type

    Returns:
        str: Path to the saved plot
    """
    # Ensure the directory exists
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    # Extract data for plotting
    network_types = []
    edges_v11 = []
    edges_v12 = []
    new_edges = []
    decrease_pcts = []
    lambda_values = []

    for metrics in metrics_list:
        network_types.append(metrics["network_type"])
        edges_v11.append(metrics["num_edges_v11"])
        edges_v12.append(metrics["num_edges_v12"])
        new_edges.append(metrics["num_new_edges"])
        decrease_pcts.append(metrics["zero_decrease_pct"])
        # Determine lambda value according to DANGO paper criterion
        # "Greater than 1% zeros decreased" -> lambda=0.1, otherwise lambda=1.0
        lambda_values.append(0.1 if metrics["zero_decrease_pct"] > 1.0 else 1.0)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [3, 2, 1]}
    )

    # Get TorchCell color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # Use colors from the torchcell color cycle
    v11_color = colors[1]
    v12_color = colors[2]
    threshold_color = colors[0]
    bar_color = colors[4]
    lambda_color = colors[3]

    # Bar width
    width = 0.35

    # Set positions
    x = np.arange(len(network_types))

    # Plot edge counts with custom colors
    bars1 = ax1.bar(
        x - width / 2, edges_v11, width, label="STRING v11.0 edges", color=v11_color
    )
    bars2 = ax1.bar(
        x + width / 2, edges_v12, width, label="STRING v12.0 edges", color=v12_color
    )

    # Calculate appropriate top margin for y-axis
    max_height = max(max(edges_v11), max(edges_v12))
    ax1.set_ylim(0, max_height * 1.2)  # Add 20% padding on top

    # Add data labels with larger font size and better positioning
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
        )

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
        )

    # Set labels and titles for edge count plot
    ax1.set_ylabel("Number of Edges", fontsize=14)
    ax1.set_title("Comparison of Edge Counts: STRING v11.0 vs v12.0", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(network_types, fontsize=12, rotation=30, ha="right")
    ax1.legend(fontsize=12)

    # Format y-axis with commas for thousands
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))

    # Add a grid
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot percentage of decreased zeros with custom color
    bars3 = ax2.bar(x, decrease_pcts, width * 1.5, color=bar_color)

    # Calculate appropriate top margin for the percentage plot
    max_pct = max(decrease_pcts)
    y_max = max(10, max_pct * 1.3)  # At least 10%, or 30% more than max value
    ax2.set_ylim(0, y_max)

    # Draw the 1% threshold line but make it less prominent
    threshold_line = ax2.axhline(
        y=1.0,
        color=threshold_color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.4,
        label="1% threshold for λ determination",
    )

    # Add data labels with appropriate positioning to avoid threshold
    for bar in bars3:
        height = bar.get_height()

        # Calculate vertical offset based on proximity to threshold
        if 0.8 < height < 1.3:
            # Value is close to the threshold - adjust position
            if height >= 1.0:
                offset = 18  # Move up for values above threshold
            else:
                offset = -18  # Move down for values below threshold
        else:
            offset = 5  # Default offset for values not near threshold

        # Add label with white background for better visibility
        ax2.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if offset > 0 else "top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, pad=2, boxstyle="round,pad=0.2"),
        )

    # Set labels and titles for percentage plot
    ax2.set_ylabel("Decreased Zeros (%)", fontsize=14)
    ax2.set_title(
        "Percentage of Decreased Zeros from STRING v11.0 to v12.0", fontsize=16
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(network_types, fontsize=12, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Plot lambda values
    bars4 = ax3.bar(x, lambda_values, width * 1.5, color=lambda_color)

    # Add data labels
    for bar in bars4:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Set labels and titles for lambda values plot
    ax3.set_ylabel("Lambda Value", fontsize=14)
    ax3.set_xlabel("Network Type", fontsize=14)
    ax3.set_title("Determined Lambda Values for STRING v11.0 Networks", fontsize=16)
    ax3.set_xticks(x)
    ax3.set_xticklabels(network_types, fontsize=12, rotation=30, ha="right")
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, linestyle="--", alpha=0.7)

    # Add more space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Ensure tight layout
    plt.tight_layout()

    # Save the figure with timestamp
    current_timestamp = timestamp()
    filename = f"string_v11.0_vs_v12.0_comparison_{current_timestamp}.png"
    filepath = os.path.join(ASSET_IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    log.info(f"Plot saved to {filepath}")

    return filepath


def determine_lambda_values(verbose=True):
    """
    Calculate lambda values for STRING v11.0 networks based on comparison with v12.0.
    Uses the original DANGO strategy: > 1% decrease = 0.1, otherwise = 1.0.

    Args:
        verbose: Whether to print detailed logs and generate plots

    Returns:
        dict: Dictionary mapping network types to lambda values
    """
    if verbose:
        log.info("Loading gene graphs...")

    graph = load_gene_graphs()

    # Network types to analyze
    network_types = [
        "neighborhood",
        "fusion",
        "cooccurence",
        "coexpression",
        "experimental",
        "database",
    ]

    # Calculate zero decrease percentage for each network type
    metrics_list = []
    lambda_values = {}

    if verbose:
        log.info("Calculating zero decrease percentages...")

    for network_type in network_types:
        if verbose:
            log.info(f"\nAnalyzing {network_type} network...")

        metrics = calculate_zero_decrease_percentage(graph, network_type)
        metrics_list.append(metrics)

        # Determine lambda value according to DANGO paper criterion
        # "Greater than 1% zeros decreased" -> lambda=0.1, otherwise lambda=1.0
        percentage = metrics["zero_decrease_pct"]
        if percentage > 1.0:
            lambda_values[f"string11_0_{network_type}"] = 0.1
        else:
            lambda_values[f"string11_0_{network_type}"] = 1.0

    if verbose:
        # Print summary
        log.info("\n--- SUMMARY OF ZERO DECREASE PERCENTAGES ---")
        for metrics in metrics_list:
            network_type = metrics["network_type"]
            percentage = metrics["zero_decrease_pct"]
            log.info(f"{network_type}: {percentage:.4f}%")

        log.info("\n--- DETERMINED LAMBDA VALUES FOR STRING V11.0 ---")
        for network, lambda_val in lambda_values.items():
            log.info(f"{network}: {lambda_val}")

        # Generate and save the plot
        plot_path = plot_string_comparison(metrics_list)
        log.info(f"Generated plot at: {plot_path}")

    # Return the lambda values to be used in the training script
    return lambda_values


def main():
    """Main function to calculate lambda values and generate visualization."""
    lambda_values = determine_lambda_values(verbose=True)
    return lambda_values


if __name__ == "__main__":
    main()
