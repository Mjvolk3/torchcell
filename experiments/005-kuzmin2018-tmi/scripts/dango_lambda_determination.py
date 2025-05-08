# experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination.py
# [[experiments.005-kuzmin2018-tmi.scripts.dango_lambda_determination]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination.py
# Test file: experiments/005-kuzmin2018-tmi/scripts/test_dango_lambda_determination.py

"""
Calculate the percentage of decreased zeros from STRING v9.1 to v11.0 for each network type.
This is used to determine the lambda values for the DANGO model as described in the paper:
"DANGO: Predicting higher-order genetic interaction phenotypes using network representation learning"
"""

import os
import os.path as osp
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
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
load_dotenv()
ASSET_IMAGES_DIR = os.environ.get("ASSET_IMAGES_DIR", "assets/images")

# Apply torchcell matplotlib style
style_file_path = osp.join(osp.dirname(torchcell.__file__), "torchcell.mplstyle")
plt.style.use(style_file_path)


def load_gene_graphs():
    """Load the SCerevisiaeGraph with STRING graphs loaded."""
    # Set paths
    if "DATA_ROOT" in os.environ:
        data_root = os.environ["DATA_ROOT"]
    else:
        data_root = osp.expanduser("~/Documents/projects/torchcell")

    # Create genome
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,  # Use existing data to avoid rebuilding
    )

    # Create graph with both STRING versions
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )

    return graph


def calculate_zero_decrease_percentage(graph, network_type):
    """
    Calculate the percentage of decreased zeros from STRING v9.1 to v11.0 for a specific network type.

    The calculation is based on the formula in the DANGO paper:
    Percentage of decreased zeros = (Number of new edges in v11.0) / (Number of zeros in v9.1) * 100
    
    From the DANGO paper: "The percentage of decreased zeroes from STRING database v9.1 to v11.0 for each network,
    ranging from 0.02% (co-occurrence) to 2.42% (co-expression)."

    Args:
        graph: SCerevisiaeGraph instance
        network_type: One of the network types (neighborhood, fusion, cooccurence, coexpression, experimental, database)

    Returns:
        dict: Dictionary with all calculated metrics
    """
    # Get the graphs for both versions
    graph_v9 = getattr(graph, f"G_string9_1_{network_type}")
    graph_v11 = getattr(graph, f"G_string11_0_{network_type}")

    # Get the edges for both versions
    edges_v9 = set(graph_v9.graph.edges())
    edges_v11 = set(graph_v11.graph.edges())

    # Calculate the number of nodes - use the intersection of nodes that are in both networks
    # This is important as different STRING versions might have different sets of genes
    nodes_v9 = set(graph_v9.graph.nodes())
    nodes_v11 = set(graph_v11.graph.nodes())
    common_nodes = nodes_v9.intersection(nodes_v11)
    
    # Filter edges to only include those between common nodes
    filtered_edges_v9 = {(u, v) for u, v in edges_v9 if u in common_nodes and v in common_nodes}
    filtered_edges_v11 = {(u, v) for u, v in edges_v11 if u in common_nodes and v in common_nodes}
    
    num_nodes = len(common_nodes)

    # Calculate the number of possible edges (excluding self-loops)
    num_possible_edges = (num_nodes * (num_nodes - 1)) // 2

    # Calculate the number of edges in each version
    num_edges_v9 = len(filtered_edges_v9)
    num_edges_v11 = len(filtered_edges_v11)

    # Calculate new edges in v11 (edges in v11 but not in v9)
    new_edges_v11 = filtered_edges_v11 - filtered_edges_v9
    num_new_edges = len(new_edges_v11)

    # Calculate the number of zeros in v9 (possible edges - actual edges)
    num_zeros_v9 = num_possible_edges - num_edges_v9

    # Calculate the percentage of decreased zeros (new interactions that were previously zeros)
    if num_zeros_v9 == 0:
        zero_decrease_pct = 0.0
    else:
        zero_decrease_pct = (num_new_edges / num_zeros_v9) * 100

    # Log details
    log.info(f"Network type: {network_type}")
    log.info(f"Common nodes across versions: {num_nodes}")
    log.info(f"Possible edges: {num_possible_edges}")
    log.info(f"Filtered edges in v9.1: {num_edges_v9}")
    log.info(f"Filtered edges in v11.0: {num_edges_v11}")
    log.info(f"New edges in v11.0: {num_new_edges}")
    log.info(f"Zero edges in v9.1: {num_zeros_v9}")
    log.info(f"Percentage of decreased zeros: {zero_decrease_pct:.4f}%")
    
    # Check if value aligns with paper's range (for cooccurence and coexpression)
    if network_type == "cooccurence" and abs(zero_decrease_pct - 0.02) > 0.1:
        log.warning(f"Calculation differs from paper: Got {zero_decrease_pct:.4f}% for co-occurrence but paper reports ~0.02%")
    elif network_type == "coexpression" and abs(zero_decrease_pct - 2.42) > 0.3:
        log.warning(f"Calculation differs from paper: Got {zero_decrease_pct:.4f}% for co-expression but paper reports ~2.42%")
    
    # Also check other networks mentioned in paper if values are very different
    elif network_type == "experimental" and zero_decrease_pct > 3.0:
        log.warning(f"Calculation for {network_type} ({zero_decrease_pct:.4f}%) seems unusually high compared to paper ranges (0.02% to 2.42%)")
    elif network_type == "database" and zero_decrease_pct > 3.0:
        log.warning(f"Calculation for {network_type} ({zero_decrease_pct:.4f}%) seems unusually high compared to paper ranges (0.02% to 2.42%)")

    # Return all metrics as a dictionary for later use
    return {
        "network_type": network_type,
        "num_nodes": num_nodes,
        "num_possible_edges": num_possible_edges,
        "num_edges_v9": num_edges_v9,
        "num_edges_v11": num_edges_v11,
        "num_new_edges": num_new_edges,
        "num_zeros_v9": num_zeros_v9,
        "zero_decrease_pct": zero_decrease_pct,
    }


def plot_string_comparison(metrics_list):
    """
    Plot a comparison of STRING v9.1 and v11.0 networks.
    Shows edge counts and percentage of decreased zeros.

    Args:
        metrics_list: List of dictionaries containing metrics for each network type

    Returns:
        str: Path to the saved plot
    """
    # Ensure the directory exists
    os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

    # Extract data for plotting
    network_types = []
    edges_v9 = []
    edges_v11 = []
    new_edges = []
    decrease_pcts = []

    for metrics in metrics_list:
        network_types.append(metrics["network_type"])
        edges_v9.append(metrics["num_edges_v9"])
        edges_v11.append(metrics["num_edges_v11"])
        new_edges.append(metrics["num_new_edges"])
        decrease_pcts.append(metrics["zero_decrease_pct"])

    # Create figure with two subplots with more height for the top plot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [4, 2]}
    )

    # Get TorchCell color cycle
    # Our cycler is defined in torchcell.mplstyle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    # For more consistent colors, we'll use specific colors from our palette
    v9_color = "#B73C39"  # dark red (colors[1])
    v11_color = "#3978B5"  # blue (colors[2])
    threshold_color = "#000000"  # black (colors[0])
    bar_color = "#8D5694"  # purple (colors[4])

    # Bar width
    width = 0.35

    # Set positions
    x = np.arange(len(network_types))

    # Plot edge counts with custom colors
    bars1 = ax1.bar(
        x - width / 2, edges_v9, width, label="STRING v9.1 edges", color=v9_color
    )
    bars2 = ax1.bar(
        x + width / 2, edges_v11, width, label="STRING v11.0 edges", color=v11_color
    )

    # Calculate appropriate top margin for y-axis
    max_height = max(max(edges_v9), max(edges_v11))
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
    ax1.set_title("Comparison of Edge Counts: STRING v9.1 vs v11.0", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(network_types, fontsize=12, rotation=30, ha='right')
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
            fontweight='bold',
            bbox=dict(facecolor="white", alpha=0.8, pad=2, boxstyle="round,pad=0.2"),
        )

    # Set labels and titles for percentage plot
    ax2.set_ylabel("Decreased Zeros (%)", fontsize=14)
    ax2.set_xlabel("Network Type", fontsize=14)
    ax2.set_title("Percentage of Decreased Zeros from STRING v9.1 to v11.0", fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(network_types, fontsize=12, rotation=30, ha='right')
    ax2.legend(fontsize=12)

    # Add a grid
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add more space between subplots
    plt.subplots_adjust(hspace=0.3)
    
    # Ensure tight layout
    plt.tight_layout()

    # Save the figure with timestamp
    current_timestamp = timestamp()
    filename = f"string_v9.1_vs_v11.0_comparison_{current_timestamp}.png"
    filepath = os.path.join(ASSET_IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")

    log.info(f"Plot saved to {filepath}")

    return filepath


def main():
    """Main function to calculate lambda values for all network types."""
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

    log.info("Calculating zero decrease percentages...")
    for network_type in network_types:
        log.info(f"\nAnalyzing {network_type} network...")
        metrics = calculate_zero_decrease_percentage(graph, network_type)
        metrics_list.append(metrics)

        # Determine lambda value according to DANGO paper criterion
        # "Greater than 1% zeros decreased" -> lambda=0.1, otherwise lambda=1.0
        percentage = metrics["zero_decrease_pct"]
        if percentage > 1.0:
            lambda_values[f"string9_1_{network_type}"] = 0.1
        else:
            lambda_values[f"string9_1_{network_type}"] = 1.0

    # Print summary
    log.info("\n--- SUMMARY OF ZERO DECREASE PERCENTAGES ---")
    for metrics in metrics_list:
        network_type = metrics["network_type"]
        percentage = metrics["zero_decrease_pct"]
        log.info(f"{network_type}: {percentage:.4f}%")

    log.info("\n--- DETERMINED LAMBDA VALUES ---")
    for network, lambda_val in lambda_values.items():
        log.info(f"{network}: {lambda_val}")
        
    # Check for significant differences compared to paper values
    paper_values = {
        "cooccurence": 0.02,
        "coexpression": 2.42
    }
    
    differences = []
    for metrics in metrics_list:
        network_type = metrics["network_type"]
        if network_type in paper_values:
            paper_val = paper_values[network_type]
            calculated_val = metrics["zero_decrease_pct"]
            if abs(calculated_val - paper_val) > (0.1 if network_type == "cooccurence" else 0.3):
                differences.append(f"{network_type}: calculated={calculated_val:.4f}%, paper={paper_val}%")
    
    if differences:
        log.warning("\n--- DIFFERENCES FROM PAPER VALUES ---")
        for diff in differences:
            log.warning(diff)
        log.warning("These differences might be due to different STRING database versions or filtering methods")
        log.warning("The DANGO paper uses STRING v9.1 to v11.0 comparison, while we may have different node sets or network construction")

    # Generate and save the plot
    plot_path = plot_string_comparison(metrics_list)
    log.info(f"Generated plot at: {plot_path}")

    # Return the lambda values to be used in the training script
    return lambda_values


if __name__ == "__main__":
    main()
