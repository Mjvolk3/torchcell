import matplotlib.pyplot as plt
import numpy as np
from torchcell.datamodules.cell import DatasetIndexSplit
from collections import defaultdict


def plot_dataset_name_index_split(
    split_index: DatasetIndexSplit, title: str, save_path: str, threshold: float = 0.02
):
    # Reverse the order: train on top, val in the middle, and test at the bottom
    splits = ["train", "val", "test"][::-1]

    fig, ax = plt.subplots(figsize=(16, 6))  # Increased figure width for a wider plot

    # Define your color list
    color_list = [
        "#53777A",
        "#8F918B",
        "#D1A0A2",
        "#A8BDB5",
        "#B8AD9E",
        "#7B9EAE",
        "#F75C4C",
        "#82B3AE",
        "#FFD3B6",
        "#746D75",
        "#FF8C94",
        "#5E8C61",
        "#B565A7",
        "#955251",
        "#009B77",
        "#DD4124",
        "#D65076",
        "#D0838E",
        "#FFA257",
        "#ECD078",
    ]

    # Initialize a color map
    color_map = defaultdict(lambda: "#000000")

    # Plotting the stacked bars
    bar_positions = np.arange(len(splits))  # y positions for the bars
    bar_height = 0.6  # Height of each bar
    final_labels_all_splits = set()  # Set to store all final labels after reduction
    split_labels = {}  # To keep track of the final labels for each split

    # Reduce labels by thresholding and combining small labels into "Other"
    for split in splits:
        split_data = getattr(split_index, split)
        final_labels, final_sizes, final_percentages = [], [], []
        if split_data:
            labels = list(split_data.keys())
            sizes = [len(indices) for indices in split_data.values()]
            total = sum(sizes)
            percentages = [size / total * 100 for size in sizes]

            other_size = 0
            for label, size, percent in zip(labels, sizes, percentages):
                if percent / 100 < threshold:
                    other_size += size
                else:
                    final_labels.append(label)
                    final_sizes.append(size)
                    final_percentages.append(percent)

            if other_size > 0:
                final_labels.append("Other")
                final_sizes.append(other_size)
                final_percentages.append(other_size / total * 100)

            split_labels[split] = (final_labels, final_sizes, final_percentages)
            final_labels_all_splits.update(final_labels)

    # Now that we have reduced labels, assign colors to them
    unique_labels = sorted(final_labels_all_splits)
    if len(unique_labels) > len(color_list):
        raise ValueError(
            f"Not enough colors for all labels. {len(unique_labels)} labels but only {len(color_list)} colors provided."
        )

    for idx, label in enumerate(unique_labels):
        color_map[label] = color_list[idx]

    # Plot the bars
    legend_labels = set()  # Keep track of which labels are plotted in the legend
    for i, split in enumerate(splits):
        final_labels, final_sizes, final_percentages = split_labels.get(
            split, ([], [], [])
        )
        if final_labels:
            total = sum(final_sizes)
            cumulative = 0

            # Stacked bars
            for label, percent in zip(final_labels, final_percentages):
                ax.barh(
                    split,  # The current split position (train, val, test)
                    percent,  # Width of the bar (percentage)
                    left=cumulative,  # Start position for the current stacked bar
                    height=bar_height,  # Bar height
                    color=color_map[label],  # Assign consistent color after reduction
                    edgecolor="white",  # White border for clarity
                    label=(
                        label if label not in legend_labels else ""
                    ),  # Avoid duplicate labels in legend
                )

                # Add smaller percentage text inside the bar
                ax.text(
                    cumulative + percent / 2,  # Center of the current bar
                    i,  # y-position (split index)
                    f"{percent:.1f}%",  # Percentage text
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=7,  # Decreased font size for percentages
                    weight="bold",
                )

                cumulative += percent  # Update the cumulative width
                legend_labels.add(label)  # Track labels used in legend

    # Set y-ticks and labels for clarity
    ax.set_yticks(bar_positions)
    ax.set_yticklabels([split.capitalize() for split in splits], fontsize=12)

    # Add gridlines for easier reading
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)

    # Set the x-axis limit to exactly 100%
    ax.set_xlim(0, 100)

    # Add legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Dataset Labels",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
    )

    # Set labels and titles
    ax.set_xlabel("Percentage", fontsize=12)
    ax.set_title(f"{title}", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved static image to {save_path}")
    plt.close(fig)
