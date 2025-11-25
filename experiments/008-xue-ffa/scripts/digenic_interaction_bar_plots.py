# experiments/008-xue-ffa/scripts/digenic_interaction_bar_plots
# [[experiments.008-xue-ffa.scripts.digenic_interaction_bar_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/digenic_interaction_bar_plots
# Test file: experiments/008-xue-ffa/scripts/test_digenic_interaction_bar_plots.py


import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = (
    "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
)
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)

# Import functions from the main script
import sys

sys.path.append(
    "/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/scripts"
)
from free_fatty_acid_interactions import (
    load_ffa_data,
    normalize_by_reference,
    parse_genotype,
    compute_interactions_with_error_propagation,
)


def create_digenic_bar_plots(
    digenic_interactions, single_mutants, double_mutants, columns,
    single_se=None, double_se=None, digenic_se=None, abbreviations=None
):
    """Create bar plots showing f_i, f_j, f_ij and epsilon for all digenic interactions.

    Now includes error bars from standard errors for all measurements and
    propagated errors for interactions.
    """

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create reverse mapping from letter to gene name if abbreviations provided
    letter_to_gene = {}
    if abbreviations:
        letter_to_gene = {v: k for k, v in abbreviations.items()}
        # Add F = PKH1 if missing
        if 'F' not in letter_to_gene:
            letter_to_gene['F'] = 'PKH1'

    # Create a figure for each FFA type
    for ffa_idx, ffa in enumerate(ffa_types):
        # Collect data for this FFA
        plot_data = []

        for (gene1, gene2), epsilon_values in digenic_interactions.items():
            if ffa_idx < len(epsilon_values):
                # Get single mutant fitnesses
                f_i = (
                    single_mutants[gene1][ffa_idx]
                    if gene1 in single_mutants
                    else np.nan
                )
                f_j = (
                    single_mutants[gene2][ffa_idx]
                    if gene2 in single_mutants
                    else np.nan
                )

                # Get double mutant fitness
                f_ij = (
                    double_mutants[(gene1, gene2)][ffa_idx]
                    if (gene1, gene2) in double_mutants
                    else np.nan
                )

                # Get interaction
                epsilon = epsilon_values[ffa_idx]

                # Get standard errors if provided
                se_i = single_se[gene1][ffa_idx] if single_se and gene1 in single_se else 0
                se_j = single_se[gene2][ffa_idx] if single_se and gene2 in single_se else 0
                se_ij = double_se[(gene1, gene2)][ffa_idx] if double_se and (gene1, gene2) in double_se else 0
                se_epsilon = digenic_se[(gene1, gene2)][ffa_idx] if digenic_se and (gene1, gene2) in digenic_se else 0

                if not any(np.isnan([f_i, f_j, f_ij, epsilon])):
                    # Convert letters to gene names if mapping available
                    display_gene1 = letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                    display_gene2 = letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2

                    plot_data.append(
                        {
                            "pair": f"{display_gene1}-{display_gene2}",
                            "f_i": f_i,
                            "f_j": f_j,
                            "f_ij": f_ij,
                            "epsilon": epsilon,
                            "se_i": se_i,
                            "se_j": se_j,
                            "se_ij": se_ij,
                            "se_epsilon": se_epsilon,
                            "f_i_name": display_gene1,
                            "f_j_name": display_gene2,
                        }
                    )

        if not plot_data:
            continue

        # Sort by interaction strength for better visualization
        plot_data = sorted(plot_data, key=lambda x: x["epsilon"])

        # Create figure with appropriate size
        n_pairs = len(plot_data)
        fig_width = max(12, n_pairs * 0.8)  # Adjust width based on number of pairs
        fig, ax1 = plt.subplots(figsize=(fig_width, 12))  # Increased height from 8 to 12

        # Set up x positions for bars
        x = np.arange(n_pairs)
        width = 0.2  # Width of each bar

        # Colors from torchcell palette
        color_f_i = "#7191A9"  # Light blue
        color_f_j = "#6B8D3A"  # Green
        color_f_ij = "#B73C39"  # Red
        color_epsilon = "#E6A65D"  # Orange

        # Plot fitness values on primary y-axis with error bars
        bars1 = ax1.bar(
            x - 1.5 * width,
            [d["f_i"] for d in plot_data],
            width,
            yerr=[d["se_i"] if not np.isnan(d["se_i"]) else 0 for d in plot_data],
            label=r"$f_i$ (4Δ)",
            color=color_f_i,
            alpha=0.8,
            capsize=3,
            error_kw={'linewidth': 1, 'alpha': 0.7}
        )
        bars2 = ax1.bar(
            x - 0.5 * width,
            [d["f_j"] for d in plot_data],
            width,
            yerr=[d["se_j"] if not np.isnan(d["se_j"]) else 0 for d in plot_data],
            label=r"$f_j$ (4Δ)",
            color=color_f_j,
            alpha=0.8,
            capsize=3,
            error_kw={'linewidth': 1, 'alpha': 0.7}
        )
        bars3 = ax1.bar(
            x + 0.5 * width,
            [d["f_ij"] for d in plot_data],
            width,
            yerr=[d["se_ij"] if not np.isnan(d["se_ij"]) else 0 for d in plot_data],
            label=r"$f_{ij}$ (5Δ)",
            color=color_f_ij,
            alpha=0.8,
            capsize=3,
            error_kw={'linewidth': 1, 'alpha': 0.7}
        )

        # Plot interaction values on same axis with propagated error bars
        # No secondary axis - use the same scale
        bars4 = ax1.bar(
            x + 1.5 * width,
            [d["epsilon"] for d in plot_data],
            width,
            yerr=[d["se_epsilon"] if not np.isnan(d["se_epsilon"]) else 0 for d in plot_data],
            label=r"$\varepsilon$ (interaction)",
            color=color_epsilon,
            alpha=0.8,
            capsize=3,
            error_kw={'linewidth': 1, 'alpha': 0.7}
        )

        # Add horizontal line at y=0
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Add horizontal line at y=1 for fitness axis (reference strain level)
        ax1.axhline(
            y=1, color="black", linestyle="--", alpha=0.5, linewidth=1.5, label="Reference (3Δ)"
        )

        # Labels and title
        ax1.set_xlabel("Gene Pairs", fontsize=14)
        ax1.set_ylabel("Fitness / Interaction Score", fontsize=14, color="black")
        ax1.set_title(f"Digenic Interactions for {ffa}", fontsize=16, fontweight="bold")

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels([d["pair"] for d in plot_data], rotation=45, ha="right")

        # Legend
        ax1.legend(loc="upper left", fontsize=11)

        # Grid
        ax1.grid(True, alpha=0.3, axis="y")

        # Align zeros of both y-axes
        # Get current axis limits
        y1_min, y1_max = ax1.get_ylim()
        # Get limits from single axis
        y2_min, y2_max = ax1.get_ylim()

        # Calculate where zero sits in the interaction range
        if y2_min < 0 and y2_max > 0:
            # Both positive and negative values
            # Calculate the proportion of the range that is below zero
            interaction_zero_position = abs(y2_min) / (y2_max - y2_min)

            # Extend the fitness axis downward by the same proportion
            # This aligns the zeros
            fitness_range = y1_max - 0  # Current fitness range (0 to max)
            new_y1_min = -fitness_range * interaction_zero_position / (1 - interaction_zero_position)
            ax1.set_ylim(new_y1_min, y1_max)

            # Keep interaction axis as is
            # Using single axis
        else:
            # No alignment needed if interactions don't cross zero
            ax1.set_ylim(0, y1_max)
            # Using single axis

        # Add integer-spaced ticks while maintaining zero alignment
        y1_min_curr, y1_max_curr = ax1.get_ylim()
        y2_min_curr, y2_max_curr = ax1.get_ylim()

        # The zeros are already aligned from the previous code
        # Now we just need to add integer ticks while preserving the alignment

        # For fitness axis - use integer ticks
        y1_min_int = int(np.floor(y1_min_curr))
        y1_max_int = int(np.ceil(y1_max_curr))

        # Choose appropriate step size
        y1_range = y1_max_int - y1_min_int
        if y1_range > 20:
            step1 = 5
        elif y1_range > 10:
            step1 = 2
        else:
            step1 = 1

        # Create ticks starting from 0 (or nearest multiple if using larger steps)
        if y1_min_int < 0:
            y1_ticks = np.arange(y1_min_int - y1_min_int % step1, y1_max_int + step1, step1)
        else:
            y1_ticks = np.arange(0, y1_max_int + step1, step1)

        # Ensure 0 and 1 are included
        if 0 not in y1_ticks and y1_min_curr <= 0 <= y1_max_curr:
            y1_ticks = np.sort(np.append(y1_ticks, 0))
        if 1 not in y1_ticks and y1_min_curr <= 1 <= y1_max_curr:
            y1_ticks = np.sort(np.append(y1_ticks, 1))

        # Filter to only show ticks within the actual limits
        y1_ticks = y1_ticks[(y1_ticks >= y1_min_curr) & (y1_ticks <= y1_max_curr)]

        ax1.set_yticks(y1_ticks)
        ax1.set_yticklabels([f'{int(t)}' for t in y1_ticks])

        # For interaction axis - use integer ticks
        y2_min_int = int(np.floor(y2_min_curr))
        y2_max_int = int(np.ceil(y2_max_curr))

        # Choose appropriate step size
        y2_range = y2_max_int - y2_min_int
        if y2_range > 20:
            step2 = 5
        elif y2_range > 10:
            step2 = 2
        else:
            step2 = 1

        # Create ticks, ensuring 0 is included if in range
        if y2_min_int < 0 and y2_max_int > 0:
            # Range crosses zero - build from zero outward
            y2_ticks = np.concatenate([
                np.arange(0, y2_min_int - step2, -step2)[::-1],
                np.arange(0, y2_max_int + step2, step2)
            ])
            y2_ticks = np.unique(y2_ticks)  # Remove duplicate 0
        else:
            y2_ticks = np.arange(y2_min_int - y2_min_int % step2, y2_max_int + step2, step2)

        # Filter to only show ticks within the actual limits
        y2_ticks = y2_ticks[(y2_ticks >= y2_min_curr) & (y2_ticks <= y2_max_curr)]

        # Removed ax2 tick setting - using single axis

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"digenic_bars_3_delta_normalized_{ffa.replace(':', '')}_{timestamp()}.png"
        filepath = osp.join(ASSET_IMAGES_DIR, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Digenic interaction bar plot for {ffa} saved to:")
        print(f"  {filepath}")

        # Print summary statistics for this FFA
        epsilon_values = [d["epsilon"] for d in plot_data]
        print(f"\n{ffa} Summary (TF interactions):")
        print(f"  Number of pairs: {n_pairs}")
        print(f"  Mean ε: {np.mean(epsilon_values):.3f}")
        print(f"  Min ε: {np.min(epsilon_values):.3f}")
        print(f"  Max ε: {np.max(epsilon_values):.3f}")
        print(f"  Negative interactions (synergistic): {sum(e < 0 for e in epsilon_values)}/{n_pairs}")
        print(f"  Positive interactions (buffering): {sum(e > 0 for e in epsilon_values)}/{n_pairs}")


def create_summary_plot(digenic_interactions, single_mutants, double_mutants, columns, abbreviations=None):
    """Create a summary plot showing all interactions across all FFAs.

    All values are normalized to P-S-Y 3Δ reference strain.
    """

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create reverse mapping from letter to gene name if abbreviations provided
    letter_to_gene = {}
    if abbreviations:
        letter_to_gene = {v: k for k, v in abbreviations.items()}
        # Add F = PKH1 if missing
        if 'F' not in letter_to_gene:
            letter_to_gene['F'] = 'PKH1'

    # Prepare data for summary
    summary_data = []

    for ffa_idx, ffa in enumerate(ffa_types):
        for (gene1, gene2), epsilon_values in digenic_interactions.items():
            if ffa_idx < len(epsilon_values):
                f_i = (
                    single_mutants[gene1][ffa_idx]
                    if gene1 in single_mutants
                    else np.nan
                )
                f_j = (
                    single_mutants[gene2][ffa_idx]
                    if gene2 in single_mutants
                    else np.nan
                )
                f_ij = (
                    double_mutants[(gene1, gene2)][ffa_idx]
                    if (gene1, gene2) in double_mutants
                    else np.nan
                )
                epsilon = epsilon_values[ffa_idx]

                if not any(np.isnan([f_i, f_j, f_ij, epsilon])):
                    # Convert letters to gene names if mapping available
                    display_gene1 = letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                    display_gene2 = letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2

                    summary_data.append(
                        {
                            "ffa": ffa,
                            "pair": f"{display_gene1}-{display_gene2}",
                            "f_i": f_i,
                            "f_j": f_j,
                            "f_ij": f_ij,
                            "expected": f_i * f_j,
                            "epsilon": epsilon,
                        }
                    )

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(summary_data)

    # Create scatter plot of observed vs expected
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Observed vs Expected
    ax = axes[0]
    for ffa in ffa_types:
        ffa_data = df[df["ffa"] == ffa]
        ax.scatter(ffa_data["expected"], ffa_data["f_ij"], alpha=0.6, s=30, label=ffa)

    # Add diagonal line (perfect multiplicative model)
    max_val = max(df["expected"].max(), df["f_ij"].max())
    ax.plot(
        [0, max_val], [0, max_val], "k--", alpha=0.5, label="Expected (multiplicative)"
    )

    ax.set_xlabel(r"Expected FFA Production ($f_i \times f_j$)", fontsize=14)
    ax.set_ylabel(r"Observed FFA Production ($f_{ij}$)", fontsize=14)
    ax.set_title(
        "Observed vs Expected Double TF Mutant Effects", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of epsilon values
    ax = axes[1]

    # Determine common bin edges for all FFAs
    all_epsilon = df["epsilon"].values
    bins = np.linspace(all_epsilon.min(), all_epsilon.max(), 50)

    for ffa in ffa_types:
        ffa_data = df[df["ffa"] == ffa]
        ax.hist(ffa_data["epsilon"], alpha=0.6, bins=bins, label=ffa, edgecolor="black", linewidth=0.5)

    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xlabel(r"Interaction Score ($\varepsilon$)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Distribution of Digenic TF Interactions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Summary of All Digenic TF Interactions", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save figure
    filename = f"digenic_summary_3_delta_normalized_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDigenic summary plot saved to:")
    print(f"  {filepath}")

    # Print overall statistics
    print("\nOverall Summary Statistics (TF interactions):")
    print(f"Total interactions analyzed: {len(df)}")
    print(f"Mean ε across all FFAs: {df['epsilon'].mean():.3f}")
    print(f"Percentage negative (synergistic): {100 * (df['epsilon'] < 0).sum() / len(df):.1f}%")
    print(f"Percentage positive (buffering): {100 * (df['epsilon'] > 0).sum() / len(df):.1f}%")
    print(f"Mean observed/expected ratio: {(df['f_ij'] / df['expected']).mean():.3f}")


def main():
    """Main function to create digenic interaction bar plots."""

    print("Loading FFA data...")
    file_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"
    raw_df, abbreviations, replicate_dict = load_ffa_data(file_path)

    print("Normalizing by positive control (POX1-FAA1-FAA4)...")
    normalized_df, normalized_replicates = normalize_by_reference(raw_df, replicate_dict)

    print("Computing interactions...")
    (
        digenic_interactions,
        digenic_sd,
        digenic_se,
        digenic_pvalues,
        trigenic_interactions,
        trigenic_sd,
        trigenic_se,
        trigenic_pvalues,
        single_mutants,
        double_mutants,
        triple_mutants,
        single_sd,
        single_se,
        double_sd,
        double_se,
        triple_sd,
        triple_se,
    ) = compute_interactions_with_error_propagation(
        normalized_df, normalized_replicates, abbreviations
    )

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    print("\nCreating bar plots for each FFA type...")
    create_digenic_bar_plots(
        digenic_interactions, single_mutants, double_mutants, columns,
        single_se=single_se, double_se=double_se, digenic_se=digenic_se,
        abbreviations=abbreviations
    )

    print("\nCreating summary plots...")
    create_summary_plot(digenic_interactions, single_mutants, double_mutants, columns, abbreviations)

    print("\nAll digenic interaction plots complete! Check the file paths above to view the plots.")


if __name__ == "__main__":
    main()
