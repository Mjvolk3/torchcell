# experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_with_interactions
# [[experiments.008-xue-ffa.scripts.additive_best_titers_per_ffa_with_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_with_interactions
# Test file: experiments/008-xue-ffa/scripts/test_additive_best_titers_per_ffa_with_interactions.py

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
from additive_free_fatty_acid_interactions import (
    load_ffa_data,
    normalize_by_reference,
    parse_genotype,
    compute_additive_interactions_with_error_propagation,
)


def find_top_titers(
    single_mutants,
    double_mutants,
    triple_mutants,
    digenic_interactions,
    trigenic_interactions,
    single_se,
    double_se,
    triple_se,
    digenic_se,
    trigenic_se,
    columns,
    abbreviations=None,
    top_n=10,
):
    """Find top genotypes by their actual fitness value.

    Returns top_n genotypes per FFA ranked by their actual fitness:
    - Singles: ranked by f_i
    - Doubles: ranked by f_ij
    - Triples: ranked by f_ijk
    """

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create reverse mapping from letter to gene name if abbreviations provided
    letter_to_gene = {}
    if abbreviations:
        letter_to_gene = {v: k for k, v in abbreviations.items()}
        if "F" not in letter_to_gene:
            letter_to_gene["F"] = "PKH1"

    top_genotypes_by_ffa = {}

    for ffa_idx, ffa in enumerate(ffa_types):
        genotype_data = []

        # Collect all genotypes with their fitness values
        # Singles
        for gene, fitness_values in single_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                display_gene = (
                    letter_to_gene.get(gene, gene) if letter_to_gene else gene
                )
                genotype_data.append(
                    {
                        "genotype": display_gene,
                        "genes": (gene,),
                        "type": "single",
                        "actual_fitness": fitness_values[ffa_idx],
                        "f_i": fitness_values[ffa_idx],
                        "f_j": np.nan,
                        "f_k": np.nan,
                        "f_ij": np.nan,
                        "f_ik": np.nan,
                        "f_jk": np.nan,
                        "f_ijk": np.nan,
                        "delta_ij": np.nan,
                        "delta_ik": np.nan,
                        "delta_jk": np.nan,
                        "sigma": np.nan,
                        "se_i": (
                            single_se.get(gene, [0] * 6)[ffa_idx] if single_se else 0
                        ),
                        "se_j": 0,
                        "se_k": 0,
                        "se_ij": 0,
                        "se_ik": 0,
                        "se_jk": 0,
                        "se_ijk": 0,
                        "se_delta_ij": 0,
                        "se_delta_ik": 0,
                        "se_delta_jk": 0,
                        "se_sigma": 0,
                    }
                )

        # Doubles
        for (gene1, gene2), fitness_values in double_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                display_gene1 = (
                    letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                )
                display_gene2 = (
                    letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2
                )

                # Get single fitnesses
                f_i = single_mutants.get(gene1, [np.nan] * 6)[ffa_idx]
                f_j = single_mutants.get(gene2, [np.nan] * 6)[ffa_idx]

                # Get interaction
                delta_ij = digenic_interactions.get((gene1, gene2), [np.nan] * 6)[ffa_idx]

                # Standard errors
                se_i = single_se.get(gene1, [0] * 6)[ffa_idx] if single_se else 0
                se_j = single_se.get(gene2, [0] * 6)[ffa_idx] if single_se else 0
                se_ij = (
                    double_se.get((gene1, gene2), [0] * 6)[ffa_idx] if double_se else 0
                )
                se_delta_ij = (
                    digenic_se.get((gene1, gene2), [0] * 6)[ffa_idx]
                    if digenic_se
                    else 0
                )

                genotype_data.append(
                    {
                        "genotype": f"{display_gene1}-{display_gene2}",
                        "genes": (gene1, gene2),
                        "type": "double",
                        "actual_fitness": fitness_values[ffa_idx],
                        "f_i": f_i,
                        "f_j": f_j,
                        "f_k": np.nan,
                        "f_ij": fitness_values[ffa_idx],
                        "f_ik": np.nan,
                        "f_jk": np.nan,
                        "f_ijk": np.nan,
                        "delta_ij": delta_ij,
                        "delta_ik": np.nan,
                        "delta_jk": np.nan,
                        "sigma": np.nan,
                        "se_i": se_i,
                        "se_j": se_j,
                        "se_k": 0,
                        "se_ij": se_ij,
                        "se_ik": 0,
                        "se_jk": 0,
                        "se_ijk": 0,
                        "se_delta_ij": se_delta_ij,
                        "se_delta_ik": 0,
                        "se_delta_jk": 0,
                        "se_sigma": 0,
                    }
                )

        # Triples
        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                display_gene1 = (
                    letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                )
                display_gene2 = (
                    letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2
                )
                display_gene3 = (
                    letter_to_gene.get(gene3, gene3) if letter_to_gene else gene3
                )

                # Get all component fitnesses
                f_i = single_mutants.get(gene1, [np.nan] * 6)[ffa_idx]
                f_j = single_mutants.get(gene2, [np.nan] * 6)[ffa_idx]
                f_k = single_mutants.get(gene3, [np.nan] * 6)[ffa_idx]

                f_ij = double_mutants.get((gene1, gene2), [np.nan] * 6)[ffa_idx]
                f_ik = double_mutants.get((gene1, gene3), [np.nan] * 6)[ffa_idx]
                f_jk = double_mutants.get((gene2, gene3), [np.nan] * 6)[ffa_idx]

                # Get interactions
                delta_ij = digenic_interactions.get((gene1, gene2), [np.nan] * 6)[ffa_idx]
                delta_ik = digenic_interactions.get((gene1, gene3), [np.nan] * 6)[ffa_idx]
                delta_jk = digenic_interactions.get((gene2, gene3), [np.nan] * 6)[ffa_idx]
                sigma = trigenic_interactions.get((gene1, gene2, gene3), [np.nan] * 6)[
                    ffa_idx
                ]

                # Standard errors
                se_i = single_se.get(gene1, [0] * 6)[ffa_idx] if single_se else 0
                se_j = single_se.get(gene2, [0] * 6)[ffa_idx] if single_se else 0
                se_k = single_se.get(gene3, [0] * 6)[ffa_idx] if single_se else 0

                se_ij = (
                    double_se.get((gene1, gene2), [0] * 6)[ffa_idx] if double_se else 0
                )
                se_ik = (
                    double_se.get((gene1, gene3), [0] * 6)[ffa_idx] if double_se else 0
                )
                se_jk = (
                    double_se.get((gene2, gene3), [0] * 6)[ffa_idx] if double_se else 0
                )

                se_ijk = (
                    triple_se.get((gene1, gene2, gene3), [0] * 6)[ffa_idx]
                    if triple_se
                    else 0
                )

                se_delta_ij = (
                    digenic_se.get((gene1, gene2), [0] * 6)[ffa_idx]
                    if digenic_se
                    else 0
                )
                se_delta_ik = (
                    digenic_se.get((gene1, gene3), [0] * 6)[ffa_idx]
                    if digenic_se
                    else 0
                )
                se_delta_jk = (
                    digenic_se.get((gene2, gene3), [0] * 6)[ffa_idx]
                    if digenic_se
                    else 0
                )

                se_sigma = (
                    trigenic_se.get((gene1, gene2, gene3), [0] * 6)[ffa_idx]
                    if trigenic_se
                    else 0
                )

                genotype_data.append(
                    {
                        "genotype": f"{display_gene1}-{display_gene2}-{display_gene3}",
                        "genes": (gene1, gene2, gene3),
                        "type": "triple",
                        "actual_fitness": fitness_values[ffa_idx],
                        "f_i": f_i,
                        "f_j": f_j,
                        "f_k": f_k,
                        "f_ij": f_ij,
                        "f_ik": f_ik,
                        "f_jk": f_jk,
                        "f_ijk": fitness_values[ffa_idx],
                        "delta_ij": delta_ij,
                        "delta_ik": delta_ik,
                        "delta_jk": delta_jk,
                        "sigma": sigma,
                        "se_i": se_i,
                        "se_j": se_j,
                        "se_k": se_k,
                        "se_ij": se_ij,
                        "se_ik": se_ik,
                        "se_jk": se_jk,
                        "se_ijk": se_ijk,
                        "se_delta_ij": se_delta_ij,
                        "se_delta_ik": se_delta_ik,
                        "se_delta_jk": se_delta_jk,
                        "se_sigma": se_sigma,
                    }
                )

        # Sort by actual fitness and take top N
        genotype_data.sort(key=lambda x: x["actual_fitness"], reverse=True)
        top_genotypes_by_ffa[ffa] = genotype_data[:top_n]

    return top_genotypes_by_ffa


def plot_top_titers(top_genotypes_by_ffa, columns):
    """Create bar plots for top titers per FFA with all fitness and interaction values."""

    # Define color scheme
    color_i = "#7191A9"  # Light blue
    color_j = "#6B8D3A"  # Green
    color_k = "#B73C39"  # Red
    color_ij = "#34699D"  # Deep blue
    color_ik = "#E6A65D"  # Orange
    color_jk = "#775A9F"  # Purple
    color_ijk = "#CC8250"  # Burnt orange

    # Interaction colors
    color_delta_ij = color_ij
    color_delta_ik = color_ik
    color_delta_jk = color_jk
    color_sigma = color_ijk

    for ffa, genotypes in top_genotypes_by_ffa.items():
        n_genotypes = len(genotypes)
        fig_width = max(20, n_genotypes * 2.5)
        fig, ax1 = plt.subplots(figsize=(fig_width, 10))

        x = np.arange(n_genotypes)
        width = 0.08
        n_bars = 11  # 7 fitness + 4 interaction bars

        # Calculate bar positions
        bar_positions = np.linspace(
            -width * (n_bars - 1) / 2, width * (n_bars - 1) / 2, n_bars
        )

        # Plot fitness values on primary y-axis with error bars (matching previous scripts style)
        # Singles
        ax1.bar(
            x + bar_positions[0],
            [g["f_i"] for g in genotypes],
            width,
            yerr=[g["se_i"] if not np.isnan(g["se_i"]) else 0 for g in genotypes],
            label=r"$f_i$ (4$\Delta$)",
            color=color_i,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[1],
            [g["f_j"] for g in genotypes],
            width,
            yerr=[g["se_j"] if not np.isnan(g["se_j"]) else 0 for g in genotypes],
            label=r"$f_j$ (4$\Delta$)",
            color=color_j,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[2],
            [g["f_k"] for g in genotypes],
            width,
            yerr=[g["se_k"] if not np.isnan(g["se_k"]) else 0 for g in genotypes],
            label=r"$f_k$ (4$\Delta$)",
            color=color_k,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        # Doubles
        ax1.bar(
            x + bar_positions[3],
            [g["f_ij"] for g in genotypes],
            width,
            yerr=[g["se_ij"] if not np.isnan(g["se_ij"]) else 0 for g in genotypes],
            label=r"$f_{ij}$ (5$\Delta$)",
            color=color_ij,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[4],
            [g["f_ik"] for g in genotypes],
            width,
            yerr=[g["se_ik"] if not np.isnan(g["se_ik"]) else 0 for g in genotypes],
            label=r"$f_{ik}$ (5$\Delta$)",
            color=color_ik,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[5],
            [g["f_jk"] for g in genotypes],
            width,
            yerr=[g["se_jk"] if not np.isnan(g["se_jk"]) else 0 for g in genotypes],
            label=r"$f_{jk}$ (5$\Delta$)",
            color=color_jk,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        # Triple
        ax1.bar(
            x + bar_positions[6],
            [g["f_ijk"] for g in genotypes],
            width,
            yerr=[g["se_ijk"] if not np.isnan(g["se_ijk"]) else 0 for g in genotypes],
            label=r"$f_{ijk}$ (6$\Delta$)",
            color=color_ijk,
            alpha=0.8,
            capsize=3,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        # Plot interaction values on same axis with hatching patterns (matching previous scripts style)
        ax1.bar(
            x + bar_positions[7],
            [g["delta_ij"] for g in genotypes],
            width,
            yerr=[
                g["se_delta_ij"] if not np.isnan(g["se_delta_ij"]) else 0 for g in genotypes
            ],
            label=r"$\delta_{ij}$",
            color=color_delta_ij,
            alpha=0.7,
            capsize=3,
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[8],
            [g["delta_ik"] for g in genotypes],
            width,
            yerr=[
                g["se_delta_ik"] if not np.isnan(g["se_delta_ik"]) else 0 for g in genotypes
            ],
            label=r"$\delta_{ik}$",
            color=color_delta_ik,
            alpha=0.7,
            capsize=3,
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[9],
            [g["delta_jk"] for g in genotypes],
            width,
            yerr=[
                g["se_delta_jk"] if not np.isnan(g["se_delta_jk"]) else 0 for g in genotypes
            ],
            label=r"$\delta_{jk}$",
            color=color_delta_jk,
            alpha=0.7,
            capsize=3,
            hatch="///",
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        ax1.bar(
            x + bar_positions[10],
            [g["sigma"] for g in genotypes],
            width,
            yerr=[g["se_sigma"] if not np.isnan(g["se_sigma"]) else 0 for g in genotypes],
            label=r"$\sigma_{ijk}$",
            color=color_sigma,
            alpha=0.9,
            capsize=3,
            hatch="xxx",
            edgecolor="black",
            linewidth=0.5,
            error_kw={"linewidth": 1, "alpha": 0.7},
        )

        # Reference lines
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)

        # Labels and title
        ax1.set_xlabel("Genotypes (sorted by actual fitness)", fontsize=14)
        ax1.set_ylabel("Fitness / Interaction Score", fontsize=14)
        ax1.set_title(
            f"Top 10 Titers for {ffa} Across All Strains - ADDITIVE MODEL",
            fontsize=16,
            fontweight="bold",
        )

        # X-axis labels with genotype names and types
        labels = [f"{g['genotype']}\n({g['type']})" for g in genotypes]
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        # Legend
        ax1.legend(loc="upper left", fontsize=9, ncol=2)

        # Grid
        ax1.grid(True, alpha=0.3, axis="y")

        # Add integer-spaced ticks
        y_min, y_max = ax1.get_ylim()
        y_min_int = int(np.floor(y_min))
        y_max_int = int(np.ceil(y_max))

        # Choose appropriate step size
        y_range = y_max_int - y_min_int
        if y_range > 20:
            step = 5
        elif y_range > 10:
            step = 2
        else:
            step = 1

        # Create ticks
        if y_min_int < 0:
            y_ticks = np.arange(y_min_int - y_min_int % step, y_max_int + step, step)
        else:
            y_ticks = np.arange(0, y_max_int + step, step)

        # Ensure 0 and 1 are included
        if 0 not in y_ticks and y_min <= 0 <= y_max:
            y_ticks = np.sort(np.append(y_ticks, 0))
        if 1 not in y_ticks and y_min <= 1 <= y_max:
            y_ticks = np.sort(np.append(y_ticks, 1))

        # Filter to only show ticks within the actual limits
        y_ticks = y_ticks[(y_ticks >= y_min) & (y_ticks <= y_max)]

        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f"{int(t)}" for t in y_ticks])

        # Adjust layout
        plt.tight_layout()

        # Save figure
        filename = f"additive_top_titers_{ffa.replace(':', '')}_{timestamp()}.png"
        filepath = osp.join(ASSET_IMAGES_DIR, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Top titers plot for {ffa} saved to:")
        print(f"  {filepath}")

        # Print summary
        print(f"\n{ffa} Top 10 Titers Summary:")
        print(f"  Max fitness: {genotypes[0]['actual_fitness']:.3f}")
        print(f"  Min fitness in top 10: {genotypes[-1]['actual_fitness']:.3f}")
        print(f"  Genotype types: {', '.join(set(g['type'] for g in genotypes))}")
        print(
            f"  Top genotype: {genotypes[0]['genotype']} ({genotypes[0]['type']}) - fitness: {genotypes[0]['actual_fitness']:.3f}"
        )


def main():
    """Main function to find and plot top titers per FFA."""

    print("Loading FFA data...")
    file_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"
    raw_df, abbreviations, replicate_dict = load_ffa_data(file_path)

    print("Normalizing by positive control (+ve Ctrl)...")
    normalized_df, normalized_replicates = normalize_by_reference(
        raw_df, replicate_dict
    )

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
    ) = compute_additive_interactions_with_error_propagation(
        normalized_df, normalized_replicates, abbreviations
    )

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    # Find top titers
    print("\nFinding top 10 titers per FFA...")
    top_genotypes_by_ffa = find_top_titers(
        single_mutants,
        double_mutants,
        triple_mutants,
        digenic_interactions,
        trigenic_interactions,
        single_se,
        double_se,
        triple_se,
        digenic_se,
        trigenic_se,
        columns,
        abbreviations,
        top_n=10,
    )

    # Create plots
    print("\nCreating plots for top titers...")
    plot_top_titers(top_genotypes_by_ffa, columns)

    print(
        f"\nAll best titers plots complete! Check the file paths above to view the plots."
    )


if __name__ == "__main__":
    main()
