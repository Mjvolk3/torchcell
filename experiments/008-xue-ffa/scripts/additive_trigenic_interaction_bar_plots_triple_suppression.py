# experiments/008-xue-ffa/scripts/additive_trigenic_interaction_bar_plots_triple_suppression
# [[experiments.008-xue-ffa.scripts.additive_trigenic_interaction_bar_plots_triple_suppression]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/additive_trigenic_interaction_bar_plots_triple_suppression
# Test file: experiments/008-xue-ffa/scripts/test_additive_trigenic_interaction_bar_plots_triple_suppression.py


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


def find_recovery_patterns(
    trigenic_interactions, single_mutants, double_mutants, triple_mutants,
    columns, abbreviations=None
):
    """Find trigenic interactions with the recovery pattern:
    - All singles (f_i, f_j, f_k) > all doubles (f_ij, f_ik, f_jk)
    - Triple (f_ijk) > all doubles (f_ij, f_ik, f_jk)

    This represents cases where single mutations and the triple mutation
    show recovery compared to the double mutant combinations.
    """

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create reverse mapping from letter to gene name if abbreviations provided
    letter_to_gene = {}
    if abbreviations:
        letter_to_gene = {v: k for k, v in abbreviations.items()}
        if 'F' not in letter_to_gene:
            letter_to_gene['F'] = 'PKH1'

    recovery_patterns = []

    for ffa_idx, ffa in enumerate(ffa_types):
        for (gene1, gene2, gene3), tau_values in trigenic_interactions.items():
            if ffa_idx < len(tau_values):
                # Get all fitness values
                f_i = single_mutants.get(gene1, [np.nan]*6)[ffa_idx]
                f_j = single_mutants.get(gene2, [np.nan]*6)[ffa_idx]
                f_k = single_mutants.get(gene3, [np.nan]*6)[ffa_idx]

                f_ij = double_mutants.get((gene1, gene2), [np.nan]*6)[ffa_idx]
                f_ik = double_mutants.get((gene1, gene3), [np.nan]*6)[ffa_idx]
                f_jk = double_mutants.get((gene2, gene3), [np.nan]*6)[ffa_idx]

                f_ijk = triple_mutants.get((gene1, gene2, gene3), [np.nan]*6)[ffa_idx]

                # Check if all values are valid
                if not any(np.isnan([f_i, f_j, f_k, f_ij, f_ik, f_jk, f_ijk])):
                    min_single = min(f_i, f_j, f_k)
                    max_double = max(f_ij, f_ik, f_jk)

                    # Check the recovery pattern
                    if min_single > max_double and f_ijk > max_double:
                        # Convert letters to gene names
                        display_gene1 = letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                        display_gene2 = letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2
                        display_gene3 = letter_to_gene.get(gene3, gene3) if letter_to_gene else gene3

                        recovery_patterns.append({
                            'ffa': ffa,
                            'ffa_idx': ffa_idx,
                            'genes': (gene1, gene2, gene3),
                            'triple': f"{display_gene1}-{display_gene2}-{display_gene3}",
                            'f_i': f_i, 'f_j': f_j, 'f_k': f_k,
                            'f_ij': f_ij, 'f_ik': f_ik, 'f_jk': f_jk,
                            'f_ijk': f_ijk,
                            'min_single': min_single,
                            'max_double': max_double,
                            'single_recovery': min_single - max_double,
                            'triple_recovery': f_ijk - max_double,
                            'sigma': tau_values[ffa_idx]
                        })

    return recovery_patterns


def create_trigenic_bar_plots(
    trigenic_interactions, digenic_interactions,
    single_mutants, double_mutants, triple_mutants, columns,
    single_se=None, double_se=None, triple_se=None,
    digenic_se=None, trigenic_se=None,
    abbreviations=None, top_n=30, recovery_only=False
):
    """Create bar plots showing all fitness and interaction values for trigenic interactions.

    Shows f_i, f_j, f_k, f_ij, f_ik, f_jk, f_ijk on primary axis
    Shows �_ij, �_ik, �_jk, �_ijk on same axis (single y-axis)
    Includes error propagation for all values.

    Args:
        top_n: Show only top N interactions by absolute tau value (default 30)
        recovery_only: If True, show only recovery pattern interactions
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

    # Define color scheme with matching indices
    # Using distinct colors from torchcell.mplstyle palette
    # Base colors for singles
    color_i = "#7191A9"  # Light blue (from palette)
    color_j = "#6B8D3A"  # Green (from palette)
    color_k = "#B73C39"  # Red (from palette)

    # Mixed colors for doubles (more distinct colors from palette)
    color_ij = "#34699D"  # Deep blue (from palette)
    color_ik = "#E6A65D"  # Orange (from palette)
    color_jk = "#775A9F"  # Purple (from palette)

    # Triple color - matching with tau
    color_ijk = "#CC8250"  # Burnt orange (from palette)

    # Interaction colors - SAME as fitness colors but with patterns
    color_delta_ij = color_ij  # Same as f_ij
    color_delta_ik = color_ik  # Same as f_ik
    color_delta_jk = color_jk  # Same as f_jk
    color_sigma = color_ijk    # Same as f_ijk

    # Create a figure for each FFA type
    for ffa_idx, ffa in enumerate(ffa_types):
        # Collect data for this FFA
        plot_data = []

        for (gene1, gene2, gene3), tau_values in trigenic_interactions.items():
            if ffa_idx < len(tau_values):
                # Get single mutant fitnesses
                f_i = single_mutants.get(gene1, [np.nan]*6)[ffa_idx]
                f_j = single_mutants.get(gene2, [np.nan]*6)[ffa_idx]
                f_k = single_mutants.get(gene3, [np.nan]*6)[ffa_idx]

                # Get double mutant fitnesses
                f_ij = double_mutants.get((gene1, gene2), [np.nan]*6)[ffa_idx]
                f_ik = double_mutants.get((gene1, gene3), [np.nan]*6)[ffa_idx]
                f_jk = double_mutants.get((gene2, gene3), [np.nan]*6)[ffa_idx]

                # Get triple mutant fitness
                f_ijk = triple_mutants.get((gene1, gene2, gene3), [np.nan]*6)[ffa_idx]

                # Get digenic interactions
                delta_ij = digenic_interactions.get((gene1, gene2), [np.nan]*6)[ffa_idx]
                delta_ik = digenic_interactions.get((gene1, gene3), [np.nan]*6)[ffa_idx]
                delta_jk = digenic_interactions.get((gene2, gene3), [np.nan]*6)[ffa_idx]

                # Get trigenic interaction
                sigma = tau_values[ffa_idx]

                # Get standard errors if provided
                se_i = single_se.get(gene1, [0]*6)[ffa_idx] if single_se else 0
                se_j = single_se.get(gene2, [0]*6)[ffa_idx] if single_se else 0
                se_k = single_se.get(gene3, [0]*6)[ffa_idx] if single_se else 0

                se_ij = double_se.get((gene1, gene2), [0]*6)[ffa_idx] if double_se else 0
                se_ik = double_se.get((gene1, gene3), [0]*6)[ffa_idx] if double_se else 0
                se_jk = double_se.get((gene2, gene3), [0]*6)[ffa_idx] if double_se else 0

                se_ijk = triple_se.get((gene1, gene2, gene3), [0]*6)[ffa_idx] if triple_se else 0

                se_delta_ij = digenic_se.get((gene1, gene2), [0]*6)[ffa_idx] if digenic_se else 0
                se_delta_ik = digenic_se.get((gene1, gene3), [0]*6)[ffa_idx] if digenic_se else 0
                se_delta_jk = digenic_se.get((gene2, gene3), [0]*6)[ffa_idx] if digenic_se else 0

                se_sigma = trigenic_se.get((gene1, gene2, gene3), [0]*6)[ffa_idx] if trigenic_se else 0

                # Check if we have valid data
                if not np.isnan(sigma) and not np.isnan(f_ijk):
                    # Convert letters to gene names if mapping available
                    display_gene1 = letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                    display_gene2 = letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2
                    display_gene3 = letter_to_gene.get(gene3, gene3) if letter_to_gene else gene3

                    plot_data.append({
                        "triple": f"{display_gene1}-{display_gene2}-{display_gene3}",
                        # Fitness values
                        "f_i": f_i, "f_j": f_j, "f_k": f_k,
                        "f_ij": f_ij, "f_ik": f_ik, "f_jk": f_jk,
                        "f_ijk": f_ijk,
                        # Interaction values
                        "delta_ij": delta_ij, "delta_ik": delta_ik, "delta_jk": delta_jk,
                        "sigma": sigma,
                        # Standard errors
                        "se_i": se_i, "se_j": se_j, "se_k": se_k,
                        "se_ij": se_ij, "se_ik": se_ik, "se_jk": se_jk,
                        "se_ijk": se_ijk,
                        "se_delta_ij": se_delta_ij, "se_delta_ik": se_delta_ik, "se_delta_jk": se_delta_jk,
                        "se_sigma": se_sigma,
                    })

        if not plot_data:
            continue

        # Sort by absolute sigma value and take top N
        plot_data = sorted(plot_data, key=lambda x: abs(x["sigma"]), reverse=True)[:top_n]
        # Re-sort by sigma value for visualization (high to low, left to right)
        plot_data = sorted(plot_data, key=lambda x: x["sigma"], reverse=True)

        # Create figure with appropriate size
        n_triples = len(plot_data)
        fig_width = max(20, n_triples * 1.5)  # Wide figure for many bars
        fig, ax1 = plt.subplots(figsize=(fig_width, 10))

        # Set up x positions for bars (11 bars per triple)
        x = np.arange(n_triples)
        n_bars = 11
        width = 0.08  # Very thin bars

        # Calculate bar positions
        bar_positions = np.linspace(-width * (n_bars-1)/2, width * (n_bars-1)/2, n_bars)

        # Plot fitness values on primary y-axis with error bars (matching digenic style)
        # Singles
        ax1.bar(x + bar_positions[0], [d["f_i"] for d in plot_data], width,
                yerr=[d["se_i"] if not np.isnan(d["se_i"]) else 0 for d in plot_data],
                label=r"$f_i$ (4$\Delta$)",
                color=color_i, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[1], [d["f_j"] for d in plot_data], width,
                yerr=[d["se_j"] if not np.isnan(d["se_j"]) else 0 for d in plot_data],
                label=r"$f_j$ (4$\Delta$)",
                color=color_j, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[2], [d["f_k"] for d in plot_data], width,
                yerr=[d["se_k"] if not np.isnan(d["se_k"]) else 0 for d in plot_data],
                label=r"$f_k$ (4$\Delta$)",
                color=color_k, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Doubles
        ax1.bar(x + bar_positions[3], [d["f_ij"] for d in plot_data], width,
                yerr=[d["se_ij"] if not np.isnan(d["se_ij"]) else 0 for d in plot_data],
                label=r"$f_{ij}$ (5$\Delta$)",
                color=color_ij, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[4], [d["f_ik"] for d in plot_data], width,
                yerr=[d["se_ik"] if not np.isnan(d["se_ik"]) else 0 for d in plot_data],
                label=r"$f_{ik}$ (5$\Delta$)",
                color=color_ik, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[5], [d["f_jk"] for d in plot_data], width,
                yerr=[d["se_jk"] if not np.isnan(d["se_jk"]) else 0 for d in plot_data],
                label=r"$f_{jk}$ (5$\Delta$)",
                color=color_jk, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Triple
        ax1.bar(x + bar_positions[6], [d["f_ijk"] for d in plot_data], width,
                yerr=[d["se_ijk"] if not np.isnan(d["se_ijk"]) else 0 for d in plot_data],
                label=r"$f_{ijk}$ (6$\Delta$)",
                color=color_ijk, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Plot interaction values on same axis with hatching patterns (matching digenic style)
        # Digenic interactions - use hatching to distinguish from fitness bars
        ax1.bar(x + bar_positions[7], [d["delta_ij"] for d in plot_data], width,
                yerr=[d["se_delta_ij"] if not np.isnan(d["se_delta_ij"]) else 0 for d in plot_data],
                label=r"$\delta_{ij}$",
                color=color_delta_ij, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[8], [d["delta_ik"] for d in plot_data], width,
                yerr=[d["se_delta_ik"] if not np.isnan(d["se_delta_ik"]) else 0 for d in plot_data],
                label=r"$\delta_{ik}$",
                color=color_delta_ik, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[9], [d["delta_jk"] for d in plot_data], width,
                yerr=[d["se_delta_jk"] if not np.isnan(d["se_delta_jk"]) else 0 for d in plot_data],
                label=r"$\delta_{jk}$",
                color=color_delta_jk, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Trigenic interaction with different pattern
        ax1.bar(x + bar_positions[10], [d["sigma"] for d in plot_data], width,
                yerr=[d["se_sigma"] if not np.isnan(d["se_sigma"]) else 0 for d in plot_data],
                label=r"$\sigma_{ijk}$",
                color=color_sigma, alpha=0.9, capsize=3,
                hatch='xxx', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Add reference lines
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)

        # Labels and title
        ax1.set_xlabel("Gene Triples", fontsize=14)
        ax1.set_ylabel("Fitness / Interaction Score", fontsize=14)
        ax1.set_title(f"Top {n_triples} Trigenic Interactions for {ffa} (by |\u03c3|)",
                     fontsize=16, fontweight="bold")

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels([d["triple"] for d in plot_data], rotation=45, ha="right", fontsize=9)

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
        ax1.set_yticklabels([f'{int(t)}' for t in y_ticks])

        # Adjust layout
        plt.tight_layout()

        # Save figure with filename matching script name
        filename = f"additive_trigenic_interaction_bar_plots_triple_suppression_{ffa.replace(':', '')}_{timestamp()}.png"
        filepath = osp.join(ASSET_IMAGES_DIR, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Trigenic interaction bar plot for {ffa} saved to:")
        print(f"  {filepath}")

        # Print summary statistics for this FFA
        sigma_values = [d["sigma"] for d in plot_data]
        print(f"\n{ffa} Summary (Top {n_triples} Trigenic TF interactions):")
        print(f"  Mean σ: {np.mean(sigma_values):.3f}")
        print(f"  Min σ: {np.min(sigma_values):.3f}")
        print(f"  Max σ: {np.max(sigma_values):.3f}")
        print(f"  Negative interactions: {sum(t < 0 for t in sigma_values)}/{n_triples}")
        print(f"  Positive interactions: {sum(t > 0 for t in sigma_values)}/{n_triples}")


def create_recovery_pattern_plots(
    recovery_patterns, digenic_interactions,
    single_se, double_se, triple_se,
    digenic_se, trigenic_se, columns,
    trigenic_interactions=None, double_mutants=None, triple_mutants=None,
    single_mutants=None
):
    """Create bar plots specifically for recovery pattern interactions, including interaction values."""

    if not recovery_patterns:
        print("No recovery patterns found!")
        return

    # Group by FFA type
    ffa_groups = {}
    for pattern in recovery_patterns:
        ffa = pattern['ffa']
        if ffa not in ffa_groups:
            ffa_groups[ffa] = []
        ffa_groups[ffa].append(pattern)

    # Create a plot for each FFA with recovery patterns
    for ffa, patterns in ffa_groups.items():
        # Sort patterns by triple recovery strength and limit to top 10
        patterns = sorted(patterns, key=lambda x: x['triple_recovery'], reverse=True)[:10]
        n_patterns = len(patterns)
        fig_width = max(16, n_patterns * 2.5)  # Wider to accommodate 11 bars
        fig, ax1 = plt.subplots(figsize=(fig_width, 10))

        # Get interaction values for each pattern
        for pattern in patterns:
            genes = pattern['genes']
            ffa_idx = pattern['ffa_idx']

            # Get digenic interactions
            pattern['delta_ij'] = digenic_interactions.get((genes[0], genes[1]), [np.nan]*6)[ffa_idx]
            pattern['delta_ik'] = digenic_interactions.get((genes[0], genes[2]), [np.nan]*6)[ffa_idx]
            pattern['delta_jk'] = digenic_interactions.get((genes[1], genes[2]), [np.nan]*6)[ffa_idx]

        x = np.arange(n_patterns)
        width = 0.08  # Thinner bars to fit 11
        n_bars = 11  # 7 fitness + 4 interaction bars

        # Colors matching the main trigenic plot
        color_i = "#7191A9"  # Light blue
        color_j = "#6B8D3A"  # Green
        color_k = "#B73C39"  # Red
        color_ij = "#34699D"  # Deep blue
        color_ik = "#E6A65D"  # Orange
        color_jk = "#775A9F"  # Purple
        color_ijk = "#CC8250"  # Burnt orange

        # Interaction colors - same as corresponding fitness
        color_delta_ij = color_ij
        color_delta_ik = color_ik
        color_delta_jk = color_jk
        color_sigma = color_ijk

        # Calculate bar positions
        bar_positions = np.linspace(-width * (n_bars-1)/2, width * (n_bars-1)/2, n_bars)

        # Get standard errors for patterns
        for pattern in patterns:
            genes = pattern['genes']
            ffa_idx = pattern['ffa_idx']

            # Get standard errors
            pattern['se_i'] = single_se.get(genes[0], [0]*6)[ffa_idx] if single_se else 0
            pattern['se_j'] = single_se.get(genes[1], [0]*6)[ffa_idx] if single_se else 0
            pattern['se_k'] = single_se.get(genes[2], [0]*6)[ffa_idx] if single_se else 0

            pattern['se_ij'] = double_se.get((genes[0], genes[1]), [0]*6)[ffa_idx] if double_se else 0
            pattern['se_ik'] = double_se.get((genes[0], genes[2]), [0]*6)[ffa_idx] if double_se else 0
            pattern['se_jk'] = double_se.get((genes[1], genes[2]), [0]*6)[ffa_idx] if double_se else 0

            pattern['se_ijk'] = triple_se.get((genes[0], genes[1], genes[2]), [0]*6)[ffa_idx] if triple_se else 0

            pattern['se_delta_ij'] = digenic_se.get((genes[0], genes[1]), [0]*6)[ffa_idx] if digenic_se else 0
            pattern['se_delta_ik'] = digenic_se.get((genes[0], genes[2]), [0]*6)[ffa_idx] if digenic_se else 0
            pattern['se_delta_jk'] = digenic_se.get((genes[1], genes[2]), [0]*6)[ffa_idx] if digenic_se else 0

            pattern['se_sigma'] = trigenic_se.get((genes[0], genes[1], genes[2]), [0]*6)[ffa_idx] if trigenic_se else 0

        # Plot fitness values on primary y-axis with error bars (matching digenic style)
        ax1.bar(x + bar_positions[0], [p['f_i'] for p in patterns], width,
                yerr=[p.get('se_i', 0) if not np.isnan(p.get('se_i', 0)) else 0 for p in patterns],
                label=r"$f_i$ (4$\Delta$)",
                color=color_i, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[1], [p['f_j'] for p in patterns], width,
                yerr=[p.get('se_j', 0) if not np.isnan(p.get('se_j', 0)) else 0 for p in patterns],
                label=r"$f_j$ (4$\Delta$)",
                color=color_j, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[2], [p['f_k'] for p in patterns], width,
                yerr=[p.get('se_k', 0) if not np.isnan(p.get('se_k', 0)) else 0 for p in patterns],
                label=r"$f_k$ (4$\Delta$)",
                color=color_k, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[3], [p['f_ij'] for p in patterns], width,
                yerr=[p.get('se_ij', 0) if not np.isnan(p.get('se_ij', 0)) else 0 for p in patterns],
                label=r"$f_{ij}$ (5$\Delta$)",
                color=color_ij, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[4], [p['f_ik'] for p in patterns], width,
                yerr=[p.get('se_ik', 0) if not np.isnan(p.get('se_ik', 0)) else 0 for p in patterns],
                label=r"$f_{ik}$ (5$\Delta$)",
                color=color_ik, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[5], [p['f_jk'] for p in patterns], width,
                yerr=[p.get('se_jk', 0) if not np.isnan(p.get('se_jk', 0)) else 0 for p in patterns],
                label=r"$f_{jk}$ (5$\Delta$)",
                color=color_jk, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[6], [p['f_ijk'] for p in patterns], width,
                yerr=[p.get('se_ijk', 0) if not np.isnan(p.get('se_ijk', 0)) else 0 for p in patterns],
                label=r"$f_{ijk}$ (6$\Delta$)",
                color=color_ijk, alpha=0.8, capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Plot interaction values with hatching patterns and error bars (matching digenic style)
        ax1.bar(x + bar_positions[7], [p.get('delta_ij', 0) for p in patterns], width,
                yerr=[p.get('se_delta_ij', 0) if not np.isnan(p.get('se_delta_ij', 0)) else 0 for p in patterns],
                label=r"$\delta_{ij}$",
                color=color_delta_ij, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[8], [p.get('delta_ik', 0) for p in patterns], width,
                yerr=[p.get('se_delta_ik', 0) if not np.isnan(p.get('se_delta_ik', 0)) else 0 for p in patterns],
                label=r"$\delta_{ik}$",
                color=color_delta_ik, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[9], [p.get('delta_jk', 0) for p in patterns], width,
                yerr=[p.get('se_delta_jk', 0) if not np.isnan(p.get('se_delta_jk', 0)) else 0 for p in patterns],
                label=r"$\delta_{jk}$",
                color=color_delta_jk, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[10], [p['sigma'] for p in patterns], width,
                yerr=[p.get('se_sigma', 0) if not np.isnan(p.get('se_sigma', 0)) else 0 for p in patterns],
                label=r"$\sigma_{ijk}$",
                color=color_sigma, alpha=0.9, capsize=3,
                hatch='xxx', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Reference lines
        ax1.axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        # Labels and title
        ax1.set_xlabel("Gene Triples", fontsize=14)
        ax1.set_ylabel("Fitness / Interaction Score", fontsize=14)
        ax1.set_title(f"Recovery Patterns for {ffa}: Singles & Triple > Doubles",
                     fontsize=16, fontweight="bold")

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels([p['triple'] for p in patterns], rotation=45, ha="right", fontsize=9)

        # Color the y-axis labels

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(
                  loc="upper left", fontsize=9, ncol=2)

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
        ax1.set_yticklabels([f'{int(t)}' for t in y_ticks])

        # Adjust layout
        plt.tight_layout()

        # Save figure with filename matching script name
        filename = f"additive_trigenic_interaction_bar_plots_triple_suppression_recovery_{ffa.replace(':', '')}_{timestamp()}.png"
        filepath = osp.join(ASSET_IMAGES_DIR, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Trigenic recovery pattern plot for {ffa} saved to:")
        print(f"  {filepath}")

        # Print summary for this FFA
        print(f"\n{ffa} Recovery Patterns:")
        print(f"  Showing top {n_patterns} patterns (max 10)")
        print(f"  Mean triple recovery: {np.mean([p['triple_recovery'] for p in patterns]):.3f}")
        print(f"  Max triple recovery: {max(p['triple_recovery'] for p in patterns):.3f}")

    # Create summary plot across all FFAs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Percentage of recovered triples by FFA
    ax = axes[0]
    ffa_counts = {ffa: len(patterns) for ffa, patterns in ffa_groups.items()}
    ffa_types = columns  # Include Total Titer

    # Calculate total number of trigenic interactions
    # This should be the same for each FFA type
    total_triples = 120  # Total number of trigenic combinations

    # Calculate percentages
    counts = [ffa_counts.get(ffa, 0) for ffa in ffa_types]
    percentages = [(count / total_triples * 100) for count in counts]

    bars = ax.bar(range(len(ffa_types)), percentages, color='#7191A9', alpha=0.8)
    ax.set_xticks(range(len(ffa_types)))
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_xlabel('FFA Type', fontsize=12)
    ax.set_ylabel('Percentage of Triples with Recovery Pattern (%)', fontsize=12)
    ax.set_title('Recovery Pattern Frequency by FFA', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to show percentage scale
    ax.set_ylim(0, max(percentages) * 1.2 if percentages else 20)

    # Add value labels above bars showing both percentage and count
    for bar, pct, cnt in zip(bars, percentages, counts):
        if cnt > 0:
            # Position text above the bar
            y_pos = bar.get_height() + 0.3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{pct:.1f}%\n({cnt})', ha='center', va='bottom',
                   fontweight='bold', color='black', fontsize=10)

    # Plot 2: Compare triple mutant fitness for different categories
    # Plot f_ijk vs max(f_ij, f_ik, f_jk) for recovery pattern identification
    ax = axes[1]

    # Collect all data points for plotting
    total_plotted_points = 0
    points_above_diagonal = 0

    for ffa_idx, ffa in enumerate(ffa_types):
        max_doubles = []
        triples = []

        # Go through ALL trigenic interactions for this FFA
        for (gene1, gene2, gene3), _ in trigenic_interactions.items():
            # Get single mutant fitnesses
            f_i = single_mutants.get(gene1, [np.nan]*6)[ffa_idx]
            f_j = single_mutants.get(gene2, [np.nan]*6)[ffa_idx]
            f_k = single_mutants.get(gene3, [np.nan]*6)[ffa_idx]

            # Get double mutant fitnesses
            f_ij = double_mutants.get((gene1, gene2), [np.nan]*6)[ffa_idx]
            f_ik = double_mutants.get((gene1, gene3), [np.nan]*6)[ffa_idx]
            f_jk = double_mutants.get((gene2, gene3), [np.nan]*6)[ffa_idx]

            # Get triple mutant fitness
            f_ijk = triple_mutants.get((gene1, gene2, gene3), [np.nan]*6)[ffa_idx]

            # Only process if all values are valid
            if not any(np.isnan([f_i, f_j, f_k, f_ij, f_ik, f_jk, f_ijk])):
                min_single = min(f_i, f_j, f_k)
                max_double = max(f_ij, f_ik, f_jk)

                # Only include data where max double < min single (recovery pattern criteria)
                if max_double < min_single:
                    max_doubles.append(max_double)
                    triples.append(f_ijk)
                    total_plotted_points += 1

                    # Count points above diagonal (f_ijk > max_double)
                    if f_ijk > max_double:
                        points_above_diagonal += 1

        # Plot all points for this FFA with same size
        if max_doubles:
            ax.scatter(max_doubles, triples,
                      label=ffa, alpha=0.7, s=30)

    # Add diagonal reference line (y=x)
    plot_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    plot_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='y=x')

    # Add reference lines at fitness = 1
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=1, color='gray', linestyle=':', alpha=0.3)

    ax.set_xlabel(r'Best Double Mutant Fitness (max($f_{ij}$, $f_{ik}$, $f_{jk}$))', fontsize=12)
    ax.set_ylabel(r'Triple Mutant Fitness ($f_{ijk}$)', fontsize=12)

    # Calculate percentage for title
    total_possible = 6 * 120  # 6 FFAs, 120 mutants each
    percentage_above = (points_above_diagonal / total_possible) * 100

    ax.set_title(r'Trigenic Recovery: $f_{ijk}$ vs $\mathrm{max}(f_{ij}, f_{ik}, f_{jk})$' +
                 f'\n{points_above_diagonal}/{total_possible} mutant-labels ({percentage_above:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation to explain the plot with counts
    total_possible = 6 * 120  # 6 FFAs, 120 mutants each
    percentage_above = (points_above_diagonal / total_possible) * 100

    ax.text(0.02, 0.98,
           f'Points above diagonal:\nTriple mutant recovers from\nbest double mutant defects\n\n' +
           f'{points_above_diagonal}/{total_possible} ({percentage_above:.1f}%)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Trigenic Recovery Patterns: Singles & Triple Rescue Double Defects',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save summary figure with filename matching script name
    filename = f"additive_trigenic_interaction_bar_plots_triple_suppression_recovery_summary_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nRecovery summary plot saved to:")
    print(f"  {filepath}")

    # Count unique strains showing recovery pattern
    unique_strains_with_pattern = set()
    for pattern in recovery_patterns:
        unique_strains_with_pattern.add(pattern['genes'])

    n_unique_strains = len(unique_strains_with_pattern)

    print(f"\n=== Overall Recovery Pattern Summary ===")
    print(f"Total recovery pattern instances: {len(recovery_patterns)}")
    print(f"Unique strains with recovery pattern: {n_unique_strains}/120 ({n_unique_strains/120*100:.1f}%)")
    print(f"FFAs with patterns: {list(ffa_groups.keys())}")


def create_trigenic_summary_plot(trigenic_interactions, triple_mutants, columns,
                                abbreviations=None, top_n=50):
    """Create a summary plot showing top trigenic interactions across all FFAs."""

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create reverse mapping from letter to gene name if abbreviations provided
    letter_to_gene = {}
    if abbreviations:
        letter_to_gene = {v: k for k, v in abbreviations.items()}
        if 'F' not in letter_to_gene:
            letter_to_gene['F'] = 'PKH1'

    # Prepare data for summary
    summary_data = []

    for ffa_idx, ffa in enumerate(ffa_types):
        for (gene1, gene2, gene3), tau_values in trigenic_interactions.items():
            if ffa_idx < len(tau_values):
                sigma = tau_values[ffa_idx]
                f_ijk = triple_mutants.get((gene1, gene2, gene3), [np.nan]*6)[ffa_idx]

                if not np.isnan(sigma) and not np.isnan(f_ijk):
                    # Convert letters to gene names
                    display_gene1 = letter_to_gene.get(gene1, gene1) if letter_to_gene else gene1
                    display_gene2 = letter_to_gene.get(gene2, gene2) if letter_to_gene else gene2
                    display_gene3 = letter_to_gene.get(gene3, gene3) if letter_to_gene else gene3

                    summary_data.append({
                        "ffa": ffa,
                        "triple": f"{display_gene1}-{display_gene2}-{display_gene3}",
                        "sigma": sigma,
                        "f_ijk": f_ijk,
                        "abs_sigma": abs(sigma)
                    })

    # Convert to DataFrame
    df = pd.DataFrame(summary_data)

    # Get top interactions by absolute tau
    top_interactions = df.nlargest(top_n, 'abs_sigma')

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Top interactions heatmap-style
    ax = axes[0]

    # Pivot data for heatmap
    pivot_data = top_interactions.pivot_table(
        index='triple', columns='ffa', values='sigma', fill_value=0
    )

    # Sort by mean absolute sigma
    row_order = pivot_data.abs().mean(axis=1).sort_values(ascending=False).index
    pivot_data = pivot_data.loc[row_order]

    # Create heatmap
    im = ax.imshow(pivot_data.values, aspect='auto', cmap='RdBu_r',
                   vmin=-pivot_data.abs().max().max(),
                   vmax=pivot_data.abs().max().max())

    # Set ticks
    ax.set_xticks(np.arange(len(ffa_types)))
    ax.set_yticks(np.arange(len(pivot_data)))
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.index, fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=ax, label=r'$\sigma$ value')

    ax.set_title(f"Top {min(top_n, len(pivot_data))} Trigenic Interactions by |\u03c3|",
                fontsize=14, fontweight="bold")
    ax.set_xlabel("FFA Type", fontsize=12)
    ax.set_ylabel("Gene Triples", fontsize=12)

    # Plot 2: Distribution of sigma values
    ax = axes[1]

    # Plot histogram for each FFA
    for ffa in ffa_types:
        ffa_data = df[df["ffa"] == ffa]
        ax.hist(ffa_data["sigma"], alpha=0.6, bins=30, label=ffa,
               edgecolor="black", linewidth=0.5)

    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xlabel(r"Trigenic Interaction Score ($\sigma$)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Distribution of All Trigenic TF Interactions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Summary of Trigenic TF Interactions", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save figure with filename matching script name
    filename = f"additive_trigenic_interaction_bar_plots_triple_suppression_summary_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nTrigenic summary plot saved to:")
    print(f"  {filepath}")

    # Print overall statistics
    print("\nOverall Trigenic Summary Statistics:")
    print(f"Total trigenic interactions: {len(df)}")
    print(f"Mean σ across all FFAs: {df['sigma'].mean():.3f}")
    print(f"Std σ: {df['sigma'].std():.3f}")
    print(f"Percentage negative: {100 * (df['sigma'] < 0).sum() / len(df):.1f}%")
    print(f"Percentage positive: {100 * (df['sigma'] > 0).sum() / len(df):.1f}%")
    print(f"\nTop 5 strongest interactions (by |�|):")
    for _, row in df.nlargest(5, 'abs_sigma').iterrows():
        print(f"  {row['triple']} ({row['ffa']}): σ = {row['sigma']:.3f}")


def main():
    """Main function to create trigenic interaction bar plots focusing on recovery patterns."""

    print("Loading FFA data...")
    file_path = "/Users/michaelvolk/Documents/projects/torchcell/data/torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx"
    raw_df, abbreviations, replicate_dict = load_ffa_data(file_path)

    print("Normalizing by positive control (+ve Ctrl)...")
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
    ) = compute_additive_interactions_with_error_propagation(
        normalized_df, normalized_replicates, abbreviations
    )

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    # Find recovery patterns
    print("\nSearching for recovery patterns...")
    recovery_patterns = find_recovery_patterns(
        trigenic_interactions,
        single_mutants,
        double_mutants,
        triple_mutants,
        columns,
        abbreviations
    )

    print(f"Found {len(recovery_patterns)} recovery patterns")

    # Create recovery pattern plots
    create_recovery_pattern_plots(
        recovery_patterns, digenic_interactions,
        single_se, double_se, triple_se,
        digenic_se, trigenic_se, columns,
        trigenic_interactions, double_mutants, triple_mutants,
        single_mutants
    )

    # Summary plot disabled
    # print("\nCreating summary plots...")
    # create_trigenic_summary_plot(
    #     trigenic_interactions, triple_mutants, columns,
    #     abbreviations, top_n=50
    # )

    print(f"\nAll trigenic interaction plots complete! Check the file paths above to view the plots.")


if __name__ == "__main__":
    main()