# experiments/008-xue-ffa/scripts/trigenic_interaction_bar_plots_triple_suppression_relaxed
# [[experiments.008-xue-ffa.scripts.trigenic_interaction_bar_plots_triple_suppression_relaxed]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/trigenic_interaction_bar_plots_triple_suppression_relaxed
# Test file: experiments/008-xue-ffa/scripts/test_trigenic_interaction_bar_plots_triple_suppression_relaxed.py

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


def find_recovery_patterns(
    trigenic_interactions, single_mutants, double_mutants, triple_mutants,
    columns, abbreviations=None
):
    """Find trigenic interactions with the relaxed recovery pattern:
    - Triple (f_ijk) > max(f_i, f_j, f_k, f_ij, f_ik, f_jk)

    This represents cases where the triple mutation shows recovery
    compared to all single and double mutant combinations.
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
                    # Find max of all singles and doubles
                    max_all_others = max(f_i, f_j, f_k, f_ij, f_ik, f_jk)

                    # Check the relaxed recovery pattern: triple > all others
                    if f_ijk > max_all_others:
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
                            'max_all_others': max_all_others,
                            'triple_recovery': f_ijk - max_all_others,
                            'tau': tau_values[ffa_idx]
                        })

    return recovery_patterns


def create_recovery_pattern_plots(
    recovery_patterns, digenic_interactions,
    single_se, double_se, triple_se,
    digenic_se, trigenic_se, columns
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
            pattern['eps_ij'] = digenic_interactions.get((genes[0], genes[1]), [np.nan]*6)[ffa_idx]
            pattern['eps_ik'] = digenic_interactions.get((genes[0], genes[2]), [np.nan]*6)[ffa_idx]
            pattern['eps_jk'] = digenic_interactions.get((genes[1], genes[2]), [np.nan]*6)[ffa_idx]

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
        color_eps_ij = color_ij
        color_eps_ik = color_ik
        color_eps_jk = color_jk
        color_tau = color_ijk

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

            pattern['se_eps_ij'] = digenic_se.get((genes[0], genes[1]), [0]*6)[ffa_idx] if digenic_se else 0
            pattern['se_eps_ik'] = digenic_se.get((genes[0], genes[2]), [0]*6)[ffa_idx] if digenic_se else 0
            pattern['se_eps_jk'] = digenic_se.get((genes[1], genes[2]), [0]*6)[ffa_idx] if digenic_se else 0

            pattern['se_tau'] = trigenic_se.get((genes[0], genes[1], genes[2]), [0]*6)[ffa_idx] if trigenic_se else 0

        # Plot fitness values on primary y-axis with error bars (matching main script style)
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

        # Plot interaction values on same axis with hatching patterns (matching main script style)
        ax1.bar(x + bar_positions[7], [p.get('eps_ij', 0) for p in patterns], width,
                yerr=[p.get('se_eps_ij', 0) if not np.isnan(p.get('se_eps_ij', 0)) else 0 for p in patterns],
                label=r"$\varepsilon_{ij}$",
                color=color_eps_ij, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[8], [p.get('eps_ik', 0) for p in patterns], width,
                yerr=[p.get('se_eps_ik', 0) if not np.isnan(p.get('se_eps_ik', 0)) else 0 for p in patterns],
                label=r"$\varepsilon_{ik}$",
                color=color_eps_ik, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[9], [p.get('eps_jk', 0) for p in patterns], width,
                yerr=[p.get('se_eps_jk', 0) if not np.isnan(p.get('se_eps_jk', 0)) else 0 for p in patterns],
                label=r"$\varepsilon_{jk}$",
                color=color_eps_jk, alpha=0.7, capsize=3,
                hatch='///', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        ax1.bar(x + bar_positions[10], [p['tau'] for p in patterns], width,
                yerr=[p.get('se_tau', 0) if not np.isnan(p.get('se_tau', 0)) else 0 for p in patterns],
                label=r"$\tau_{ijk}$",
                color=color_tau, alpha=0.9, capsize=3,
                hatch='xxx', edgecolor='black', linewidth=0.5,
                error_kw={'linewidth': 1, 'alpha': 0.7})

        # Reference lines
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axhline(y=1, color="black", linestyle="--", alpha=0.3, linewidth=1)

        # Labels and title
        ax1.set_xlabel("Gene Triples", fontsize=14)
        ax1.set_ylabel("Fitness / Interaction Score", fontsize=14)
        ax1.set_title(f"Relaxed Recovery Patterns for {ffa}: Triple > All Singles & Doubles",
                     fontsize=16, fontweight="bold")

        # X-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels([p['triple'] for p in patterns], rotation=45, ha="right", fontsize=9)

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

        # Save figure
        # Replace spaces and colons in filename
        safe_ffa = ffa.replace(':', '').replace(' ', '_')
        filename = f"multiplicative_trigenic_suppression_relaxed_{safe_ffa}.png"
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        filepath = osp.join(ffa_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Trigenic suppression (relaxed) plot for {ffa} saved to:")
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
    ax.set_title('Relaxed Recovery Pattern Frequency by FFA', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-axis to show percentage scale
    ax.set_ylim(0, max(percentages) * 1.2 if percentages else 50)

    # Add value labels above bars showing both percentage and count
    for bar, pct, cnt in zip(bars, percentages, counts):
        if cnt > 0:
            # Position text above the bar
            y_pos = bar.get_height() + 0.3
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{pct:.1f}%\n({cnt})', ha='center', va='bottom',
                   fontweight='bold', color='black', fontsize=10)

    # Plot 2: Recovery strength scatter
    ax = axes[1]
    total_points = 0

    for ffa, patterns in ffa_groups.items():
        max_others = [p['max_all_others'] for p in patterns]
        triple_fitness = [p['f_ijk'] for p in patterns]
        ax.scatter(max_others, triple_fitness, label=ffa, alpha=0.6, s=50)
        total_points += len(patterns)

    # Add diagonal line (y=x) to show recovery threshold
    lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lims, lims, 'k-', alpha=0.3, zorder=0, linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    # Update x-axis label with proper subscripts
    ax.set_xlabel(r'Max($f_i$, $f_j$, $f_k$, $f_{ij}$, $f_{ik}$, $f_{jk}$) Fitness', fontsize=12)
    ax.set_ylabel(r'Triple Mutant Fitness ($f_{ijk}$)', fontsize=12)

    # Calculate total percentage based on strain-FFA combinations
    total_possible = 6 * 120  # 6 FFAs (including Total Titer), 120 mutants each
    overall_percentage = (total_points / total_possible) * 100

    ax.set_title(r'Relaxed Recovery Pattern: $f_{ijk} > \mathrm{Max}(f_i, f_j, f_k, f_{ij}, f_{ik}, f_{jk})$' +
                 f'\n{total_points}/{total_possible} mutant-labels ({overall_percentage:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add integer ticks for both axes
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Set integer x-ticks
    x_min_int = int(np.floor(x_min))
    x_max_int = int(np.ceil(x_max))
    x_ticks = np.arange(x_min_int, x_max_int + 1, 1 if x_max_int - x_min_int <= 10 else 2)
    ax.set_xticks(x_ticks)

    # Set integer y-ticks
    y_min_int = int(np.floor(y_min))
    y_max_int = int(np.ceil(y_max))
    y_ticks = np.arange(y_min_int, y_max_int + 1, 1 if y_max_int - y_min_int <= 10 else 2)
    ax.set_yticks(y_ticks)

    plt.suptitle('Relaxed Trigenic Recovery: Triple Mutant > All Singles & Doubles',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save summary figure
    filename = "multiplicative_trigenic_suppression_summary_relaxed.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nTrigenic suppression summary (relaxed) plot saved to:")
    print(f"  {filepath}")

    print(f"\n=== Overall Relaxed Recovery Pattern Summary ===")
    print(f"Total recovery patterns found (f_ijk > max(all others)): {len(recovery_patterns)}")
    print(f"Percentage of all trigenic interactions showing recovery: {len(recovery_patterns)/120*100:.1f}% ({len(recovery_patterns)}/120)")
    print(f"FFAs with patterns: {list(ffa_groups.keys())}")


def main():
    """Main function to create trigenic interaction bar plots focusing on relaxed recovery patterns."""

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
    ) = compute_interactions_with_error_propagation(
        normalized_df, normalized_replicates, abbreviations
    )

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    # Find recovery patterns
    print("\nSearching for relaxed recovery patterns (f_ijk > max(all others))...")
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
        digenic_se, trigenic_se, columns
    )

    print(f"\nAll trigenic suppression (relaxed) plots complete! Check the file paths above to view the plots.")


if __name__ == "__main__":
    main()