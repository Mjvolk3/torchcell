# experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_distribution_comparison
# [[experiments.008-xue-ffa.scripts.additive_best_titers_per_ffa_distribution_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_distribution_comparison
# Test file: experiments/008-xue-ffa/scripts/test_additive_best_titers_per_ffa_distribution_comparison.py


import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.style as mplstyle
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import warnings
import seaborn as sns

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


def create_distribution_comparison(
    single_mutants, double_mutants, triple_mutants,
    columns, abbreviations=None
):
    """Create distribution plots comparing singles, doubles, and triples."""

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create figure with subplots for each FFA
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ffa_idx, ffa in enumerate(ffa_types):
        ax = axes[ffa_idx]

        # Collect fitness values by type
        singles = []
        doubles = []
        triples = []

        # Singles
        for gene, fitness_values in single_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                singles.append(fitness_values[ffa_idx])

        # Doubles
        for (gene1, gene2), fitness_values in double_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                doubles.append(fitness_values[ffa_idx])

        # Triples
        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                triples.append(fitness_values[ffa_idx])

        # Create violin plot
        data_to_plot = []
        labels = []

        if singles:
            data_to_plot.append(singles)
            labels.append(f'Singles\n(n={len(singles)})')
        if doubles:
            data_to_plot.append(doubles)
            labels.append(f'Doubles\n(n={len(doubles)})')
        if triples:
            data_to_plot.append(triples)
            labels.append(f'Triples\n(n={len(triples)})')

        # Create violin plot (without mean/median lines to avoid confusion)
        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                              widths=0.7, showmeans=False, showmedians=False)

        # Customize colors
        colors = ['#7191A9', '#34699D', '#CC8250']  # Light blue, deep blue, burnt orange
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Add box plot overlay with mean
        bp = ax.boxplot(data_to_plot, positions=range(len(data_to_plot)),
                        widths=0.3, patch_artist=False,
                        showmeans=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='#B73C39'),  # Red from palette
                        meanprops=dict(marker='D', markerfacecolor='#775A9F',  # Purple from palette
                                     markeredgecolor='black', markersize=8, markeredgewidth=1.5))

        # Add reference lines
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Fitness (relative to +ve Ctrl)', fontsize=12)
        ax.set_title(f'{ffa}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistical summary text
        if singles and doubles and triples:
            stats_text = (
                f"Means: {np.mean(singles):.2f}, {np.mean(doubles):.2f}, {np.mean(triples):.2f}\n"
                f"Max: {np.max(singles):.2f}, {np.max(doubles):.2f}, {np.max(triples):.2f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # All subplots are now used (6 total)
    # Add legend below the plot to avoid overlap with Total Titer
    legend_elements = [
        Line2D([0], [0], color='#B73C39', linewidth=2, label='Median'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#775A9F',
                   markeredgecolor='black', markersize=8, label='Mean'),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1.5,
                     label='IQR (25th-75th percentile)'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='-', label='Whiskers (1.5×IQR)')
    ]

    # Add legend below the plots
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
              title='Statistical Elements', title_fontsize=11,
              bbox_to_anchor=(0.5, -0.05))

    plt.suptitle('Distribution Comparison: Singles vs Doubles vs Triples - ADDITIVE MODEL',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = "additive_distribution_comparison.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Distribution comparison plot saved to:")
    print(f"  {filepath}")


def create_triple_vs_best_double_plot(
    double_mutants, triple_mutants,
    columns, abbreviations=None,
    single_mutants=None
):
    """Plot triple mutant fitness vs their best constituent double."""

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Create standard 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot data for each FFA
    for ffa_idx, ffa in enumerate(ffa_types):
        ax = axes[ffa_idx]

    for ffa_idx, ffa in enumerate(ffa_types):
        ax = axes[ffa_idx]

        best_doubles = []
        triple_fitness = []
        improvements = []

        # For each triple, find its best double
        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                f_ijk = fitness_values[ffa_idx]

                # Get constituent doubles
                f_ij = double_mutants.get((gene1, gene2), [np.nan]*6)[ffa_idx]
                f_ik = double_mutants.get((gene1, gene3), [np.nan]*6)[ffa_idx]
                f_jk = double_mutants.get((gene2, gene3), [np.nan]*6)[ffa_idx]

                if not any(np.isnan([f_ij, f_ik, f_jk])):
                    best_double = max(f_ij, f_ik, f_jk)
                    best_doubles.append(best_double)
                    triple_fitness.append(f_ijk)
                    improvements.append(f_ijk > best_double)

        if best_doubles:
            # Create scatter plot
            colors = ['#CC8250' if imp else '#7191A9' for imp in improvements]
            ax.scatter(best_doubles, triple_fitness, c=colors, alpha=0.6, s=50)

            # Add diagonal reference line
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                   max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')

            # Add reference lines
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
            ax.axvline(x=1, color='gray', linestyle=':', alpha=0.3)

            # Calculate statistics
            n_improved = sum(improvements)
            pct_improved = (n_improved / len(improvements)) * 100

            # Labels
            ax.set_xlabel('Best Constituent Double Fitness', fontsize=12)
            ax.set_ylabel('Triple Mutant Fitness', fontsize=12)
            ax.set_title(f'{ffa}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = (
                f"Triples > Best Double: {n_improved}/{len(improvements)} ({pct_improved:.1f}%)\n"
                f"Mean improvement: {np.mean([t-d for t,d in zip(triple_fitness, best_doubles)]):.3f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # All 6 subplots are used for the 6 FFAs including Total Titer

    plt.suptitle('Triple Mutant Performance vs Best Constituent Double - ADDITIVE MODEL',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = "additive_triple_vs_best_double.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Triple vs best double plot saved to:")
    print(f"  {filepath}")


def analyze_top_performer_composition(
    single_mutants, double_mutants, triple_mutants,
    columns, percentile=90
):
    """Analyze the composition of top performers by percentile.

    Args:
        percentile: The percentile cutoff (e.g., 90 means top 10%)
    """

    # Use all FFA types including Total Titer
    ffa_types = columns

    # Prepare data for plotting
    composition_data = []

    # Calculate actual numbers from the data
    n_singles = len(single_mutants)
    n_doubles = len(double_mutants)
    n_triples = len(triple_mutants)
    total_strains = n_singles + n_doubles + n_triples

    print(f"Detected genotype counts from data:")
    print(f"  Singles: {n_singles}")
    print(f"  Doubles: {n_doubles}")
    print(f"  Triples: {n_triples}")
    print(f"  Total: {total_strains}")

    # Debug: print actual single mutant keys
    print(f"\nSingle mutant genes detected: {list(single_mutants.keys())}")

    # Calculate how many strains are in top (100-percentile)%
    top_percentage = 100 - percentile
    n_top_strains = int(np.ceil(total_strains * (top_percentage / 100)))

    for ffa_idx, ffa in enumerate(ffa_types):
        # Collect all genotypes with their fitness
        all_genotypes = []

        # Singles
        for gene, fitness_values in single_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                all_genotypes.append({
                    'type': 'single',
                    'fitness': fitness_values[ffa_idx]
                })

        # Doubles
        for (gene1, gene2), fitness_values in double_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                all_genotypes.append({
                    'type': 'double',
                    'fitness': fitness_values[ffa_idx]
                })

        # Triples
        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                all_genotypes.append({
                    'type': 'triple',
                    'fitness': fitness_values[ffa_idx]
                })

        # Sort and get top performers by percentile
        all_genotypes.sort(key=lambda x: x['fitness'], reverse=True)
        top_genotypes = all_genotypes[:n_top_strains]

        # Count types in top performers
        single_count = sum(1 for g in top_genotypes if g['type'] == 'single')
        double_count = sum(1 for g in top_genotypes if g['type'] == 'double')
        triple_count = sum(1 for g in top_genotypes if g['type'] == 'triple')

        composition_data.append({
            'ffa': ffa,
            'singles': single_count,
            'doubles': double_count,
            'triples': triple_count,
            'single_pct': (single_count / n_singles) * 100,  # Normalized by actual counts
            'double_pct': (double_count / n_doubles) * 100,
            'triple_pct': (triple_count / n_triples) * 100
        })

    # Create stacked bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Raw counts
    x = np.arange(len(ffa_types))
    width = 0.6

    singles = [d['singles'] for d in composition_data]
    doubles = [d['doubles'] for d in composition_data]
    triples = [d['triples'] for d in composition_data]

    p1 = ax1.bar(x, singles, width, label='Singles', color='#7191A9', alpha=0.8)
    p2 = ax1.bar(x, doubles, width, bottom=singles, label='Doubles', color='#34699D', alpha=0.8)
    p3 = ax1.bar(x, triples, width, bottom=np.array(singles)+np.array(doubles),
                label='Triples', color='#CC8250', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax1.set_ylabel(f'Count in Top {top_percentage}% (n={n_top_strains})', fontsize=12)
    ax1.set_title(f'Top {top_percentage}% Performer Composition (n={n_top_strains} strains)',
                 fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Normalized percentages (enrichment)
    single_pcts = [d['single_pct'] for d in composition_data]
    double_pcts = [d['double_pct'] for d in composition_data]
    triple_pcts = [d['triple_pct'] for d in composition_data]

    x_pos = np.arange(len(ffa_types))
    bar_width = 0.25

    ax2.bar(x_pos - bar_width, single_pcts, bar_width, label=f'Singles (n={n_singles})', color='#7191A9', alpha=0.8)
    ax2.bar(x_pos, double_pcts, bar_width, label=f'Doubles (n={n_doubles})', color='#34699D', alpha=0.8)
    ax2.bar(x_pos + bar_width, triple_pcts, bar_width, label=f'Triples (n={n_triples})', color='#CC8250', alpha=0.8)

    # The expected percentage for each type in top performers should equal top_percentage
    # if selection is random (e.g., 20% of each type would be in top 20%)
    ax2.axhline(y=top_percentage, color='gray', linestyle=':', alpha=0.5, label=f'Expected if random ({top_percentage:.0f}%)')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax2.set_ylabel(f'Percent of Type in Top {top_percentage:.0f} Percent', fontsize=12)
    ax2.set_title(f'Normalized Enrichment in Top {top_percentage}% (n={n_top_strains})',
                 fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Composition Analysis of Top {top_percentage}% Performers (n={n_top_strains} strains) - ADDITIVE MODEL',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = f"additive_top_performer_composition_p{percentile}.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Top {top_percentage}% performer composition plot saved to:")
    print(f"  {filepath}")

    # Print summary statistics
    print(f"\nSummary across all FFAs (Top {top_percentage}%, n={n_top_strains}):")
    total_singles = sum(singles)
    total_doubles = sum(doubles)
    total_triples = sum(triples)
    total_top = len(ffa_types) * n_top_strains
    print(f"  Total singles: {total_singles}/{total_top} ({total_singles/total_top*100:.1f}%)")
    print(f"  Total doubles: {total_doubles}/{total_top} ({total_doubles/total_top*100:.1f}%)")
    print(f"  Total triples: {total_triples}/{total_top} ({total_triples/total_top*100:.1f}%)")


def main():
    """Main function to analyze distribution and composition of mutant types."""

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

    print("\nCreating distribution comparison plots...")
    create_distribution_comparison(
        single_mutants, double_mutants, triple_mutants,
        columns, abbreviations
    )

    print("\nCreating triple vs best double plots...")
    create_triple_vs_best_double_plot(
        double_mutants, triple_mutants,
        columns, abbreviations,
        single_mutants
    )

    print("\nAnalyzing top 10% performer composition...")
    analyze_top_performer_composition(
        single_mutants, double_mutants, triple_mutants,
        columns, percentile=90  # Top 10%
    )

    print("\nAnalyzing top 20% performer composition...")
    analyze_top_performer_composition(
        single_mutants, double_mutants, triple_mutants,
        columns, percentile=80  # Top 20%
    )

    print(f"\nAll analysis complete! Check the file paths above to view the plots.")


if __name__ == "__main__":
    main()