# experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_cost_benefit
# [[experiments.008-xue-ffa.scripts.additive_best_titers_per_ffa_cost_benefit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/additive_best_titers_per_ffa_cost_benefit
# Test file: experiments/008-xue-ffa/scripts/test_additive_best_titers_per_ffa_cost_benefit.py

import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import warnings
from scipy import stats

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


def create_cost_benefit_analysis(
    single_mutants, double_mutants, triple_mutants,
    columns, abbreviations=None
):
    """Create cost-benefit analysis comparing knockouts vs fitness gain."""

    # Use all FFA types including Total Titer
    ffa_types = columns

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for ffa_idx, ffa in enumerate(ffa_types):
        ax = axes[ffa_idx]

        # Collect data for each mutation type
        data_points = []

        # Singles (cost = 1)
        for gene, fitness_values in single_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                data_points.append({
                    'cost': 1,
                    'fitness': fitness_values[ffa_idx],
                    'type': 'single'
                })

        # Doubles (cost = 2)
        for (gene1, gene2), fitness_values in double_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                data_points.append({
                    'cost': 2,
                    'fitness': fitness_values[ffa_idx],
                    'type': 'double'
                })

        # Triples (cost = 3)
        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                data_points.append({
                    'cost': 3,
                    'fitness': fitness_values[ffa_idx],
                    'type': 'triple'
                })

        # Create jittered scatter plot
        costs = [d['cost'] for d in data_points]
        fitness = [d['fitness'] for d in data_points]
        types = [d['type'] for d in data_points]

        # Add jitter for visualization
        jittered_costs = []
        for c in costs:
            jitter = np.random.uniform(-0.15, 0.15)
            jittered_costs.append(c + jitter)

        # Color by type (using torchcell.mplstyle palette)
        colors = []
        for t in types:
            if t == 'single':
                colors.append('#7191A9')  # Light blue
            elif t == 'double':
                colors.append('#34699D')  # Deep blue
            else:
                colors.append('#CC8250')  # Burnt orange

        ax.scatter(jittered_costs, fitness, c=colors, alpha=0.6, s=30)

        # Add box plots on top
        singles_fitness = [d['fitness'] for d in data_points if d['type'] == 'single']
        doubles_fitness = [d['fitness'] for d in data_points if d['type'] == 'double']
        triples_fitness = [d['fitness'] for d in data_points if d['type'] == 'triple']

        bp = ax.boxplot([singles_fitness, doubles_fitness, triples_fitness],
                        positions=[1, 2, 3], widths=0.3, patch_artist=False,
                        boxprops=dict(linewidth=2, color='black'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='#B73C39'),  # Red from palette
                        meanprops=dict(marker='D', markerfacecolor='#775A9F',  # Purple from palette
                                     markeredgecolor='black', markersize=8, markeredgewidth=1.5),
                        showmeans=True)

        # Add reference line
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)

        # Calculate marginal benefits
        mean_single = np.mean(singles_fitness)
        mean_double = np.mean(doubles_fitness)
        mean_triple = np.mean(triples_fitness)

        marginal_1_to_2 = mean_double - mean_single
        marginal_2_to_3 = mean_triple - mean_double

        # Labels
        ax.set_xlabel('Number of Knockouts', fontsize=12)
        ax.set_ylabel('Fitness (relative to +ve Ctrl)', fontsize=12)
        ax.set_title(f'{ffa}', fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['1\n(Singles)', '2\n(Doubles)', '3\n(Triples)'])
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics text
        stats_text = (
            f"Mean fitness: {mean_single:.3f}, {mean_double:.3f}, {mean_triple:.3f}\n"
            f"Marginal gain 1→2: {marginal_1_to_2:+.3f}\n"
            f"Marginal gain 2→3: {marginal_2_to_3:+.3f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # All subplots are now used (6 total)

    plt.suptitle('Cost-Benefit Analysis: Number of Knockouts vs Fitness - ADDITIVE MODEL',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = "additive_cost_benefit_analysis.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Cost-benefit analysis plot saved to:")
    print(f"  {filepath}")


def create_marginal_benefit_summary(
    single_mutants, double_mutants, triple_mutants,
    columns
):
    """Create summary plot of marginal benefits across FFAs."""

    # Use all FFA types including Total Titer
    ffa_types = columns

    marginal_data = []

    for ffa_idx, ffa in enumerate(ffa_types):
        # Collect fitness values
        singles = []
        doubles = []
        triples = []

        for gene, fitness_values in single_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                singles.append(fitness_values[ffa_idx])

        for (gene1, gene2), fitness_values in double_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                doubles.append(fitness_values[ffa_idx])

        for (gene1, gene2, gene3), fitness_values in triple_mutants.items():
            if ffa_idx < len(fitness_values) and not np.isnan(fitness_values[ffa_idx]):
                triples.append(fitness_values[ffa_idx])

        # Calculate means and marginal benefits
        mean_single = np.mean(singles)
        mean_double = np.mean(doubles)
        mean_triple = np.mean(triples)

        max_single = np.max(singles)
        max_double = np.max(doubles)
        max_triple = np.max(triples)

        marginal_data.append({
            'ffa': ffa,
            'mean_single': mean_single,
            'mean_double': mean_double,
            'mean_triple': mean_triple,
            'max_single': max_single,
            'max_double': max_double,
            'max_triple': max_triple,
            'marginal_mean_1_to_2': mean_double - mean_single,
            'marginal_mean_2_to_3': mean_triple - mean_double,
            'marginal_max_1_to_2': max_double - max_single,
            'marginal_max_2_to_3': max_triple - max_double
        })

    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean fitness by number of knockouts
    ax = axes[0, 0]
    x = np.arange(len(ffa_types))
    width = 0.25

    ax.bar(x - width, [d['mean_single'] for d in marginal_data], width,
          label='Singles', color='#7191A9', alpha=0.8)
    ax.bar(x, [d['mean_double'] for d in marginal_data], width,
          label='Doubles', color='#34699D', alpha=0.8)
    ax.bar(x + width, [d['mean_triple'] for d in marginal_data], width,
          label='Triples', color='#CC8250', alpha=0.8)

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_ylabel('Mean Fitness', fontsize=12)
    ax.set_title('Mean Fitness by Mutation Type', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Max fitness by number of knockouts
    ax = axes[0, 1]
    ax.bar(x - width, [d['max_single'] for d in marginal_data], width,
          label='Singles', color='#7191A9', alpha=0.8)
    ax.bar(x, [d['max_double'] for d in marginal_data], width,
          label='Doubles', color='#34699D', alpha=0.8)
    ax.bar(x + width, [d['max_triple'] for d in marginal_data], width,
          label='Triples', color='#CC8250', alpha=0.8)

    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_ylabel('Max Fitness', fontsize=12)
    ax.set_title('Maximum Fitness by Mutation Type', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Marginal benefits (mean)
    ax = axes[1, 0]
    ax.bar(x - width/2, [d['marginal_mean_1_to_2'] for d in marginal_data], width,
          label='1→2 knockouts', color='#3D796E', alpha=0.8)  # Teal from palette
    ax.bar(x + width/2, [d['marginal_mean_2_to_3'] for d in marginal_data], width,
          label='2→3 knockouts', color='#E6A65D', alpha=0.8)  # Orange from palette

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_ylabel('Marginal Benefit', fontsize=12)
    ax.set_title('Marginal Mean Fitness Benefit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Marginal benefits (max)
    ax = axes[1, 1]
    ax.bar(x - width/2, [d['marginal_max_1_to_2'] for d in marginal_data], width,
          label='1→2 knockouts', color='#3D796E', alpha=0.8)  # Teal from palette
    ax.bar(x + width/2, [d['marginal_max_2_to_3'] for d in marginal_data], width,
          label='2→3 knockouts', color='#E6A65D', alpha=0.8)  # Orange from palette

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.set_ylabel('Marginal Benefit', fontsize=12)
    ax.set_title('Marginal Maximum Fitness Benefit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Marginal Benefit Analysis Across FFAs - ADDITIVE MODEL',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = "additive_marginal_benefit_summary.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    filepath = osp.join(ffa_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Marginal benefit summary saved to:")
    print(f"  {filepath}")

    # Print statistics
    print("\nMarginal Benefit Summary:")
    for d in marginal_data:
        print(f"\n{d['ffa']}:")
        print(f"  Mean marginal benefit 1→2: {d['marginal_mean_1_to_2']:+.3f}")
        print(f"  Mean marginal benefit 2→3: {d['marginal_mean_2_to_3']:+.3f}")
        print(f"  Max marginal benefit 1→2: {d['marginal_max_1_to_2']:+.3f}")
        print(f"  Max marginal benefit 2→3: {d['marginal_max_2_to_3']:+.3f}")

    # Overall statistics
    avg_marginal_1_2 = np.mean([d['marginal_mean_1_to_2'] for d in marginal_data])
    avg_marginal_2_3 = np.mean([d['marginal_mean_2_to_3'] for d in marginal_data])
    print(f"\nOverall average marginal benefits:")
    print(f"  1→2 knockouts: {avg_marginal_1_2:+.3f}")
    print(f"  2→3 knockouts: {avg_marginal_2_3:+.3f}")
    print(f"  Diminishing returns: {'Yes' if avg_marginal_2_3 < avg_marginal_1_2 else 'No'}")


def main():
    """Main function for cost-benefit analysis."""

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

    print("\nCreating cost-benefit analysis plots...")
    create_cost_benefit_analysis(
        single_mutants, double_mutants, triple_mutants,
        columns, abbreviations
    )

    print("\nCreating marginal benefit summary...")
    create_marginal_benefit_summary(
        single_mutants, double_mutants, triple_mutants,
        columns
    )

    print(f"\nAll analysis complete! Check the file paths above to view the plots.")


if __name__ == "__main__":
    main()