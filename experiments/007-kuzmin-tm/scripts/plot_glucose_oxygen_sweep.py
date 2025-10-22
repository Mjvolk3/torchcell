#!/usr/bin/env python3
"""
Plotting script for glucose/oxygen sensitivity analysis.
Reads all FBA results and generates comparison plots.
"""

import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from dotenv import load_dotenv
from torchcell.timestamp import timestamp


def load_all_results(results_dir):
    """Load all matched results from glucose/oxygen sweep."""
    print("=== Loading Results ===")

    # Find all matched result files
    pattern = osp.join(results_dir, 'matched_*_glc*_o2*.parquet')
    files = glob.glob(pattern)

    print(f"Found {len(files)} result files")

    if len(files) == 0:
        print("Warning: No result files found!")
        return pd.DataFrame()

    # Load and combine all results
    all_results = []
    for file in files:
        df = pd.read_parquet(file)
        # Rename columns for consistency with plotting code
        # Map FBA fitness
        if 'fba_fitness' in df.columns:
            df['fba_predicted'] = df['fba_fitness']

        # Map experimental fitness
        if 'experimental_fitness' in df.columns:
            df['experimental'] = df['experimental_fitness']
        elif 'fitness' in df.columns:
            df['experimental'] = df['fitness']

        # Map experimental interaction (already named correctly from postprocessing)
        if 'experimental_interaction' in df.columns:
            df['gene_interaction'] = df['experimental_interaction']

        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(combined_df)} total measurements")

    return combined_df


def create_comprehensive_correlation_heatmaps(all_matched_df):
    """Create comprehensive heatmaps: 2 rows (fitness, interactions) × 3 columns (media)."""

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Calculate correlations for each condition
    conditions = all_matched_df[['media', 'glucose', 'oxygen']].drop_duplicates()

    fitness_correlation_data = []
    interaction_correlation_data = []

    for _, cond in conditions.iterrows():
        subset = all_matched_df[
            (all_matched_df['media'] == cond['media']) &
            (all_matched_df['glucose'] == cond['glucose']) &
            (all_matched_df['oxygen'] == cond['oxygen'])
        ]

        # Fitness correlation
        if len(subset) > 1:
            mask = ~(np.isnan(subset['fba_predicted']) | np.isnan(subset['experimental']))
            if mask.sum() > 1:
                fitness_r, _ = stats.pearsonr(
                    subset.loc[mask, 'fba_predicted'],
                    subset.loc[mask, 'experimental']
                )
            else:
                fitness_r = np.nan
        else:
            fitness_r = np.nan

        fitness_correlation_data.append({
            'media': cond['media'],
            'glucose': cond['glucose'],
            'oxygen': cond['oxygen'],
            'pearson_r': fitness_r,
            'n': len(subset)
        })

        # Interaction correlation
        exp_col = 'experimental_interaction' if 'experimental_interaction' in subset.columns else 'gene_interaction'
        interaction_n = 0
        if 'fba_triple_interaction' in subset.columns and exp_col in subset.columns:
            interaction_mask = ~(subset['fba_triple_interaction'].isna() | subset[exp_col].isna())
            interaction_subset = subset[interaction_mask]
            interaction_n = len(interaction_subset)

            if len(interaction_subset) > 1:
                interaction_r, _ = stats.pearsonr(
                    interaction_subset['fba_triple_interaction'],
                    interaction_subset[exp_col]
                )
            else:
                interaction_r = np.nan
        else:
            interaction_r = np.nan

        interaction_correlation_data.append({
            'media': cond['media'],
            'glucose': cond['glucose'],
            'oxygen': cond['oxygen'],
            'pearson_r': interaction_r,
            'n': interaction_n
        })

    fitness_corr_df = pd.DataFrame(fitness_correlation_data)
    interaction_corr_df = pd.DataFrame(interaction_correlation_data)

    # Create 2×3 figure (2 rows: fitness/interactions, 3 columns: media types)
    media_types = ['minimal', 'YNB', 'YPD']
    media_labels = {'minimal': 'Minimal', 'YNB': 'YNB', 'YPD': 'YPD'}
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 0: Fitness correlations
    for col_idx, media in enumerate(media_types):
        ax = axes[0, col_idx]
        media_corr = fitness_corr_df[fitness_corr_df['media'] == media]

        if len(media_corr) > 0:
            pivot = media_corr.pivot(index='oxygen', columns='glucose', values='pearson_r')

            # Plot heatmap
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                          vmin=-0.3, vmax=0.3)

            # Set ticks
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticklabels(pivot.index)

            # Add correlation values
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}',
                               ha='center', va='center',
                               color='white' if abs(val) > 0.15 else 'black',
                               fontsize=9)

            ax.set_title(f'{media_labels[media]} - Fitness', fontweight='bold', fontsize=13)
            ax.set_xlabel('Glucose (mmol/gDW/h)', fontsize=11)
            if col_idx == 0:
                ax.set_ylabel('O₂ (mmol/gDW/h)', fontsize=11)

    # Row 1: Interaction correlations
    for col_idx, media in enumerate(media_types):
        ax = axes[1, col_idx]
        media_corr = interaction_corr_df[interaction_corr_df['media'] == media]

        if len(media_corr) > 0:
            pivot = media_corr.pivot(index='oxygen', columns='glucose', values='pearson_r')

            # Plot heatmap
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                          vmin=-0.3, vmax=0.3)

            # Set ticks
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_yticks(range(len(pivot.index)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticklabels(pivot.index)

            # Add correlation values
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}',
                               ha='center', va='center',
                               color='white' if abs(val) > 0.15 else 'black',
                               fontsize=9)

            ax.set_title(f'{media_labels[media]} - Interactions', fontweight='bold', fontsize=13)
            ax.set_xlabel('Glucose (mmol/gDW/h)', fontsize=11)
            if col_idx == 0:
                ax.set_ylabel('O₂ (mmol/gDW/h)', fontsize=11)

    # Add colorbar
    fig.colorbar(im, ax=axes, label='Pearson r', fraction=0.046, pad=0.04)

    fig.suptitle('FBA Correlation vs Glucose/O₂ Constraints', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()

    output_file = osp.join(ASSET_IMAGES_DIR, f'experiments.007.glucose_oxygen_correlations_{timestamp()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\nSaved comprehensive correlation heatmap to {output_file}')
    plt.close()

    return fitness_corr_df, interaction_corr_df


def create_fitness_distribution_plots(results_dir):
    """Create plots showing fitness distributions across conditions."""

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Load raw FBA results (not matched, to see all fitness values)
    pattern = osp.join(results_dir, 'singles_deletions_*_glc*_o2*.parquet')
    files = glob.glob(pattern)

    if len(files) == 0:
        print("Warning: No singles deletion files found for distribution plots")
        return

    # Select subset of conditions to plot
    all_results = []
    for file in files[:12]:  # Plot first 12 conditions
        df = pd.read_parquet(file)
        all_results.append(df)

    if len(all_results) == 0:
        return

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, df in enumerate(all_results):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ax.hist(df['fitness'].values, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.set_xlabel('Fitness', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)

        # Get condition from dataframe
        if 'media' in df.columns and len(df) > 0:
            media = df['media'].iloc[0]
            glucose = df['glucose'].iloc[0]
            oxygen = df['oxygen'].iloc[0]
            ax.set_title(f"{media}, glc={glucose}, O₂={oxygen}", fontsize=11, fontweight='bold')
        else:
            ax.set_title(f"Condition {idx+1}", fontsize=11)

        ax.grid(True, alpha=0.3)

        # Add statistics
        unique_vals = len(np.unique(np.round(df['fitness'], 4)))
        ax.text(0.98, 0.98, f'n={len(df)}\nunique={unique_vals}',
               transform=ax.transAxes, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
               fontsize=9)

    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Fitness Distributions Across Glucose/O₂ Conditions', fontweight='bold', fontsize=16)
    plt.tight_layout()

    output_file = osp.join(ASSET_IMAGES_DIR, f'experiments.007.glucose_oxygen_distributions_{timestamp()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved distribution plot to {output_file}')
    plt.close()


def create_media_comparison_plot(all_matched_df):
    """Create 3x2 grid plot comparing fitness and interactions across media types."""

    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Filter to default condition (glucose=10, oxygen=1000) for cleaner comparison
    default_df = all_matched_df[
        (all_matched_df['glucose'] == 10) &
        (all_matched_df['oxygen'] == 1000)
    ]

    if len(default_df) == 0:
        print("Warning: No data for default condition (glucose=10, oxygen=1000)")
        return

    # Create figure with 3 rows (media types) and 2 columns (fitness, interactions)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    media_types = ['minimal', 'YNB', 'YPD']
    media_labels = {'minimal': 'Minimal Media', 'YNB': 'YNB Media', 'YPD': 'YPD Media'}

    for idx, media in enumerate(media_types):
        media_data = default_df[default_df['media'] == media]

        # Left column: Fitness comparison (all matched data)
        ax_fitness = axes[idx, 0]

        # Plot ALL matched data for fitness (no filtering)
        if len(media_data) > 0:
            # Filter out NaN values
            mask = ~(np.isnan(media_data['fba_predicted']) | np.isnan(media_data['experimental']))
            x = media_data.loc[mask, 'fba_predicted'].values
            y = media_data.loc[mask, 'experimental'].values

            if len(x) > 1:
                pearson_r, _ = stats.pearsonr(x, y)

                # Scatter plot
                ax_fitness.scatter(x, y, alpha=0.5, s=20, color='steelblue', edgecolors='none')

                # Diagonal line
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax_fitness.plot([min_val, max_val], [min_val, max_val],
                               'r--', alpha=0.7, linewidth=2)

                # Labels
                ax_fitness.set_xlabel('FBA Predicted', fontsize=12)
                ax_fitness.set_ylabel('Experimental', fontsize=12)
                ax_fitness.set_title(f'{media_labels[media]} - Fitness',
                                    fontsize=13, fontweight='bold')

                # Correlation text
                ax_fitness.text(0.05, 0.95, f'Pearson r = {pearson_r:.4f}\nn = {len(x):,}',
                               transform=ax_fitness.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                               fontsize=11)

                ax_fitness.grid(True, alpha=0.3)

        # Right column: Genetic interactions
        ax_interactions = axes[idx, 1]

        # Use fba_triple_interaction column (calculated in post-processing)
        if len(media_data) > 0 and 'fba_triple_interaction' in media_data.columns:
            # Map column names (handle both conventions)
            exp_col = 'experimental_interaction' if 'experimental_interaction' in media_data.columns else 'gene_interaction'

            # Filter for valid interaction data (not NaN)
            interaction_mask = ~(media_data['fba_triple_interaction'].isna() | media_data[exp_col].isna())
            interaction_data = media_data[interaction_mask]

            if len(interaction_data) > 1:
                x = interaction_data['fba_triple_interaction'].values
                y = interaction_data[exp_col].values

                # Calculate correlation
                pearson_r, _ = stats.pearsonr(x, y)

                # Scatter plot
                ax_interactions.scatter(x, y, alpha=0.5, s=20, color='steelblue', edgecolors='none')

                # Diagonal line
                max_abs = max(abs(x).max(), abs(y).max()) if len(x) > 0 else 1
                ax_interactions.plot([-max_abs, max_abs], [-max_abs, max_abs],
                                    'r--', alpha=0.7, linewidth=2)

                # Labels
                ax_interactions.set_xlabel('FBA Predicted', fontsize=12)
                ax_interactions.set_ylabel('Experimental', fontsize=12)
                ax_interactions.set_title(f'{media_labels[media]} - Interactions',
                                         fontsize=13, fontweight='bold')

                # Correlation text
                ax_interactions.text(0.05, 0.95, f'Pearson r = {pearson_r:.4f}\nn = {len(x):,}',
                                    transform=ax_interactions.transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                                    fontsize=11)

                ax_interactions.grid(True, alpha=0.3)

    fig.suptitle('FBA Predictions vs Experimental Data by Media (glucose=10, O₂=1000)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = osp.join(ASSET_IMAGES_DIR,
                          f'experiments.007.fba_vs_experimental_by_media_default_{timestamp()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Saved media comparison plot to {output_file}')
    plt.close()


def print_summary_stats(fitness_corr_df, interaction_corr_df):
    """Print summary statistics from correlation analysis."""

    print("\n=== Summary Statistics ===")

    print("\n" + "="*70)
    print("FITNESS CORRELATIONS")
    print("="*70)
    print("\nCorrelation ranges by media:")
    print("-"*60)

    for media in ['minimal', 'YNB', 'YPD']:
        media_corr = fitness_corr_df[fitness_corr_df['media'] == media]['pearson_r']
        media_corr = media_corr[~np.isnan(media_corr)]

        if len(media_corr) > 0:
            print(f"{media:8s}: min={media_corr.min():6.4f}, "
                  f"max={media_corr.max():6.4f}, "
                  f"mean={media_corr.mean():6.4f}")

    print("\nBest fitness correlations:")
    print("-"*60)
    top_corr = fitness_corr_df.nlargest(5, 'pearson_r')
    for _, row in top_corr.iterrows():
        print(f"{row['media']:8s}, glucose={row['glucose']:4.0f}, "
              f"O2={row['oxygen']:6.0f}: r={row['pearson_r']:.4f}, n={row['n']}")

    print("\n" + "="*70)
    print("INTERACTION CORRELATIONS")
    print("="*70)
    print("\nCorrelation ranges by media:")
    print("-"*60)

    for media in ['minimal', 'YNB', 'YPD']:
        media_corr = interaction_corr_df[interaction_corr_df['media'] == media]['pearson_r']
        media_corr = media_corr[~np.isnan(media_corr)]

        if len(media_corr) > 0:
            print(f"{media:8s}: min={media_corr.min():6.4f}, "
                  f"max={media_corr.max():6.4f}, "
                  f"mean={media_corr.mean():6.4f}")

    print("\nBest interaction correlations:")
    print("-"*60)
    top_corr = interaction_corr_df.nlargest(5, 'pearson_r')
    for _, row in top_corr.iterrows():
        print(f"{row['media']:8s}, glucose={row['glucose']:4.0f}, "
              f"O2={row['oxygen']:6.0f}: r={row['pearson_r']:.4f}, n={row['n']}")


def main():
    """Main execution."""

    BASE_DIR = "experiments/007-kuzmin-tm"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"

    print("="*70)
    print("Glucose/O2 Sensitivity Analysis - Plotting")
    print("="*70)
    print()

    # Load all results
    all_matched_df = load_all_results(RESULTS_DIR)

    if len(all_matched_df) == 0:
        print("Error: No results to plot!")
        return

    # Generate comprehensive correlation heatmaps
    print("\n=== Generating Comprehensive Correlation Heatmaps ===")
    fitness_corr_df, interaction_corr_df = create_comprehensive_correlation_heatmaps(all_matched_df)

    # Print statistics
    print_summary_stats(fitness_corr_df, interaction_corr_df)

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE:")
    print("="*70)
    print()
    print("IF correlations improve with richer media or higher glucose/O2:")
    print("  → Issue is CONSTRAINT-DRIVEN (media composition matters)")
    print("  → Recommendation: Use nutrient-rich conditions (YPD, high glucose)")
    print()
    print("IF correlations remain poor across all conditions:")
    print("  → Issue is MODEL-INTRINSIC")
    print("  → Check: GPR logic, biomass composition, reaction bottlenecks")
    print("="*70)
    print("\nPlotting complete!")
    print("="*70)


if __name__ == "__main__":
    main()
