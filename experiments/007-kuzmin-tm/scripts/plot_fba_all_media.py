#!/usr/bin/env python3
"""
Lightweight plotting script for FBA media comparison.
Reads matched parquet files and generates plots only.
NO heavy computation - assumes post-processing is already done.
"""

import os
import os.path as osp
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add torchcell to path
sys.path.append('/home/michaelvolk/Documents/projects/torchcell')
from dotenv import load_dotenv
from torchcell.timestamp import timestamp


def generate_plots(results_dir):
    """Generate comparison plots for all media."""
    print("\n=== Generating Comparison Plots ===")

    # Load environment for ASSET_IMAGES_DIR
    load_dotenv()
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    media_conditions = ['minimal', 'YNB', 'YPD']
    media_labels = {'minimal': 'Minimal Media', 'YNB': 'YNB Media', 'YPD': 'YPD Media'}

    # Create figure with 3 rows (one per media) and 2 columns (fitness and interactions)
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('FBA Predictions vs Experimental Data by Media Condition', fontsize=16, y=0.995)

    for idx, media in enumerate(media_conditions):
        # Load matched data
        matched_file = f'{results_dir}/matched_fba_experimental_{media}.parquet'
        if not osp.exists(matched_file):
            print(f'Warning: No matched file for {media}')
            continue

        matched = pd.read_parquet(matched_file)
        print(f'{media}: loaded {len(matched)} matched records')

        # Filter for valid data
        matched = matched.dropna(subset=['fitness', 'fba_fitness'])

        print(f'  Matched records: {len(matched)}')

        # Plot 1: Fitness comparison (all matched data)
        ax1 = axes[idx, 0]
        if len(matched) > 0:
            x = matched['fba_fitness'].values
            y = matched['fitness'].values

            # Calculate correlation
            pearson_r = stats.pearsonr(x, y)[0] if len(x) > 1 else np.nan

            # Scatter plot
            ax1.scatter(x, y, alpha=0.5, s=20, color='steelblue', edgecolors='none')

            # Add diagonal reference line
            max_val = max(x.max(), y.max())
            min_val = min(x.min(), y.min())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2)

            # Labels and title
            ax1.set_xlabel('FBA Predicted', fontsize=12)
            ax1.set_ylabel('Experimental', fontsize=12)
            ax1.set_title(f'{media_labels[media]} - Fitness', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Add correlation text
            ax1.text(0.05, 0.95, f'Pearson r = {pearson_r:.4f}\nn = {len(matched):,}',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                    fontsize=11)

        # Plot 2: Genetic interactions
        # Use fba_triple_interaction column (calculated in post-processing)
        ax2 = axes[idx, 1]
        if len(matched) > 0 and 'fba_triple_interaction' in matched.columns and 'gene_interaction' in matched.columns:
            # Filter for valid interaction data (not NaN)
            interaction_mask = ~(matched['fba_triple_interaction'].isna() | matched['gene_interaction'].isna())
            interaction_data = matched[interaction_mask]

            print(f'  Found {len(interaction_data)} triple interactions')

            if len(interaction_data) > 0:
                x = interaction_data['fba_triple_interaction'].values
                y = interaction_data['gene_interaction'].values

                # Calculate correlation
                pearson_r = stats.pearsonr(x, y)[0] if len(x) > 1 else np.nan

                # Scatter plot
                ax2.scatter(x, y, alpha=0.5, s=20, color='steelblue', edgecolors='none')

                # Add diagonal reference line
                max_abs = max(abs(x).max(), abs(y).max()) if len(x) > 0 else 1
                ax2.plot([-max_abs, max_abs], [-max_abs, max_abs], 'r--', alpha=0.7, linewidth=2)

                # Labels and title
                ax2.set_xlabel('FBA Predicted', fontsize=12)
                ax2.set_ylabel('Experimental', fontsize=12)
                ax2.set_title(f'{media_labels[media]} - Interactions', fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3)

                # Add correlation text
                ax2.text(0.05, 0.95, f'Pearson r = {pearson_r:.4f}\nn = {len(x):,}',
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                        fontsize=11)

    # Adjust layout
    plt.tight_layout()

    # Save figure to ASSET_IMAGES_DIR
    output_file = osp.join(ASSET_IMAGES_DIR, f'experiments.007.fba_comparison_all_media_{timestamp()}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\nSaved comparison plot to {output_file}')
    plt.close()


def print_summary_stats(results_dir):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===")
    print("\nGrowth rate comparison across media:")

    print('Media     | Wild-type Growth | Singles Affected | Doubles Affected | Triples Affected')
    print('----------|------------------|------------------|------------------|------------------')

    for media in ['minimal', 'YNB', 'YPD']:
        # Load results
        singles = pd.read_parquet(f'{results_dir}/singles_deletions_{media}.parquet')
        doubles = pd.read_parquet(f'{results_dir}/doubles_deletions_{media}.parquet')
        triples = pd.read_parquet(f'{results_dir}/triples_deletions_{media}.parquet')

        # Get wild-type growth (should be same for all, but check)
        wt_growth = singles['fitness'].max() if len(singles) > 0 else 1.0

        # Count affected mutants (fitness < 0.99)
        singles_affected = (singles['fitness'] < 0.99).sum()
        doubles_affected = (doubles['fitness'] < 0.99).sum()
        triples_affected = (triples['fitness'] < 0.99).sum()

        print(f'{media:9s} | {wt_growth:16.4f} | {singles_affected:16d} | {doubles_affected:16d} | {triples_affected:16d}')

    print('\nFitness distribution (unique values):')
    for media in ['minimal', 'YNB', 'YPD']:
        singles = pd.read_parquet(f'{results_dir}/singles_deletions_{media}.parquet')
        unique_values = len(singles['fitness'].round(3).unique())
        print(f'  {media}: {unique_values} unique fitness values')


def main():
    """Main execution."""
    BASE_DIR = "experiments/007-kuzmin-tm"
    RESULTS_DIR = f"{BASE_DIR}/results/cobra-fba-growth"

    print("="*70)
    print("FBA Media Comparison - Plotting")
    print("="*70)
    print()

    # Generate plots
    generate_plots(RESULTS_DIR)

    # Print statistics
    print_summary_stats(RESULTS_DIR)

    print("\n" + "="*70)
    print("Plotting complete!")
    print("="*70)


if __name__ == "__main__":
    main()
