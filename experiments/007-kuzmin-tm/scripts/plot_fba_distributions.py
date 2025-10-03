"""
Generate distribution plots for FBA predictions and genetic interactions.
Shows distributions for single, double, and triple gene knockouts.
"""

import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from scipy import stats
from dotenv import load_dotenv
from datetime import datetime

# Use the torchcell style
mplstyle.use('/home/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle')

def plot_fitness_distributions(single_df, double_df, triple_df, matched_df, results_dir):
    """Plot fitness distributions for single, double, and triple knockouts."""
    
    # Create figure with 3 subplots for fitness
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get experimental values from matched data
    matched_fitness = matched_df[matched_df['phenotype_type'] == 'fitness']
    
    # Single knockouts
    ax = axes[0]
    if 'single' in matched_fitness['perturbation_type'].values:
        singles_matched = matched_fitness[matched_fitness['perturbation_type'] == 'single']
        ax.hist(singles_matched['fba_predicted'].values, bins=50, alpha=0.6, label='FBA Predicted', color='#2971A0')
        ax.hist(singles_matched['experimental'].values, bins=50, alpha=0.6, label='Experimental', color='#E74C3C')
    else:
        # No matched singles, just show FBA predictions if available
        if len(single_df) > 0:
            ax.hist(single_df['fitness'].values, bins=50, alpha=0.6, label='FBA Predicted', color='#2971A0')
    
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Count')
    ax.set_title(f'Single Gene KO (n={len(single_df):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Double knockouts
    ax = axes[1]
    if 'double' in matched_fitness['perturbation_type'].values:
        doubles_matched = matched_fitness[matched_fitness['perturbation_type'] == 'double']
        ax.hist(doubles_matched['fba_predicted'].values, bins=50, alpha=0.6, label='FBA Predicted', color='#2971A0')
        ax.hist(doubles_matched['experimental'].values, bins=50, alpha=0.6, label='Experimental', color='#E74C3C')
    else:
        # No matched doubles, just show FBA predictions if available
        if len(double_df) > 0:
            ax.hist(double_df['fitness'].values, bins=50, alpha=0.6, label='FBA Predicted', color='#2971A0')
    
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Count')
    ax.set_title(f'Double Gene KO (n={len(double_df):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Triple knockouts
    ax = axes[2]
    triples_matched = matched_fitness[matched_fitness['perturbation_type'] == 'triple']
    if len(triples_matched) > 0:
        # Remove NaN values for plotting
        fba_vals = triples_matched['fba_predicted'].dropna().values
        exp_vals = triples_matched['experimental'].dropna().values
        
        ax.hist(fba_vals, bins=50, alpha=0.6, label=f'FBA Predicted (n={len(fba_vals):,})', color='#2971A0')
        ax.hist(exp_vals, bins=50, alpha=0.6, label=f'Experimental (n={len(exp_vals):,})', color='#E74C3C')
        
        # Add text about missing values
        n_missing = len(triple_df) - len(triples_matched)
        if n_missing > 0:
            ax.text(0.02, 0.98, f'Unmatched: {n_missing:,}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Fitness')
    ax.set_ylabel('Count')
    ax.set_title(f'Triple Gene KO (n={len(triple_df):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Fitness Distributions by Perturbation Type', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = osp.join(results_dir, f"fitness_distributions_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved fitness distributions to {output_file}")
    
    return fig


def plot_interaction_distributions(digenic_df, trigenic_df, matched_df, results_dir):
    """Plot interaction distributions for digenic and trigenic interactions."""
    
    # Create figure with 2 subplots for interactions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get experimental values from matched data
    matched_interactions = matched_df[matched_df['phenotype_type'] == 'gene_interaction']
    
    # Digenic interactions
    ax = axes[0]
    if 'double' in matched_interactions['perturbation_type'].values:
        digenic_matched = matched_interactions[matched_interactions['perturbation_type'] == 'double']
        
        # Remove NaN values
        fba_vals = digenic_matched['fba_predicted'].dropna().values
        exp_vals = digenic_matched['experimental'].dropna().values
        
        ax.hist(fba_vals, bins=50, alpha=0.6, label=f'FBA Predicted (ε)', color='#2971A0')
        ax.hist(exp_vals, bins=50, alpha=0.6, label=f'Experimental', color='#E74C3C')
    else:
        # No matched digenic, just show FBA predictions
        if len(digenic_df) > 0:
            ax.hist(digenic_df['epsilon'].values, bins=50, alpha=0.6, 
                   label='FBA Predicted (ε)', color='#2971A0')
    
    ax.set_xlabel('Digenic Interaction (ε)')
    ax.set_ylabel('Count')
    ax.set_title(f'Digenic Interactions (n={len(digenic_df):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Trigenic interactions
    ax = axes[1]
    trigenic_matched = matched_interactions[matched_interactions['perturbation_type'] == 'triple']
    
    if len(trigenic_matched) > 0:
        # Remove NaN values
        fba_vals = trigenic_matched['fba_predicted'].dropna().values
        exp_vals = trigenic_matched['experimental'].dropna().values
        
        # Clip extreme values for better visualization
        fba_vals_clipped = np.clip(fba_vals, -0.5, 0.5)
        exp_vals_clipped = np.clip(exp_vals, -0.5, 0.5)
        
        ax.hist(fba_vals_clipped, bins=50, alpha=0.6, 
               label=f'FBA Predicted (τ, n={len(fba_vals):,})', color='#2971A0')
        ax.hist(exp_vals_clipped, bins=50, alpha=0.6, 
               label=f'Experimental (n={len(exp_vals):,})', color='#E74C3C')
        
        # Note about clipping if values were clipped
        if (fba_vals != fba_vals_clipped).any() or (exp_vals != exp_vals_clipped).any():
            ax.text(0.02, 0.98, 'Values clipped to [-0.5, 0.5]', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Trigenic Interaction (τ)')
    ax.set_ylabel('Count')
    ax.set_title(f'Trigenic Interactions (n={len(trigenic_df):,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    plt.suptitle('Genetic Interaction Distributions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = osp.join(results_dir, f"interaction_distributions_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved interaction distributions to {output_file}")
    
    return fig


def analyze_missing_data(triple_df, matched_df, results_dir):
    """Analyze why some fitness values are missing."""
    
    print("\n=== Analyzing Missing Fitness Data ===")
    
    # Get matched triples
    matched_fitness = matched_df[
        (matched_df['phenotype_type'] == 'fitness') & 
        (matched_df['perturbation_type'] == 'triple')
    ]
    
    matched_interactions = matched_df[
        (matched_df['phenotype_type'] == 'gene_interaction') & 
        (matched_df['perturbation_type'] == 'triple')
    ]
    
    print(f"Total triples in FBA: {len(triple_df):,}")
    print(f"Matched fitness values: {len(matched_fitness):,}")
    print(f"Matched interaction values: {len(matched_interactions):,}")
    print(f"Missing fitness matches: {len(triple_df) - len(matched_fitness):,}")
    
    # Check for NaN values
    print(f"\nNaN values in matched data:")
    print(f"  Fitness - FBA predicted NaN: {matched_fitness['fba_predicted'].isna().sum():,}")
    print(f"  Fitness - Experimental NaN: {matched_fitness['experimental'].isna().sum():,}")
    print(f"  Interactions - FBA predicted NaN: {matched_interactions['fba_predicted'].isna().sum():,}")
    print(f"  Interactions - Experimental NaN: {matched_interactions['experimental'].isna().sum():,}")
    
    # After removing NaN
    fitness_clean = len(matched_fitness['fba_predicted'].dropna())
    interaction_clean = len(matched_interactions['fba_predicted'].dropna())
    
    print(f"\nAfter removing NaN:")
    print(f"  Valid fitness values: {fitness_clean:,}")
    print(f"  Valid interaction values: {interaction_clean:,}")
    
    return {
        'total_triples': len(triple_df),
        'matched_fitness': len(matched_fitness),
        'matched_interactions': len(matched_interactions),
        'valid_fitness': fitness_clean,
        'valid_interactions': interaction_clean
    }


def main():
    """Main function to generate distribution plots."""
    load_dotenv()
    
    # Set up paths
    results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    
    print("Loading FBA results...")
    
    # Load FBA results
    single_df = pd.read_parquet(osp.join(results_dir, "singles_deletions.parquet"))
    double_df = pd.read_parquet(osp.join(results_dir, "doubles_deletions.parquet"))
    triple_df = pd.read_parquet(osp.join(results_dir, "triples_deletions.parquet"))
    digenic_df = pd.read_parquet(osp.join(results_dir, "digenic_interactions.parquet"))
    trigenic_df = pd.read_parquet(osp.join(results_dir, "trigenic_interactions.parquet"))
    
    # Load matched data - try fixed version first
    matched_file = osp.join(results_dir, "matched_fba_experimental_fixed.parquet")
    if not osp.exists(matched_file):
        matched_file = osp.join(results_dir, "matched_fba_experimental.parquet")
    
    matched_df = pd.read_parquet(matched_file)
    
    print(f"\nLoaded data:")
    print(f"  Singles: {len(single_df):,}")
    print(f"  Doubles: {len(double_df):,}")
    print(f"  Triples: {len(triple_df):,}")
    print(f"  Digenic interactions: {len(digenic_df):,}")
    print(f"  Trigenic interactions: {len(trigenic_df):,}")
    print(f"  Matched records: {len(matched_df):,}")
    
    # Analyze missing data
    missing_stats = analyze_missing_data(triple_df, matched_df, results_dir)
    
    # Generate plots
    print("\nGenerating distribution plots...")
    
    # Plot fitness distributions
    fig1 = plot_fitness_distributions(single_df, double_df, triple_df, matched_df, results_dir)
    
    # Plot interaction distributions
    fig2 = plot_interaction_distributions(digenic_df, trigenic_df, matched_df, results_dir)
    
    plt.show()
    
    print("\n=== Summary ===")
    print(f"The difference between fitness ({missing_stats['valid_fitness']:,}) and ")
    print(f"interactions ({missing_stats['valid_interactions']:,}) is due to:")
    print(f"1. Missing matches during merge: {missing_stats['total_triples'] - missing_stats['matched_fitness']:,}")
    print(f"2. NaN values after matching: {missing_stats['matched_fitness'] - missing_stats['valid_fitness']:,}")
    print("\nInteraction values can still be computed even when some fitness matches fail,")
    print("as long as all required components (singles, doubles, triple) exist in FBA results.")


if __name__ == "__main__":
    main()