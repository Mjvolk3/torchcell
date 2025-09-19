"""
Generate comparison plots for FBA predictions vs experimental data.
Creates two plots: fitness comparison and interaction comparison.
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

def plot_correlations(x_pred, y_exp, title, xlabel="FBA Predicted", ylabel="Experimental"):
    """
    Plot correlation between predicted and experimental values.
    x_pred: predicted values (x-axis)
    y_exp: experimental values (y-axis)
    """
    # Remove any NaN values
    mask = ~(np.isnan(x_pred) | np.isnan(y_exp))
    x_pred = x_pred[mask]
    y_exp = y_exp[mask]
    
    if len(x_pred) < 2:
        print(f"Not enough valid points for {title}")
        return None
    
    # Calculate correlation
    pearson_r, pearson_p = stats.pearsonr(x_pred, y_exp)
    spearman_r, spearman_p = stats.spearmanr(x_pred, y_exp)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Scatter plot
    ax.scatter(x_pred, y_exp, alpha=0.6, color='#2971A0', s=20)
    
    # Add diagonal line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add correlation text
    text_str = f'Pearson = {pearson_r:.3f}\nn = {len(x_pred):,}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8))
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to generate FBA comparison plots."""
    load_dotenv()
    
    # Set up paths
    results_dir = "/home/michaelvolk/Documents/projects/torchcell/experiments/007-kuzmin-tm/results/cobra-fba-growth"
    
    # Check if matched data exists - try fixed version first
    matched_file_fixed = osp.join(results_dir, "matched_fba_experimental_fixed.parquet")
    matched_file = osp.join(results_dir, "matched_fba_experimental.parquet")
    
    if osp.exists(matched_file_fixed):
        matched_file = matched_file_fixed
        print("Using fixed matched data...")
    elif osp.exists(matched_file):
        print("Using regular matched data...")
    else:
        print("Matched data not found. Please run match_fba_to_experiments_fixed.py first.")
        return
    
    # Load matched data
    print("Loading matched FBA-experimental data...")
    matched_df = pd.read_parquet(matched_file)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Fitness comparison (all perturbation types)
    fitness_df = matched_df[matched_df['phenotype_type'] == 'fitness']
    
    if len(fitness_df) > 0:
        x_pred = fitness_df['fba_predicted'].values
        y_exp = fitness_df['experimental'].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_pred) | np.isnan(y_exp))
        x_pred = x_pred[mask]
        y_exp = y_exp[mask]
        
        # Calculate correlation
        pearson_r, _ = stats.pearsonr(x_pred, y_exp)
        
        # Scatter plot
        ax1.scatter(x_pred, y_exp, alpha=0.6, color='#2971A0', s=20)
        
        # Add diagonal line
        min_val = min(ax1.get_xlim()[0], ax1.get_ylim()[0])
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Labels and title
        ax1.set_xlabel('FBA Predicted', fontsize=12)
        ax1.set_ylabel('Experimental', fontsize=12)
        ax1.set_title('Fitness', fontsize=14, fontweight='bold')
        
        # Add correlation text
        text_str = f'Pearson = {pearson_r:.4f}\nn = {len(x_pred):,}'
        ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
    else:
        ax1.text(0.5, 0.5, 'No fitness data', transform=ax1.transAxes,
                ha='center', va='center', fontsize=14)
        ax1.set_title('Fitness', fontsize=14, fontweight='bold')
    
    # Plot 2: Gene interactions (trigenic tau)
    interaction_df = matched_df[matched_df['phenotype_type'] == 'gene_interaction']
    
    if len(interaction_df) > 0:
        x_pred = interaction_df['fba_predicted'].values
        y_exp = interaction_df['experimental'].values
        
        # Remove NaN values
        mask = ~(np.isnan(x_pred) | np.isnan(y_exp))
        x_pred = x_pred[mask]
        y_exp = y_exp[mask]
        
        # Calculate correlation
        pearson_r, _ = stats.pearsonr(x_pred, y_exp)
        
        # Scatter plot
        ax2.scatter(x_pred, y_exp, alpha=0.6, color='#2971A0', s=20)
        
        # Add diagonal line
        min_val = min(ax2.get_xlim()[0], ax2.get_ylim()[0])
        max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # Labels and title
        ax2.set_xlabel('FBA Predicted', fontsize=12)
        ax2.set_ylabel('Experimental', fontsize=12)
        ax2.set_title('Interactions', fontsize=14, fontweight='bold')
        
        # Add correlation text
        text_str = f'Pearson = {pearson_r:.4f}\nn = {len(x_pred):,}'
        ax2.text(0.05, 0.95, text_str, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
    else:
        ax2.text(0.5, 0.5, 'No interaction data', transform=ax2.transAxes,
                ha='center', va='center', fontsize=14)
        ax2.set_title('Interactions', fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle('FBA Predictions vs Experimental Data', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = osp.join(results_dir, f"fba_comparison_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    
    # Also save without timestamp for easy reference
    output_file_latest = osp.join(results_dir, "fba_comparison_latest.png")
    plt.savefig(output_file_latest, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file_latest}")
    
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    if len(fitness_df) > 0:
        print(f"Fitness: {len(fitness_df)} data points")
        fitness_corr = fitness_df[['fba_predicted', 'experimental']].corr().iloc[0, 1]
        print(f"  Pearson correlation: {fitness_corr:.4f}")
    
    if len(interaction_df) > 0:
        print(f"Interactions: {len(interaction_df)} data points")
        interaction_corr = interaction_df[['fba_predicted', 'experimental']].corr().iloc[0, 1]
        print(f"  Pearson correlation: {interaction_corr:.4f}")

if __name__ == "__main__":
    main()