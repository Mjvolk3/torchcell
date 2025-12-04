#!/usr/bin/env python3

"""
Visualization for GLM with log link epistatic interaction analysis.
Creates plots showing epistatic fold (φ) with proper representation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from typing import Dict
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa")
RESULTS_DIR = BASE_DIR / "results" / "glm_log_link"
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)

# Colors
DIGENIC_COLOR = '#7191A9'  # Light blue
TRIGENIC_COLOR = '#B73C39'  # Red
POSITIVE_EPISTASIS_COLOR = '#6B8D3A'  # Green
NEGATIVE_EPISTASIS_COLOR = '#B73C39'  # Red
NEUTRAL_COLOR = '#7F7F7F'  # Gray

# FFA labels
FFA_LABELS = {
    'C140': 'C14:0',
    'C160': 'C16:0',
    'C180': 'C18:0',
    'C161': 'C16:1',
    'C181': 'C18:1',
    'Total Titer': 'Total'
}


def load_results(results_file: Path = RESULTS_DIR / 'glm_log_link_results.pkl') -> Dict:
    """Load saved GLM log link results."""
    with open(results_file, 'rb') as f:
        return pickle.load(f)


def plot_digenic_interactions(results: Dict):
    """
    Create bar plots for ALL digenic interactions.
    Bars start at φ=1 (no epistasis).
    """
    for trait in results:
        if not results[trait] or 'epistasis' not in results[trait]:
            continue

        # Get ALL digenic interactions (not just significant)
        digenic = []
        for k, v in results[trait]['epistasis'].items():
            if v['order'] == 'digenic':  # Include all digenic, regardless of p-value
                genes = k.replace('ko_', '').replace(':', '-')
                digenic.append({
                    'interaction': genes,
                    'phi': v['phi'],
                    'pvalue': v['pvalue'],
                    'gamma': v['gamma']
                })

        if not digenic:
            continue

        # Sort by phi value for visualization
        digenic.sort(key=lambda x: x['phi'])

        # Create figure with dynamic width based on number of interactions
        n_interactions = len(digenic)
        fig_width = max(14, n_interactions * 0.4)  # Scale width with number of interactions
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        # Prepare data
        interactions = [d['interaction'] for d in digenic]
        phi_values = [d['phi'] for d in digenic]
        pvalues = [d['pvalue'] for d in digenic]

        # Colors based on epistasis direction
        colors = []
        for phi in phi_values:
            if phi > 1.2:
                colors.append(POSITIVE_EPISTASIS_COLOR)
            elif phi < 0.8:
                colors.append(NEGATIVE_EPISTASIS_COLOR)
            else:
                colors.append(NEUTRAL_COLOR)

        # Create bars starting from 1
        x = np.arange(len(interactions))
        bars = ax.bar(x, [phi - 1 for phi in phi_values], bottom=1,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Set y-axis limits with more space for legend
        y_min = min(0.2, min(phi_values) - 0.2)
        y_max = max(2.5, max(phi_values) + 0.5)  # More space at top for legend
        ax.set_ylim(y_min, y_max)

        # Add reference line at φ=1
        ax.axhline(y=1, color='black', linestyle='-', linewidth=2,
                   alpha=0.8, label='No epistasis (φ=1)')

        # Add threshold lines
        ax.axhline(y=1.2, color=POSITIVE_EPISTASIS_COLOR, linestyle='--', alpha=0.3)
        ax.axhline(y=0.8, color=NEGATIVE_EPISTASIS_COLOR, linestyle='--', alpha=0.3)

        # Labels and formatting
        ax.set_xlabel('Gene pair', fontsize=12)
        ax.set_ylabel('φ (epistatic fold)', fontsize=12)
        ax.set_title(f'Digenic Interactions - {FFA_LABELS.get(trait, trait)} (GLM Log Link)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(interactions, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance markers only for significant interactions
        for i, (phi, pval) in enumerate(zip(phi_values, pvalues)):
            if pval < 0.001:
                pval_text = '***'
            elif pval < 0.01:
                pval_text = '**'
            elif pval < 0.05:
                pval_text = '*'
            else:
                continue  # Don't show anything for non-significant interactions

            y_pos = phi + 0.05 if phi > 1 else phi - 0.05
            ax.text(i, y_pos, pval_text,
                    ha='center', va='bottom' if phi > 1 else 'top',
                    fontsize=7)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=POSITIVE_EPISTASIS_COLOR, alpha=0.7,
                  label='Positive epistasis (φ > 1.2)'),
            Patch(facecolor=NEUTRAL_COLOR, alpha=0.7,
                  label='Neutral (0.8 ≤ φ ≤ 1.2)'),
            Patch(facecolor=NEGATIVE_EPISTASIS_COLOR, alpha=0.7,
                  label='Negative epistasis (φ < 0.8)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        # Add significance explanation
        ax.text(0.98, 0.98, '* p<0.05, ** p<0.01, *** p<0.001',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filename = f'glm_log_link_digenic_{trait}.png'
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        save_path = osp.join(ffa_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_trigenic_interactions(results: Dict):
    """
    Create bar plots for ALL trigenic interactions.
    """
    for trait in results:
        if not results[trait] or 'epistasis' not in results[trait]:
            continue

        # Get ALL trigenic interactions (not just significant)
        trigenic = []
        for k, v in results[trait]['epistasis'].items():
            if v['order'] == 'trigenic':  # Include all trigenic, regardless of p-value
                genes = k.replace('ko_', '').replace(':', '-')
                trigenic.append({
                    'interaction': genes,
                    'phi': v['phi'],
                    'pvalue': v['pvalue'],
                    'gamma': v['gamma']
                })

        if not trigenic:
            continue

        # Sort by phi for visualization
        trigenic.sort(key=lambda x: x['phi'])

        # Create figure with dynamic width
        n_interactions = len(trigenic)
        fig_width = max(14, n_interactions * 0.3)  # Scale width (slightly less than digenic due to more bars)
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        # Prepare data
        interactions = [d['interaction'] for d in trigenic]
        phi_values = [d['phi'] for d in trigenic]
        pvalues = [d['pvalue'] for d in trigenic]

        # Colors
        colors = []
        for phi in phi_values:
            if phi > 1.2:
                colors.append(POSITIVE_EPISTASIS_COLOR)
            elif phi < 0.8:
                colors.append(NEGATIVE_EPISTASIS_COLOR)
            else:
                colors.append(NEUTRAL_COLOR)

        # Create bars starting from 1
        x = np.arange(len(interactions))
        bars = ax.bar(x, [phi - 1 for phi in phi_values], bottom=1,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        # Set y-axis with more space
        y_min = min(0.2, min(phi_values) - 0.2)
        y_max = max(3.0, max(phi_values) + 0.5)  # More space at top
        ax.set_ylim(y_min, y_max)

        # Reference lines
        ax.axhline(y=1, color='black', linestyle='-', linewidth=2,
                   alpha=0.8, label='No epistasis (φ=1)')
        ax.axhline(y=1.2, color=POSITIVE_EPISTASIS_COLOR, linestyle='--', alpha=0.3)
        ax.axhline(y=0.8, color=NEGATIVE_EPISTASIS_COLOR, linestyle='--', alpha=0.3)

        # Labels
        ax.set_xlabel('Gene triple', fontsize=12)
        ax.set_ylabel('φ (epistatic fold)', fontsize=12)
        ax.set_title(f'Trigenic Interactions - {FFA_LABELS.get(trait, trait)} (GLM Log Link)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(interactions, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add significance markers only for significant interactions
        for i, (phi, pval) in enumerate(zip(phi_values, pvalues)):
            if pval < 0.001:
                pval_text = '***'
            elif pval < 0.01:
                pval_text = '**'
            elif pval < 0.05:
                pval_text = '*'
            else:
                continue  # Don't show anything for non-significant interactions

            y_pos = phi + 0.05 if phi > 1 else phi - 0.05
            ax.text(i, y_pos, pval_text,
                    ha='center', va='bottom' if phi > 1 else 'top',
                    fontsize=7)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=POSITIVE_EPISTASIS_COLOR, alpha=0.7,
                  label='Positive epistasis (φ > 1.2)'),
            Patch(facecolor=NEUTRAL_COLOR, alpha=0.7,
                  label='Neutral (0.8 ≤ φ ≤ 1.2)'),
            Patch(facecolor=NEGATIVE_EPISTASIS_COLOR, alpha=0.7,
                  label='Negative epistasis (φ < 0.8)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        plt.tight_layout()
        filename = f'glm_log_link_trigenic_{trait}.png'
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        save_path = osp.join(ffa_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_summary_statistics(results: Dict):
    """
    Create summary plots showing counts of significant interactions.
    """
    # Prepare summary data
    summary_data = []

    for trait in results:
        if not results[trait]:
            continue

        summary_data.append({
            'trait': trait,
            'digenic_total': results[trait].get('n_digenic', 0),
            'digenic_sig': results[trait].get('n_sig_digenic', 0),
            'trigenic_total': results[trait].get('n_trigenic', 0),
            'trigenic_sig': results[trait].get('n_sig_trigenic', 0)
        })

    if not summary_data:
        print("No summary data to plot")
        return

    df_summary = pd.DataFrame(summary_data)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Percentage of significant interactions by FFA
    ax1 = axes[0, 0]

    ffa_labels = []
    digenic_pct = []
    trigenic_pct = []

    for trait in ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']:
        trait_data = df_summary[df_summary['trait'] == trait]
        if not trait_data.empty:
            row = trait_data.iloc[0]
            ffa_labels.append(FFA_LABELS.get(trait, trait))
            if row['digenic_total'] > 0:
                digenic_pct.append(100 * row['digenic_sig'] / row['digenic_total'])
            else:
                digenic_pct.append(0)
            if row['trigenic_total'] > 0:
                trigenic_pct.append(100 * row['trigenic_sig'] / row['trigenic_total'])
            else:
                trigenic_pct.append(0)

    x_pos = np.arange(len(ffa_labels))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, digenic_pct, width,
                   label='Digenic', color=DIGENIC_COLOR, alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, trigenic_pct, width,
                   label='Trigenic', color=TRIGENIC_COLOR, alpha=0.7)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xlabel('FFA Type', fontsize=10)
    ax1.set_ylabel('% Significant (p < 0.05)', fontsize=10)
    ax1.set_title('GLM Log Link: Significant Interactions by FFA', fontsize=11)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ffa_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(max(digenic_pct + trigenic_pct, default=10), 10) * 1.3)  # More space at top

    # Plot 2: Total counts
    ax2 = axes[0, 1]
    totals = df_summary[['digenic_total', 'digenic_sig',
                           'trigenic_total', 'trigenic_sig']].sum()

    categories = ['Digenic\n(Total)', 'Digenic\n(Sig)', 'Trigenic\n(Total)', 'Trigenic\n(Sig)']
    values = [totals['digenic_total'], totals['digenic_sig'],
             totals['trigenic_total'], totals['trigenic_sig']]

    # Create bars with different alphas
    colors = ['#7191A9', '#7191A9', '#B73C39', '#B73C39']
    alphas = [0.5, 0.9, 0.5, 0.9]

    bars = []
    for i, (cat, val, col, alpha) in enumerate(zip(categories, values, colors, alphas)):
        bar = ax2.bar(i, val, color=col, alpha=alpha)
        bars.append(bar)

    # Add value labels
    for i, val in enumerate(values):
        ax2.text(i, val + 5, f'{int(val)}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories)

    ax2.set_ylabel('Number of Interactions', fontsize=10)
    ax2.set_title('GLM Log Link: Total Interaction Counts', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Pseudo R-squared by FFA
    ax3 = axes[1, 0]
    pseudo_r2_values = []
    pseudo_r2_labels = []

    for trait in ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']:
        if trait in results and 'pseudo_r_squared' in results[trait]:
            pseudo_r2_labels.append(FFA_LABELS.get(trait, trait))
            pseudo_r2_values.append(results[trait]['pseudo_r_squared'])

    if pseudo_r2_values:
        x_pos = np.arange(len(pseudo_r2_labels))
        bars = ax3.bar(x_pos, pseudo_r2_values, color='#34699D', alpha=0.7)

        # Add value labels
        for i, val in enumerate(pseudo_r2_values):
            ax3.text(i, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(pseudo_r2_labels, rotation=45, ha='right')
        ax3.set_ylabel('Pseudo-R² (McFadden)', fontsize=10)
        ax3.set_title('GLM Log Link: Model Fit by FFA', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, max(pseudo_r2_values) * 1.2)

    # Plot 4: P-value distribution
    ax4 = axes[1, 1]
    all_pvalues = []

    for trait in results:
        if 'epistasis' in results[trait]:
            for k, v in results[trait]['epistasis'].items():
                all_pvalues.append(v['pvalue'])

    if all_pvalues:
        ax4.hist(all_pvalues, bins=50, alpha=0.7, color='#34699D', edgecolor='black')
        ax4.axvline(x=0.05, color='red', linestyle='--', label='p=0.05', linewidth=2)
        ax4.set_xlabel('P-value', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('GLM Log Link: P-value Distribution', fontsize=11)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.suptitle('GLM Log Link Epistatic Interactions - Summary Statistics', fontsize=14)
    plt.tight_layout()

    filename = 'glm_log_link_summary.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    save_path = osp.join(ffa_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("Loading GLM log link results...")
    results_file = RESULTS_DIR / 'glm_log_link_results.pkl'

    if not results_file.exists():
        print(f"Results file not found at {results_file}")
        print("Please run glm_log_link_epistatic_interactions.py first")
        return

    results = load_results(results_file)

    print("\nCreating visualizations...")

    # Create plots
    print("1. Digenic interaction plots...")
    plot_digenic_interactions(results)

    print("2. Trigenic interaction plots...")
    plot_trigenic_interactions(results)

    print("3. Summary statistics...")
    plot_summary_statistics(results)

    print(f"\nAll visualizations saved to {ASSET_IMAGES_DIR}")
    print("Visualization complete!")


if __name__ == "__main__":
    main()
