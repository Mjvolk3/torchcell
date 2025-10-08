#!/usr/bin/env python3

"""
Comparison visualizations for Models A, B, and C.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
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
RESULTS_DIR = BASE_DIR / "results" / "glm_models"
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
MODEL_A_COLOR = '#7191A9'  # Light blue
MODEL_B_COLOR = '#B73C39'  # Red
MODEL_C_COLOR = '#6B8D3A'  # Green
CAPACITY_COLOR = '#34699D'  # Blue
COMPOSITION_COLOR = '#E6A23C'  # Orange
BOTH_COLOR = '#909399'  # Gray

# FFA labels
FFA_LABELS = {
    'C140': 'C14:0',
    'C160': 'C16:0',
    'C180': 'C18:0',
    'C161': 'C16:1',
    'C181': 'C18:1',
    'Total Titer': 'Total'
}


def load_all_results():
    """Load results from all three models."""
    results = {}

    # Load Model A and B results
    ab_file = RESULTS_DIR / 'glm_results.pkl'
    if ab_file.exists():
        with open(ab_file, 'rb') as f:
            ab_results = pickle.load(f)
            results.update(ab_results)

    # Load Model C results
    c_file = RESULTS_DIR / 'model_c_clr_results.pkl'
    if c_file.exists():
        with open(c_file, 'rb') as f:
            c_results = pickle.load(f)
            results.update(c_results)

    return results


def compare_pvalues_across_models(results):
    """Create scatter plots comparing p-values across models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    traits = ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']

    for idx, trait in enumerate(traits):
        ax = axes[idx // 3, idx % 3]

        # Collect p-values from Model A and B
        model_a_pvals = []
        model_b_pvals = []
        interaction_names = []

        if 'model_a' in results and trait in results['model_a']:
            epistasis_a = results['model_a'][trait].get('epistasis', {})

            for interaction, data in epistasis_a.items():
                interaction_names.append(interaction)
                model_a_pvals.append(data['pvalue'])

                # Find corresponding Model B p-value
                if 'model_b' in results and trait in results['model_b']:
                    epistasis_b = results['model_b'][trait].get('epistasis', {})
                    if interaction in epistasis_b:
                        model_b_pvals.append(epistasis_b[interaction]['pvalue'])
                    else:
                        model_b_pvals.append(1.0)
                else:
                    model_b_pvals.append(1.0)

        if model_a_pvals and model_b_pvals:
            # Convert to -log10 for better visualization
            log_a = -np.log10(np.array(model_a_pvals) + 1e-10)
            log_b = -np.log10(np.array(model_b_pvals) + 1e-10)

            # Color by significance
            colors = []
            for pa, pb in zip(model_a_pvals, model_b_pvals):
                if pa < 0.05 and pb < 0.05:
                    colors.append('green')  # Significant in both
                elif pa < 0.05 or pb < 0.05:
                    colors.append('orange')  # Significant in one
                else:
                    colors.append('gray')  # Not significant

            ax.scatter(log_a, log_b, c=colors, alpha=0.6, s=30)

            # Add diagonal line
            max_val = max(max(log_a), max(log_b))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

            # Add significance threshold lines
            threshold = -np.log10(0.05)
            ax.axvline(threshold, color='red', linestyle='--', alpha=0.3)
            ax.axhline(threshold, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('Model A: -log₁₀(p-value)', fontsize=10)
            ax.set_ylabel('Model B: -log₁₀(p-value)', fontsize=10)
            ax.set_title(FFA_LABELS.get(trait, trait), fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add correlation
            if len(log_a) > 1:
                corr = np.corrcoef(log_a, log_b)[0, 1]
                # Show more decimal places to avoid misleading r=1.00
                if corr > 0.995:
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                           fontsize=9, va='top')
                else:
                    ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                           fontsize=9, va='top')

            # Add legend for first subplot only
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.6, label='Significant in both'),
                    Patch(facecolor='orange', alpha=0.6, label='Significant in one'),
                    Patch(facecolor='gray', alpha=0.6, label='Not significant')
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.suptitle('Model A vs Model B: P-value Comparison', fontsize=14)
    plt.tight_layout()

    save_path = osp.join(ASSET_IMAGES_DIR, f'model_ab_pvalue_comparison_{timestamp()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_c_decomposition(results):
    """Plot Model C decomposition into capacity and composition effects.

    Classification threshold: |E| > 0.1 (corresponding to φ outside 0.9-1.1)
    - Capacity-only: |E_tot| > 0.1 and |E_mix| < 0.1
    - Composition-only: |E_tot| < 0.1 and |E_mix| > 0.1
    - Both: |E_tot| > 0.1 and |E_mix| > 0.1
    - Neither: both |E_tot| < 0.1 and |E_mix| < 0.1
    """
    if 'model_c' not in results:
        print("Model C results not found")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    chains = ['C140', 'C160', 'C180', 'C161', 'C181']

    for idx, chain in enumerate(chains):
        ax = axes[idx // 3, idx % 3]

        # Add text box explaining the threshold to first subplot only
        if idx == 0:
            textstr = 'Classification Threshold:\n|E| > 0.1 (φ outside 0.9-1.1)\n\nE = epistasis coefficient (log scale)\nφ = exp(E) = epistatic fold'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        combined_key = f'{chain}_combined'
        if combined_key not in results['model_c']:
            continue

        combined = results['model_c'][combined_key]

        # Count interactions by type
        types = {'capacity-only': 0, 'composition-only': 0, 'both': 0, 'neither': 0}

        for interaction, data in combined.items():
            types[data['type']] += 1

        # Create bar plot
        labels = list(types.keys())
        values = [types[k] for k in labels]
        colors_map = {
            'capacity-only': CAPACITY_COLOR,
            'composition-only': COMPOSITION_COLOR,
            'both': BOTH_COLOR,
            'neither': 'lightgray'
        }
        colors = [colors_map[k] for k in labels]

        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace('-', '\n') for l in labels], fontsize=9)
        ax.set_ylabel('Number of Interactions', fontsize=10)
        ax.set_title(f'{FFA_LABELS.get(chain, chain)}', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(values) * 1.15 if values else 10)

    # Add summary in the last subplot
    ax = axes[1, 2]

    # Calculate totals across all chains
    total_types = {'capacity-only': 0, 'composition-only': 0, 'both': 0, 'neither': 0}

    for chain in chains:
        combined_key = f'{chain}_combined'
        if combined_key in results['model_c']:
            combined = results['model_c'][combined_key]
            for interaction, data in combined.items():
                total_types[data['type']] += 1

    # Average across chains
    avg_types = {k: v/len(chains) for k, v in total_types.items()}

    labels = list(avg_types.keys())
    values = [avg_types[k] for k in labels]
    colors = [colors_map[k] for k in labels]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace('-', '\n') for l in labels], fontsize=9)
    ax.set_ylabel('Average Count', fontsize=10)
    ax.set_title('Average Across All FFAs', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model C: Capacity vs Composition Decomposition\n(Threshold: |E| > 0.1, corresponding to φ < 0.9 or φ > 1.1)', fontsize=14)
    plt.tight_layout()

    save_path = osp.join(ASSET_IMAGES_DIR, f'model_c_decomposition_{timestamp()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_significant_interactions_summary(results):
    """Create summary plot comparing significant interactions across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bar chart of significant interactions by model and FFA
    ax1 = axes[0, 0]

    traits = ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']
    model_names = ['Model A', 'Model B']

    sig_counts = {model: {'digenic': [], 'trigenic': []} for model in model_names}

    for trait in traits:
        for model, key in zip(model_names, ['model_a', 'model_b']):
            if key in results and trait in results[key]:
                result = results[key][trait]
                sig_counts[model]['digenic'].append(result.get('n_sig_digenic', 0))
                sig_counts[model]['trigenic'].append(result.get('n_sig_trigenic', 0))
            else:
                sig_counts[model]['digenic'].append(0)
                sig_counts[model]['trigenic'].append(0)

    x = np.arange(len(traits))
    width = 0.2

    bars1 = ax1.bar(x - width*1.5, sig_counts['Model A']['digenic'], width,
                   label='Model A (digenic)', color=MODEL_A_COLOR, alpha=0.7)
    bars2 = ax1.bar(x - width*0.5, sig_counts['Model A']['trigenic'], width,
                   label='Model A (trigenic)', color=MODEL_A_COLOR, alpha=0.9)
    bars3 = ax1.bar(x + width*0.5, sig_counts['Model B']['digenic'], width,
                   label='Model B (digenic)', color=MODEL_B_COLOR, alpha=0.7)
    bars4 = ax1.bar(x + width*1.5, sig_counts['Model B']['trigenic'], width,
                   label='Model B (trigenic)', color=MODEL_B_COLOR, alpha=0.9)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=6)

    ax1.set_xlabel('FFA Type', fontsize=10)
    ax1.set_ylabel('Significant Interactions', fontsize=10)
    ax1.set_title('Significant Interactions by Model and FFA\n(Max possible: 45 digenic, 120 trigenic per FFA)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([FFA_LABELS.get(t, t) for t in traits], rotation=45, ha='right')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 125)  # Set fixed y-limit to show scale clearly

    # Plot 2: Percentage of significant interactions
    ax2 = axes[0, 1]

    pct_data = []
    for model, key in zip(model_names, ['model_a', 'model_b']):
        digenic_sig = 0
        digenic_total = 0
        trigenic_sig = 0
        trigenic_total = 0

        if key in results:
            for trait in results[key]:
                if results[key][trait]:
                    digenic_sig += results[key][trait].get('n_sig_digenic', 0)
                    digenic_total += results[key][trait].get('n_digenic', 0)
                    trigenic_sig += results[key][trait].get('n_sig_trigenic', 0)
                    trigenic_total += results[key][trait].get('n_trigenic', 0)

        pct_data.append({
            'Model': model,
            'Digenic': 100 * digenic_sig / digenic_total if digenic_total > 0 else 0,
            'Trigenic': 100 * trigenic_sig / trigenic_total if trigenic_total > 0 else 0
        })

    df_pct = pd.DataFrame(pct_data)
    x_pos = np.arange(len(df_pct))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, df_pct['Digenic'], width,
                   label='Digenic', color='#7191A9', alpha=0.7)
    bars2 = ax2.bar(x_pos + width/2, df_pct['Trigenic'], width,
                   label='Trigenic', color='#B73C39', alpha=0.7)

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Model', fontsize=10)
    ax2.set_ylabel('% Significant', fontsize=10)
    ax2.set_title('Percentage of Significant Interactions', fontsize=11)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_pct['Model'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    # Plot 3: R-squared comparison
    ax3 = axes[1, 0]

    r2_data = {'Model A': [], 'Model B (pseudo-R²)': []}
    trait_labels = []

    for trait in traits:
        if 'model_a' in results and trait in results['model_a']:
            r2_a = results['model_a'][trait].get('r_squared', 0)
            r2_data['Model A'].append(r2_a)
        else:
            r2_data['Model A'].append(0)

        if 'model_b' in results and trait in results['model_b']:
            # Model B has pseudo-R-squared
            pseudo_r2 = results['model_b'][trait].get('pseudo_r_squared', 0)
            r2_data['Model B (pseudo-R²)'].append(pseudo_r2)
        else:
            r2_data['Model B (pseudo-R²)'].append(0)

        trait_labels.append(FFA_LABELS.get(trait, trait))

    # Plot both Model A R-squared and Model B pseudo-R-squared
    x_pos = np.arange(len(trait_labels))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, r2_data['Model A'], width,
                   label='Model A (R²)', color=MODEL_A_COLOR, alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, r2_data['Model B (pseudo-R²)'], width,
                   label='Model B (pseudo-R²)', color=MODEL_B_COLOR, alpha=0.7)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    ax3.set_xlabel('FFA Type', fontsize=10)
    ax3.set_ylabel('Variance Explained', fontsize=10)
    ax3.set_title('Model A (R²) vs Model B (Pseudo-R²): Explained Variance', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(trait_labels, rotation=45, ha='right')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.1)

    # Plot 4: Model C summary - as percentages
    ax4 = axes[1, 1]

    if 'model_c' in results and 'total' in results['model_c']:
        # Calculate total possible interactions (45 digenic + 120 trigenic = 165 per trait)
        total_possible = 165

        # Get capacity significant interactions
        capacity_sig = results['model_c']['total'].get('n_sig_digenic', 0) + \
                      results['model_c']['total'].get('n_sig_trigenic', 0)
        capacity_pct = 100 * capacity_sig / total_possible

        # Get composition significant interactions (average across chains)
        composition_sig = []
        for chain in ['C140', 'C160', 'C180', 'C161', 'C181']:
            if chain in results['model_c']:
                chain_sig = (results['model_c'][chain].get('n_sig_digenic', 0) +
                           results['model_c'][chain].get('n_sig_trigenic', 0))
                composition_sig.append(100 * chain_sig / total_possible)

        if composition_sig:
            avg_composition_pct = np.mean(composition_sig)

            categories = ['Capacity\n(Total)', 'Composition\n(Avg)']
            values = [capacity_pct, avg_composition_pct]
            colors = [CAPACITY_COLOR, COMPOSITION_COLOR]

            bars = ax4.bar(categories, values, color=colors, alpha=0.7)

            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

            ax4.set_ylabel('% Significant Interactions', fontsize=10)
            ax4.set_title('Model C: Capacity vs Composition\n(% of 165 possible interactions)', fontsize=11)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 100)

    plt.suptitle('GLM Models Comparison Summary', fontsize=14)
    plt.tight_layout()

    save_path = osp.join(ASSET_IMAGES_DIR, f'glm_models_summary_{timestamp()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function."""
    print("Loading results from all models...")
    results = load_all_results()

    if not results:
        print("No results found. Please run the GLM analyses first.")
        return

    print("Creating comparison visualizations...")

    # 1. Compare p-values between Model A and B
    print("1. P-value comparison (Model A vs B)...")
    compare_pvalues_across_models(results)

    # 2. Model C decomposition
    print("2. Model C capacity/composition decomposition...")
    plot_model_c_decomposition(results)

    # 3. Summary comparison
    print("3. Summary comparison across all models...")
    plot_significant_interactions_summary(results)

    print(f"\nAll visualizations saved to {ASSET_IMAGES_DIR}")
    print("Model comparison complete!")


if __name__ == "__main__":
    main()