#!/usr/bin/env python3

"""
Comparison visualizations for Models A, B, and C.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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
    """Load results from CSV files."""
    results = {}

    # Load log-OLS results from CSV (no wildcards/timestamps)
    log_ols_path = RESULTS_DIR / 'log_ols_all_interactions.csv'

    if log_ols_path.exists():
        df = pd.read_csv(log_ols_path)

        ols_model = {}
        for trait in ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']:
            trait_data = df[df['ffa_type'] == trait]

            epistasis = {}
            for _, row in trait_data.iterrows():
                epistasis[row['gene_set']] = {
                    'E': row['interaction_score'],
                    'phi': row['epistatic_fold'],
                    'pvalue': row['p_value'],
                    'se': row['standard_error'],
                    'order': row['interaction_type']
                }

            digenic_count = len(trait_data[trait_data['interaction_type'] == 'digenic'])
            trigenic_count = len(trait_data[trait_data['interaction_type'] == 'trigenic'])
            sig_digenic = len(trait_data[(trait_data['interaction_type'] == 'digenic') & (trait_data['p_value'] < 0.05)])
            sig_trigenic = len(trait_data[(trait_data['interaction_type'] == 'trigenic') & (trait_data['p_value'] < 0.05)])

            ols_model[trait] = {
                'epistasis': epistasis,
                'n_digenic': digenic_count,
                'n_trigenic': trigenic_count,
                'n_sig_digenic': sig_digenic,
                'n_sig_trigenic': sig_trigenic
            }

        results['ols'] = ols_model

    # Load GLM log link results from CSV (no wildcards/timestamps)
    glm_path = BASE_DIR / 'results' / 'glm_log_link' / 'glm_log_link_all_interactions.csv'

    if glm_path.exists():
        df = pd.read_csv(glm_path)

        glm_model = {}
        for trait in ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']:
            trait_data = df[df['ffa_type'] == trait]

            epistasis = {}
            for _, row in trait_data.iterrows():
                epistasis[row['gene_set']] = {
                    'gamma': row['interaction_score'],
                    'phi': row['epistatic_fold'],
                    'pvalue': row['p_value'],
                    'se': row['standard_error'],
                    'order': row['interaction_type']
                }

            digenic_count = len(trait_data[trait_data['interaction_type'] == 'digenic'])
            trigenic_count = len(trait_data[trait_data['interaction_type'] == 'trigenic'])
            sig_digenic = len(trait_data[(trait_data['interaction_type'] == 'digenic') & (trait_data['p_value'] < 0.05)])
            sig_trigenic = len(trait_data[(trait_data['interaction_type'] == 'trigenic') & (trait_data['p_value'] < 0.05)])

            glm_model[trait] = {
                'epistasis': epistasis,
                'n_digenic': digenic_count,
                'n_trigenic': trigenic_count,
                'n_sig_digenic': sig_digenic,
                'n_sig_trigenic': sig_trigenic
            }

        results['glm'] = glm_model

    # Load CLR Composition Analysis results (Model C)
    clr_capacity_digenic_path = RESULTS_DIR / 'clr_capacity_digenic_interactions.csv'
    clr_composition_digenic_path = RESULTS_DIR / 'clr_composition_digenic_interactions.csv'

    if clr_capacity_digenic_path.exists() and clr_composition_digenic_path.exists():
        # Load capacity data
        cap_dig_df = pd.read_csv(clr_capacity_digenic_path)
        cap_tri_df = pd.read_csv(RESULTS_DIR / 'clr_capacity_trigenic_interactions.csv')

        # Load composition data
        comp_dig_df = pd.read_csv(clr_composition_digenic_path)
        comp_tri_df = pd.read_csv(RESULTS_DIR / 'clr_composition_trigenic_interactions.csv')

        clr_model = {}

        # Process capacity (using 'Total Capacity' as the FFA type)
        clr_model['Total Capacity'] = {
            'n_digenic': len(cap_dig_df),
            'n_trigenic': len(cap_tri_df),
            'n_sig_digenic': len(cap_dig_df[cap_dig_df['p_value'] < 0.05]),
            'n_sig_trigenic': len(cap_tri_df[cap_tri_df['p_value'] < 0.05])
        }

        # Process composition for each FFA type
        for ffa in ['C140', 'C160', 'C180', 'C161', 'C181']:
            comp_dig_ffa = comp_dig_df[comp_dig_df['ffa_type'] == ffa]
            comp_tri_ffa = comp_tri_df[comp_tri_df['ffa_type'] == ffa]

            clr_model[ffa] = {
                'n_digenic': len(comp_dig_ffa),
                'n_trigenic': len(comp_tri_ffa),
                'n_sig_digenic': len(comp_dig_ffa[comp_dig_ffa['p_value'] < 0.05]),
                'n_sig_trigenic': len(comp_tri_ffa[comp_tri_ffa['p_value'] < 0.05]),
                'type': 'composition'  # Mark as composition data
            }

        results['clr'] = clr_model

        # Also load combined effects for the decomposition plot
        clr_combined_path = RESULTS_DIR / 'clr_combined_effects.csv'
        if clr_combined_path.exists():
            results['clr_combined'] = pd.read_csv(clr_combined_path)

    return results


def compare_pvalues_across_models(results):
    """Create scatter plots comparing p-values across models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    traits = ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']

    for idx, trait in enumerate(traits):
        ax = axes[idx // 3, idx % 3]

        # Collect p-values from OLS and GLM models
        ols_pvals = []
        glm_pvals = []
        interaction_names = []

        if 'ols' in results and trait in results['ols']:
            epistasis_ols = results['ols'][trait].get('epistasis', {})

            for interaction, data in epistasis_ols.items():
                interaction_names.append(interaction)
                ols_pvals.append(data['pvalue'])

                # Find corresponding GLM p-value
                if 'glm' in results and trait in results['glm']:
                    epistasis_glm = results['glm'][trait].get('epistasis', {})
                    if interaction in epistasis_glm:
                        glm_pvals.append(epistasis_glm[interaction]['pvalue'])
                    else:
                        glm_pvals.append(1.0)
                else:
                    glm_pvals.append(1.0)

        if ols_pvals and glm_pvals:
            # Convert to -log10 for better visualization
            log_ols = -np.log10(np.array(ols_pvals) + 1e-10)
            log_glm = -np.log10(np.array(glm_pvals) + 1e-10)

            # Color by significance
            colors = []
            for p_ols, p_glm in zip(ols_pvals, glm_pvals):
                if p_ols < 0.05 and p_glm < 0.05:
                    colors.append('green')  # Significant in both
                elif p_ols < 0.05 or p_glm < 0.05:
                    colors.append('orange')  # Significant in one
                else:
                    colors.append('gray')  # Not significant

            ax.scatter(log_ols, log_glm, c=colors, alpha=0.6, s=30)

            # Add diagonal line
            max_val = max(max(log_ols), max(log_glm))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

            # Add significance threshold lines
            threshold = -np.log10(0.05)
            ax.axvline(threshold, color='red', linestyle='--', alpha=0.3)
            ax.axhline(threshold, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('WT-diff Log-OLS: -log₁₀(p-value)', fontsize=10)
            ax.set_ylabel('GLM Gamma: -log₁₀(p-value)', fontsize=10)
            ax.set_title(FFA_LABELS.get(trait, trait), fontsize=11)
            ax.grid(True, alpha=0.3)

            # Add correlation
            if len(log_ols) > 1:
                corr = np.corrcoef(log_ols, log_glm)[0, 1]
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

    plt.suptitle('WT-diff Log-OLS vs GLM Gamma: P-value Comparison', fontsize=14)
    plt.tight_layout()

    filename = 'ols_vs_glm_pvalue_comparison.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    save_path = osp.join(ffa_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_c_decomposition(results):
    """Plot CLR Composition Analysis decomposition into capacity and composition effects.

    Model C uses Centered Log-Ratio (CLR) transformation to separate:
    - Capacity: Total FFA production (sum of all FFAs)
    - Composition: Relative mixture of which FFAs are produced

    Classification threshold: |E| > 0.1 (corresponding to φ outside 0.9-1.1)
    - Capacity-only: Effect only on total production
    - Composition-only: Effect only on FFA mixture ratios
    - Both: Effect on both total and mixture
    - Neither: No significant effect
    """
    if 'clr_combined' not in results:
        print("CLR Composition Analysis results not found - run clr_composition_analysis.py first")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    chains = ['C140', 'C160', 'C180', 'C161', 'C181']

    # Load combined effects data
    df_combined = results['clr_combined']

    for idx, chain in enumerate(chains):
        ax = axes[idx // 3, idx % 3]

        # Add text box explaining the threshold to first subplot only
        if idx == 0:
            textstr = 'Classification Threshold:\n|E| > 0.1 (φ outside 0.9-1.1)\n\nE = epistasis coefficient (log scale)\nφ = exp(E) = epistatic fold'
            ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Filter data for this chain
        chain_data = df_combined[df_combined['ffa_type'] == chain]
        if chain_data.empty:
            continue

        # Count interactions by effect type
        types = chain_data['effect_type'].value_counts().to_dict()

        # Ensure all types are present
        for t in ['capacity-only', 'composition-only', 'both', 'neither']:
            if t not in types:
                types[t] = 0

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
        chain_data = df_combined[df_combined['ffa_type'] == chain]
        if not chain_data.empty:
            type_counts = chain_data['effect_type'].value_counts().to_dict()
            for t, count in type_counts.items():
                if t in total_types:
                    total_types[t] += count

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

    plt.suptitle('CLR Composition Analysis: Capacity vs Composition Decomposition\n(Threshold: |E| > 0.1, corresponding to φ < 0.9 or φ > 1.1)', fontsize=14)
    plt.tight_layout()

    filename = 'clr_decomposition.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    save_path = osp.join(ffa_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_significant_interactions_summary(results):
    """Create summary plot comparing significant interactions across all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bar chart of significant interactions by model and FFA
    ax1 = axes[0, 0]

    traits = ['C140', 'C160', 'C180', 'C161', 'C181', 'Total Titer']
    model_names = ['WT-diff Log-OLS', 'GLM Gamma']

    sig_counts = {model: {'digenic': [], 'trigenic': []} for model in model_names}

    for trait in traits:
        for model, key in zip(model_names, ['ols', 'glm']):
            if key in results and trait in results[key]:
                result = results[key][trait]
                sig_counts[model]['digenic'].append(result.get('n_sig_digenic', 0))
                sig_counts[model]['trigenic'].append(result.get('n_sig_trigenic', 0))
            else:
                sig_counts[model]['digenic'].append(0)
                sig_counts[model]['trigenic'].append(0)

    x = np.arange(len(traits))
    width = 0.2

    bars1 = ax1.bar(x - width*1.5, sig_counts['WT-diff Log-OLS']['digenic'], width,
                   label='WT-diff Log-OLS (digenic)', color=MODEL_A_COLOR, alpha=0.7)
    bars2 = ax1.bar(x - width*0.5, sig_counts['WT-diff Log-OLS']['trigenic'], width,
                   label='WT-diff Log-OLS (trigenic)', color=MODEL_A_COLOR, alpha=0.9)
    bars3 = ax1.bar(x + width*0.5, sig_counts['GLM Gamma']['digenic'], width,
                   label='GLM Gamma (digenic)', color=MODEL_B_COLOR, alpha=0.7)
    bars4 = ax1.bar(x + width*1.5, sig_counts['GLM Gamma']['trigenic'], width,
                   label='GLM Gamma (trigenic)', color=MODEL_B_COLOR, alpha=0.9)

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
    for model, key in zip(model_names, ['ols', 'glm']):
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

    # Plot 3: Total significant interactions (sum of digenic and trigenic)
    ax3 = axes[1, 0]

    total_data = {'WT-diff Log-OLS': {'digenic': 0, 'trigenic': 0},
                  'GLM Gamma': {'digenic': 0, 'trigenic': 0}}

    for model, key in zip(['WT-diff Log-OLS', 'GLM Gamma'], ['ols', 'glm']):
        if key in results:
            for trait in results[key]:
                if results[key][trait]:
                    total_data[model]['digenic'] += results[key][trait].get('n_sig_digenic', 0)
                    total_data[model]['trigenic'] += results[key][trait].get('n_sig_trigenic', 0)

    # Create bars
    x_pos = np.arange(len(total_data))
    width = 0.35

    digenic_totals = [total_data[m]['digenic'] for m in total_data.keys()]
    trigenic_totals = [total_data[m]['trigenic'] for m in total_data.keys()]

    bars1 = ax3.bar(x_pos - width/2, digenic_totals, width,
                   label='Digenic', color='#7191A9', alpha=0.7)
    bars2 = ax3.bar(x_pos + width/2, trigenic_totals, width,
                   label='Trigenic', color='#B73C39', alpha=0.7)

    # Add value labels and percentage of maximum
    max_digenic = 270  # 45 digenic * 6 FFAs
    max_trigenic = 720  # 120 trigenic * 6 FFAs

    for bar, total, max_val in zip(bars1, digenic_totals, [max_digenic]*len(bars1)):
        height = bar.get_height()
        pct = 100 * height / max_val
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=8)

    for bar, total, max_val in zip(bars2, trigenic_totals, [max_trigenic]*len(bars2)):
        height = bar.get_height()
        pct = 100 * height / max_val
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=8)

    # Add horizontal line for max possible
    ax3.axhline(y=max_trigenic, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax3.text(1.5, max_trigenic + 10, f'Maximum possible: {max_trigenic}',
            fontsize=8, ha='right', color='gray')

    ax3.set_xlabel('Model', fontsize=10)
    ax3.set_ylabel('Total Significant Interactions', fontsize=10)
    ax3.set_title('Total Significant Interactions (Max: 990)', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(list(total_data.keys()))
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1000)

    # Plot 4: CLR Composition Analysis (Model C)
    ax4 = axes[1, 1]

    if 'clr' in results:
        clr_data = results['clr']

        # Create bars for capacity and average composition
        capacity_dig = clr_data['Total Capacity']['n_sig_digenic']
        capacity_tri = clr_data['Total Capacity']['n_sig_trigenic']
        capacity_total = capacity_dig + capacity_tri

        # Calculate average composition across all FFAs
        comp_dig_total = 0
        comp_tri_total = 0
        n_ffas = 0
        for ffa in ['C140', 'C160', 'C180', 'C161', 'C181']:
            if ffa in clr_data:
                comp_dig_total += clr_data[ffa]['n_sig_digenic']
                comp_tri_total += clr_data[ffa]['n_sig_trigenic']
                n_ffas += 1

        avg_comp_dig = comp_dig_total / n_ffas if n_ffas > 0 else 0
        avg_comp_tri = comp_tri_total / n_ffas if n_ffas > 0 else 0
        avg_comp_total = avg_comp_dig + avg_comp_tri

        # Create grouped bar chart
        x = np.arange(2)
        width = 0.35

        dig_values = [capacity_dig, avg_comp_dig]
        tri_values = [capacity_tri, avg_comp_tri]

        bars1 = ax4.bar(x - width/2, dig_values, width, label='Digenic',
                       color='#7191A9', alpha=0.7)
        bars2 = ax4.bar(x + width/2, tri_values, width, label='Trigenic',
                       color='#B73C39', alpha=0.7)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

        # Add totals above bars
        for i, (d, t) in enumerate(zip(dig_values, tri_values)):
            total = d + t
            pct = 100 * total / 165  # 165 = 45 digenic + 120 trigenic
            ax4.text(i, d + t + 5, f'Total: {int(total)}\n({pct:.0f}%)',
                    ha='center', va='bottom', fontsize=7)

        ax4.set_ylabel('Significant Interactions', fontsize=10)
        ax4.set_title('CLR Composition Analysis (Model C)', fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Capacity\n(Total FFA)', 'Composition\n(Avg per FFA)'])
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 160)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'CLR Composition Analysis\n(Model C)\n\nRun clr_composition_analysis.py\nto generate data',
                ha='center', va='center', fontsize=10, color='gray', alpha=0.5)

    plt.suptitle('GLM Models Comparison Summary', fontsize=14)
    plt.tight_layout()

    filename = 'glm_models_summary.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    save_path = osp.join(ffa_dir, filename)
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

    # 1. Compare p-values between OLS and GLM
    print("1. P-value comparison (OLS vs GLM)...")
    compare_pvalues_across_models(results)

    # 2. CLR Composition Analysis (Model C)
    print("2. CLR Composition Analysis (Model C) decomposition...")
    plot_model_c_decomposition(results)

    # 3. Summary comparison
    print("3. Summary comparison across all models...")
    plot_significant_interactions_summary(results)

    print(f"\nAll visualizations saved to {ASSET_IMAGES_DIR}")
    print("Model comparison complete!")


if __name__ == "__main__":
    main()