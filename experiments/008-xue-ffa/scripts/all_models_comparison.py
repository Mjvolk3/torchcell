#!/usr/bin/env python3

"""
Comprehensive comparison of all 4 models using CSV results.
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
import glob

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Define paths
BASE_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa")
RESULTS_DIR = BASE_DIR / "results"
GLM_RESULTS_DIR = RESULTS_DIR / "glm_models"
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)

# Model colors from torchcell.mplstyle
MULTIPLICATIVE_COLOR = '#7191A9'  # Blue-gray from style
ADDITIVE_COLOR = '#CC8250'  # Warm orange from style
MODEL_A_COLOR = '#6B8D3A'  # Green from style
MODEL_B_COLOR = '#B73C39'  # Red from style

# FFA mapping
FFA_MAP = {
    'C14:0': 'C14:0',
    'C16:0': 'C16:0',
    'C18:0': 'C18:0',
    'C16:1': 'C16:1',
    'C18:1': 'C18:1',
    'Total Titer': 'Total'
}


def load_simple_model_results(model_type='multiplicative'):
    """Load results from CSV files for simple models."""

    # Find the most recent files
    if model_type == 'multiplicative':
        digenic_pattern = str(RESULTS_DIR / 'digenic_interactions_*.csv')
        trigenic_pattern = str(RESULTS_DIR / 'trigenic_interactions_*.csv')
    else:  # additive
        digenic_pattern = str(RESULTS_DIR / 'additive_digenic_interactions_*.csv')
        trigenic_pattern = str(RESULTS_DIR / 'additive_trigenic_interactions_*.csv')

    digenic_files = glob.glob(digenic_pattern)
    trigenic_files = glob.glob(trigenic_pattern)

    results = {}

    if digenic_files:
        # Use most recent file
        digenic_file = sorted(digenic_files)[-1]
        dig_df = pd.read_csv(digenic_file)

        # Count significant interactions per FFA
        for ffa in ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1', 'Total Titer']:
            if ffa not in results:
                results[ffa] = {}

            ffa_data = dig_df[dig_df['ffa_type'] == ffa]
            results[ffa]['n_digenic'] = len(ffa_data)
            results[ffa]['n_sig_digenic'] = len(ffa_data[ffa_data['p_value'] < 0.05])

    if trigenic_files:
        # Use most recent file
        trigenic_file = sorted(trigenic_files)[-1]
        tri_df = pd.read_csv(trigenic_file)

        # Count significant interactions per FFA
        for ffa in ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1', 'Total Titer']:
            if ffa not in results:
                results[ffa] = {}

            ffa_data = tri_df[tri_df['ffa_type'] == ffa]
            results[ffa]['n_trigenic'] = len(ffa_data)
            results[ffa]['n_sig_trigenic'] = len(ffa_data[ffa_data['p_value'] < 0.05])

    return results


def load_glm_results():
    """Load GLM models A and B results."""
    results = {}

    glm_file = GLM_RESULTS_DIR / 'glm_results.pkl'
    if glm_file.exists():
        with open(glm_file, 'rb') as f:
            glm_data = pickle.load(f)

            # Process Model A
            if 'model_a' in glm_data:
                model_a = {}
                for trait, data in glm_data['model_a'].items():
                    # Map trait names
                    if trait == 'C140':
                        ffa = 'C14:0'
                    elif trait == 'C160':
                        ffa = 'C16:0'
                    elif trait == 'C180':
                        ffa = 'C18:0'
                    elif trait == 'C161':
                        ffa = 'C16:1'
                    elif trait == 'C181':
                        ffa = 'C18:1'
                    else:
                        ffa = trait

                    model_a[ffa] = {
                        'n_digenic': data.get('n_digenic', 45),
                        'n_sig_digenic': data.get('n_sig_digenic', 0),
                        'n_trigenic': data.get('n_trigenic', 120),
                        'n_sig_trigenic': data.get('n_sig_trigenic', 0),
                        'r_squared': data.get('r_squared', 0)
                    }
                results['model_a'] = model_a

            # Process Model B
            if 'model_b' in glm_data:
                model_b = {}
                for trait, data in glm_data['model_b'].items():
                    # Map trait names
                    if trait == 'C140':
                        ffa = 'C14:0'
                    elif trait == 'C160':
                        ffa = 'C16:0'
                    elif trait == 'C180':
                        ffa = 'C18:0'
                    elif trait == 'C161':
                        ffa = 'C16:1'
                    elif trait == 'C181':
                        ffa = 'C18:1'
                    else:
                        ffa = trait

                    model_b[ffa] = {
                        'n_digenic': data.get('n_digenic', 45),
                        'n_sig_digenic': data.get('n_sig_digenic', 0),
                        'n_trigenic': data.get('n_trigenic', 120),
                        'n_sig_trigenic': data.get('n_sig_trigenic', 0),
                        'pseudo_r_squared': data.get('pseudo_r_squared', 0)
                    }
                results['model_b'] = model_b

    return results


def create_comparison_plot():
    """Create the comprehensive comparison plot."""

    # Load all results
    print("Loading results from all models...")
    mult_results = load_simple_model_results('multiplicative')
    add_results = load_simple_model_results('additive')
    glm_results = load_glm_results()

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Setup
    ffa_order = ['C14:0', 'C16:0', 'C18:0', 'C16:1', 'C18:1', 'Total Titer']
    ffa_labels = [FFA_MAP.get(f, f) for f in ffa_order]

    models = ['Multiplicative', 'Additive', 'WT-diff Log-OLS', 'GLM Gamma']
    colors = [MULTIPLICATIVE_COLOR, ADDITIVE_COLOR, MODEL_A_COLOR, MODEL_B_COLOR]

    # Get data
    all_data = {
        'Multiplicative': mult_results,
        'Additive': add_results,
        'WT-diff Log-OLS': glm_results.get('model_a', {}),
        'GLM Gamma': glm_results.get('model_b', {})
    }

    # ========== Plot 1: Digenic Interactions ==========
    ax1 = axes[0, 0]
    x = np.arange(len(ffa_order))
    width = 0.2

    for i, (model, color) in enumerate(zip(models, colors)):
        counts = []
        for ffa in ffa_order:
            if ffa in all_data[model]:
                counts.append(all_data[model][ffa].get('n_sig_digenic', 0))
            else:
                counts.append(0)

        bars = ax1.bar(x + i * width - width * 1.5, counts, width,
                      label=model, color=color, alpha=0.8)

        # Add values on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{count}', ha='center', va='bottom', fontsize=7)

    ax1.set_xlabel('FFA Type', fontsize=11)
    ax1.set_ylabel('Significant Digenic Interactions', fontsize=11)
    ax1.set_title('Digenic Interactions (p < 0.05)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ffa_labels, rotation=45, ha='right')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 50)
    ax1.axhline(y=45, color='red', linestyle='--', alpha=0.3, linewidth=0.5)

    # ========== Plot 2: Trigenic Interactions ==========
    ax2 = axes[0, 1]

    for i, (model, color) in enumerate(zip(models, colors)):
        counts = []
        for ffa in ffa_order:
            if ffa in all_data[model]:
                counts.append(all_data[model][ffa].get('n_sig_trigenic', 0))
            else:
                counts.append(0)

        bars = ax2.bar(x + i * width - width * 1.5, counts, width,
                      label=model, color=color, alpha=0.8)

        # Add values on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{count}', ha='center', va='bottom', fontsize=7)

    ax2.set_xlabel('FFA Type', fontsize=11)
    ax2.set_ylabel('Significant Trigenic Interactions', fontsize=11)
    ax2.set_title('Trigenic Interactions (p < 0.05)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ffa_labels, rotation=45, ha='right')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 125)
    ax2.axhline(y=120, color='red', linestyle='--', alpha=0.3, linewidth=0.5)

    # ========== Plot 3: Overall Percentage ==========
    ax3 = axes[1, 0]

    # Calculate percentages
    pct_data = []
    for model in models:
        data = all_data[model]
        total_dig = 0
        sig_dig = 0
        total_tri = 0
        sig_tri = 0

        for ffa in ffa_order:
            if ffa in data:
                total_dig += 45  # Always 45 possible
                sig_dig += data[ffa].get('n_sig_digenic', 0)
                total_tri += 120  # Always 120 possible
                sig_tri += data[ffa].get('n_sig_trigenic', 0)

        pct_data.append({
            'Model': model,
            'Digenic': 100 * sig_dig / total_dig if total_dig > 0 else 0,
            'Trigenic': 100 * sig_tri / total_tri if total_tri > 0 else 0
        })

    # Plot
    x = np.arange(len(models))
    width = 0.35

    dig_pcts = [d['Digenic'] for d in pct_data]
    tri_pcts = [d['Trigenic'] for d in pct_data]

    bars1 = ax3.bar(x - width/2, dig_pcts, width, label='Digenic', alpha=0.7)
    bars2 = ax3.bar(x + width/2, tri_pcts, width, label='Trigenic', alpha=0.9)

    # Color by model
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        bar1.set_color(colors[i])
        bar2.set_color(colors[i])

    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('% Significant Interactions', fontsize=11)
    ax3.set_title('Overall Percentage of Significant Interactions', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 100)

    # ========== Plot 4: Summary Statistics ==========
    ax4 = axes[1, 1]

    # Total significant interactions
    totals = []
    for model in models:
        data = all_data[model]
        total = 0
        for ffa in ffa_order:
            if ffa in data:
                total += data[ffa].get('n_sig_digenic', 0)
                total += data[ffa].get('n_sig_trigenic', 0)
        totals.append(total)

    x = np.arange(len(models))
    bars = ax4.bar(x, totals, color=colors, alpha=0.8)

    # Add value and percentage labels
    max_possible = 6 * (45 + 120)  # 6 FFAs × (45 digenic + 120 trigenic)
    for bar, total in zip(bars, totals):
        pct = 100 * total / max_possible
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{total}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax4.set_xlabel('Model', fontsize=11)
    ax4.set_ylabel('Total Significant Interactions', fontsize=11)
    ax4.set_title(f'Total Significant Interactions (Max: {max_possible})', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1050)
    ax4.axhline(y=max_possible, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Maximum possible')
    ax4.legend(fontsize=8, loc='upper left')

    plt.suptitle('Comprehensive Model Comparison: Simple vs GLM Approaches', fontsize=14)
    plt.tight_layout()

    save_path = osp.join(ASSET_IMAGES_DIR, f'all_models_comparison_{timestamp()}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {save_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)

    for model in models:
        data = all_data[model]
        total_dig = 0
        sig_dig = 0
        total_tri = 0
        sig_tri = 0

        for ffa in ffa_order:
            if ffa in data:
                total_dig += 45
                sig_dig += data[ffa].get('n_sig_digenic', 0)
                total_tri += 120
                sig_tri += data[ffa].get('n_sig_trigenic', 0)

        print(f"\n{model}:")
        print(f"  Digenic: {sig_dig}/{total_dig} ({100*sig_dig/total_dig:.1f}%)")
        print(f"  Trigenic: {sig_tri}/{total_tri} ({100*sig_tri/total_tri:.1f}%)")
        print(f"  Total: {sig_dig + sig_tri}/{total_dig + total_tri} ({100*(sig_dig + sig_tri)/(total_dig + total_tri):.1f}%)")


if __name__ == "__main__":
    create_comparison_plot()