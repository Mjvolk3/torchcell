# experiments/008-xue-ffa/scripts/model_upset_plots
# [[experiments.008-xue-ffa.scripts.model_upset_plots]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/model_upset_plots
# Test file: experiments/008-xue-ffa/scripts/test_model_upset_plots.py


import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import glob
import warnings
warnings.filterwarnings('ignore')

try:
    from upsetplot import UpSet, from_contents
except ImportError:
    print("upsetplot not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'upsetplot'])
    from upsetplot import UpSet, from_contents

# Load environment variables
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)
else:
    # Fallback style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")


def load_latest_csv(filepath_str):
    """Load CSV file from specific filepath."""
    filepath = RESULTS_DIR / filepath_str
    if not filepath.exists():
        print(f"  Warning: File not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows from: {filepath_str}")
    return df


def normalize_ffa_names(df):
    """Normalize FFA names to standard format with colons."""
    if df is None:
        return None

    # Map GLM-style names (C140) to standard format (C14:0)
    ffa_map = {
        'C140': 'C14:0',
        'C160': 'C16:0',
        'C180': 'C18:0',
        'C161': 'C16:1',
        'C181': 'C18:1',
        'Total Titer': 'Total Titer'
    }

    # Create a copy to avoid modifying original
    df = df.copy()

    # Replace FFA names if they match the GLM format
    if 'ffa_type' in df.columns:
        df['ffa_type'] = df['ffa_type'].replace(ffa_map)

    return df


def load_all_model_data():
    """Load interaction data from all four models."""
    print("Loading interaction data from all models...")
    models = {
        'Mult': {  # Shortened for better plot labels
            'digenic': normalize_ffa_names(load_latest_csv('multiplicative_digenic_interactions_3_delta_normalized.csv')),
            'trigenic': normalize_ffa_names(load_latest_csv('multiplicative_trigenic_interactions_3_delta_normalized.csv'))
        },
        'Add': {  # Shortened for better plot labels
            'digenic': normalize_ffa_names(load_latest_csv('additive_digenic_interactions_3_delta_normalized.csv')),
            'trigenic': normalize_ffa_names(load_latest_csv('additive_trigenic_interactions_3_delta_normalized.csv'))
        },
        'OLS': {
            'digenic': normalize_ffa_names(load_latest_csv('glm_models/log_ols_digenic_interactions.csv')),
            'trigenic': normalize_ffa_names(load_latest_csv('glm_models/log_ols_trigenic_interactions.csv'))
        },
        'GLM': {
            'digenic': normalize_ffa_names(load_latest_csv('glm_log_link/glm_log_link_digenic_interactions.csv')),
            'trigenic': normalize_ffa_names(load_latest_csv('glm_log_link/glm_log_link_trigenic_interactions.csv'))
        }
    }
    return models


def get_significant_gene_sets(df, ffa_type=None, use_fdr=False):
    """Extract significant gene sets from a dataframe."""
    if df is None:
        return set()

    # Filter by FFA type if specified
    if ffa_type:
        df_filtered = df[df['ffa_type'] == ffa_type]
        print(f"    Filtered to {len(df_filtered)} rows for FFA type {ffa_type}")
    else:
        df_filtered = df

    # Use p-value significance by default (FDR is too conservative)
    sig_col = 'significant_fdr05' if use_fdr else 'significant_p05'

    if sig_col not in df_filtered.columns:
        print(f"Warning: {sig_col} not found in dataframe columns")
        print(f"Available columns: {list(df_filtered.columns)}")
        return set()

    sig_df = df_filtered[df_filtered[sig_col]]
    print(f"    Found {len(sig_df)} significant interactions")

    # Normalize gene_set format: convert colons to underscores and sort genes
    # This handles different separators used by different models
    normalized_sets = set()
    for gene_set in sig_df['gene_set'].values:
        # Replace colon with underscore
        gene_set = str(gene_set).replace(':', '_')
        # Sort genes alphabetically for consistency
        genes = sorted(gene_set.split('_'))
        normalized_sets.add('_'.join(genes))

    return normalized_sets


def plot_upset_diagram(sets_dict, title, fig):
    """
    Create an UpSet plot for multi-way set comparison.
    UpSet plots are the standard for comparing 4+ sets.
    """
    from itertools import combinations

    # Print set sizes for debugging
    print(f"\n  Creating upset plot for: {title}")
    for model, gene_sets in sets_dict.items():
        print(f"    {model}: {len(gene_sets)} gene sets")

    # Get all unique elements
    all_elements = set()
    for s in sets_dict.values():
        all_elements.update(s)

    if len(all_elements) == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No significant interactions',
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return

    print(f"    Total unique gene sets across all models: {len(all_elements)}")

    # Create a MultiIndex dataframe with all possible combinations
    model_names = list(sets_dict.keys())

    # Build membership matrix for each element
    data_dict = {}
    for element in all_elements:
        membership = tuple(element in sets_dict[model] for model in model_names)
        data_dict[element] = membership

    # Create MultiIndex
    index_tuples = list(data_dict.values())
    index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=model_names
    )

    # Count occurrences of each combination
    upset_data = pd.Series(1, index=index).groupby(level=model_names).sum()

    # Debug: print the upset data
    print(f"    Upset data shape: {upset_data.shape}")
    print(f"    Sample of upset data:")
    for idx, count in list(upset_data.items())[:5]:
        print(f"      {idx}: {count}")

    # Create UpSet plot
    upset = UpSet(
        upset_data,
        show_counts='%d',
        sort_by='cardinality',
        element_size=40
    )
    upset.plot(fig=fig)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)


def create_upset_comparison_plots():
    """Create UpSet plots comparing significant interactions across all models."""
    models_data = load_all_model_data()

    # Always include all 4 models (use empty sets for missing data)
    all_model_names = ['Mult', 'Add', 'OLS', 'GLM']

    # Check which have actual data
    models_with_data = [m for m in all_model_names
                        if models_data[m]['digenic'] is not None]
    print(f"Models with data: {models_with_data}")
    print(f"Models without data (will use empty sets): "
          f"{[m for m in all_model_names if m not in models_with_data]}")

    # Check if we have any data at all
    if len(models_with_data) == 0:
        print("No model data found")
        return

    # Get all FFA types from the first available model
    ffa_types = None
    for model in models_with_data:
        sample_df = models_data[model]['digenic']
        if sample_df is not None:
            ffa_types = sample_df['ffa_type'].unique()
            break

    if ffa_types is None:
        print("Could not determine FFA types from data")
        return

    print(f"FFA types found: {ffa_types}")

    # Create plots for each FFA type
    for ffa in ffa_types:
        print(f"\nProcessing {ffa}...")

        # Digenic interactions - include all models
        digenic_sets = {}
        for model in all_model_names:
            df = models_data[model]['digenic']
            digenic_sets[model] = get_significant_gene_sets(df, ffa)

        fig = plt.figure(figsize=(12, 6))
        plot_upset_diagram(
            digenic_sets,
            f'Significant Digenic Interactions - {ffa}',
            fig
        )
        ffa_clean = ffa.replace(':', '').replace(' ', '_')
        filename = f"upset_digenic_{ffa_clean}.png"
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        output_path = osp.join(ffa_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

        # Trigenic interactions - include all models
        trigenic_sets = {}
        for model in all_model_names:
            df = models_data[model]['trigenic']
            trigenic_sets[model] = get_significant_gene_sets(df, ffa)

        fig = plt.figure(figsize=(12, 6))
        plot_upset_diagram(
            trigenic_sets,
            f'Significant Trigenic Interactions - {ffa}',
            fig
        )
        filename = f"upset_trigenic_{ffa_clean}.png"
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        output_path = osp.join(ffa_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

        # All interactions combined - include all models
        all_sets = {}
        for model in all_model_names:
            dig = get_significant_gene_sets(models_data[model]['digenic'], ffa)
            tri = get_significant_gene_sets(models_data[model]['trigenic'], ffa)
            all_sets[model] = dig | tri

        fig = plt.figure(figsize=(12, 6))
        plot_upset_diagram(
            all_sets,
            f'All Significant Interactions - {ffa}',
            fig
        )
        filename = f"upset_all_{ffa_clean}.png"
        ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
        os.makedirs(ffa_dir, exist_ok=True)
        output_path = osp.join(ffa_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    # Create overall summary across all FFAs
    print("\nCreating overall summary...")

    # Digenic - all FFAs - include all models
    digenic_sets_all = {}
    for model in all_model_names:
        df = models_data[model]['digenic']
        digenic_sets_all[model] = get_significant_gene_sets(df, None)

    fig = plt.figure(figsize=(12, 6))
    plot_upset_diagram(
        digenic_sets_all,
        'Significant Digenic Interactions - All FFAs',
        fig
    )
    filename = "upset_digenic_all_ffas.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Trigenic - all FFAs - include all models
    trigenic_sets_all = {}
    for model in all_model_names:
        df = models_data[model]['trigenic']
        trigenic_sets_all[model] = get_significant_gene_sets(df, None)

    fig = plt.figure(figsize=(12, 6))
    plot_upset_diagram(
        trigenic_sets_all,
        'Significant Trigenic Interactions - All FFAs',
        fig
    )
    filename = "upset_trigenic_all_ffas.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # All interactions - all FFAs - include all models
    all_sets_all = {}
    for model in all_model_names:
        dig = get_significant_gene_sets(models_data[model]['digenic'], None)
        tri = get_significant_gene_sets(models_data[model]['trigenic'], None)
        all_sets_all[model] = dig | tri

    fig = plt.figure(figsize=(12, 6))
    plot_upset_diagram(
        all_sets_all,
        'All Significant Interactions - All FFAs',
        fig
    )
    filename = "upset_all_all_ffas.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    print("\nUpSet plot analysis complete!")


if __name__ == "__main__":
    create_upset_comparison_plots()
