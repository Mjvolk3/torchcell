# experiments/008-xue-ffa/scripts/plot_graph_enrichment
# [[experiments.008-xue-ffa.scripts.plot_graph_enrichment]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/plot_graph_enrichment
# Test file: experiments/008-xue-ffa/scripts/test_plot_graph_enrichment.py

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

# Load environment variables
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Results directory
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")
GRAPH_ENRICHMENT_DIR = RESULTS_DIR / "graph_enrichment"

# Apply torchcell style
STYLE_PATH = "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)


def load_latest_enrichment_csv(model_name: str, interaction_type: str, metric: str = 'enrichment',
                               sign: str = ''):
    """Load the most recent enrichment CSV file.

    Args:
        model_name: Model name ('multiplicative', 'additive', etc.)
        interaction_type: Interaction type ('digenic', 'trigenic', or 'trigenic_triangle'/'trigenic_connected')
        metric: Metric type ('enrichment' or 'graph_overlap')
        sign: Interaction sign filter ('', '_positive', '_negative')
    """
    # For trigenic interactions, we need to handle the metric (triangle/connected) placement
    # Files are saved as: model_trigenic{sign}_triangle_enrichment_*.csv
    # NOT as: model_trigenic_triangle{sign}_enrichment_*.csv

    if interaction_type.startswith('trigenic_'):
        # Extract the base and metric (e.g., 'trigenic_triangle' -> 'trigenic', 'triangle')
        parts = interaction_type.split('_', 1)
        if len(parts) == 2:
            base = parts[0]  # 'trigenic'
            trig_metric = parts[1]  # 'triangle' or 'connected'
            # Pattern: model_trigenic{sign}_{triangle/connected}_enrichment
            pattern = f"{model_name}_{base}{sign}_{trig_metric}_{metric}_*.csv"
        else:
            # Just 'trigenic' without metric
            pattern = f"{model_name}_{interaction_type}{sign}_{metric}_*.csv"
    else:
        # For digenic: model_digenic{sign}_enrichment
        pattern = f"{model_name}_{interaction_type}{sign}_{metric}_*.csv"

    # Remove wildcard from pattern if present
    filename = pattern.replace('_*.csv', '.csv')
    filepath = GRAPH_ENRICHMENT_DIR / filename
    if not filepath.exists():
        return None
    return pd.read_csv(filepath)


def clean_graph_name(graph_type: str) -> str:
    """Clean graph type name for display."""
    # Remove suffixes
    name = graph_type.replace('_edge', '').replace('_triangle', '').replace('_connected', '')
    # Clean string names
    name = name.replace('string12_0_', 'STRING: ')
    # Capitalize first letter of main graph types
    if name in ['physical', 'genetic', 'regulatory', 'tflink']:
        name = name.capitalize()
    return name


def get_graph_order():
    """Return standardized graph ordering for plots."""
    return [
        'physical_edge', 'genetic_edge', 'regulatory_edge', 'tflink_edge',
        'string12_0_neighborhood_edge', 'string12_0_fusion_edge',
        'string12_0_cooccurence_edge', 'string12_0_coexpression_edge',
        'string12_0_experimental_edge', 'string12_0_database_edge'
    ]


def reorder_enrichment_df(df: pd.DataFrame, graph_order: list) -> pd.DataFrame:
    """Reorder dataframe by standardized graph order."""
    # Create a mapping of graph types to their order
    order_map = {graph: i for i, graph in enumerate(graph_order)}

    # Add order column
    df['_order'] = df['graph_type'].map(lambda x: order_map.get(x, 999))

    # Sort by order
    df = df.sort_values('_order').drop('_order', axis=1)

    return df


def plot_digenic_enrichment(models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create enrichment plots for digenic interactions across all models."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    model_labels = {
        'multiplicative': 'Multiplicative',
        'additive': 'Additive',
        'log_ols': 'WT-diff Log-OLS',
        'glm_log_link': 'GLM Gamma'
    }

    # Get standardized graph order
    graph_order = get_graph_order()

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Load enrichment data
        enrichment_df = load_latest_enrichment_csv(model, 'digenic', 'enrichment')

        if enrichment_df is None:
            ax.text(0.5, 0.5, f'No data for {model}',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue

        # Reorder by standardized graph order
        enrichment_df = reorder_enrichment_df(enrichment_df, graph_order)

        # Clean graph names
        enrichment_df['graph_clean'] = enrichment_df['graph_type'].apply(clean_graph_name)

        # Create horizontal bar plot
        y_pos = np.arange(len(enrichment_df))

        # Color by significance (using green from torchcell.mplstyle)
        torchcell_green = '#4A9C60'
        colors = [torchcell_green if p < 0.05 else 'gray' for p in enrichment_df['p_value']]

        bars = ax.barh(y_pos, enrichment_df['fold_enrichment'], color=colors, alpha=0.7)

        # Add vertical line at fold enrichment = 1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Add value labels - skip NaN and suspicious cases
        for i, (bar, row) in enumerate(zip(bars, enrichment_df.itertuples())):
            width = bar.get_width()
            # Skip NaN values (undefined enrichment)
            if np.isnan(width):
                continue
            # Skip suspicious cases where enrichment is 0 but p-value is significant
            if width == 0 and row.p_value < 0.05:
                continue
            # Handle infinite enrichment
            if np.isinf(width):
                label = 'Inf'
                if row.p_value < 0.05:
                    label += '*'
                # Place label at a reasonable position
                ax.text(ax.get_xlim()[1] * 0.95, bar.get_y() + bar.get_height()/2,
                       label, ha='right', va='center', fontsize=12)
            else:
                label = f'{width:.2f}'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=12)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(enrichment_df['graph_clean'], fontsize=13)
        ax.set_xlabel('Fold Enrichment\n(Significant / Non-significant)', fontsize=14)
        ax.set_title(f'{model_labels[model]}\nDigenic Interactions', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=12)

        # Set x-axis limits
        max_val = enrichment_df['fold_enrichment'].replace([np.inf, -np.inf], np.nan).max()
        if not np.isnan(max_val):
            ax.set_xlim(0, max_val * 1.2)

    plt.suptitle('Graph Enrichment in Significant Digenic Interactions\n(* p < 0.05)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save
    filename = 'digenic_graph_enrichment.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved digenic enrichment plot to: {output_path}")


def plot_trigenic_enrichment(models=['multiplicative', 'additive', 'log_ols', 'glm_log_link'],
                             metric='triangle'):
    """Create enrichment plots for trigenic interactions across all models."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    model_labels = {
        'multiplicative': 'Multiplicative',
        'additive': 'Additive',
        'log_ols': 'WT-diff Log-OLS',
        'glm_log_link': 'GLM Gamma'
    }

    metric_label = {
        'triangle': 'Triangle',
        'connected': 'Connected'
    }

    # Get standardized graph order (replace _edge with appropriate suffix)
    graph_order_base = get_graph_order()
    suffix = '_triangle' if metric == 'triangle' else '_connected'
    graph_order = [g.replace('_edge', suffix) for g in graph_order_base]

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Load enrichment data
        enrichment_df = load_latest_enrichment_csv(model, f'trigenic_{metric}', 'enrichment')

        if enrichment_df is None:
            ax.text(0.5, 0.5, f'No data for {model}',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue

        # Reorder by standardized graph order
        enrichment_df = reorder_enrichment_df(enrichment_df, graph_order)

        # Clean graph names
        enrichment_df['graph_clean'] = enrichment_df['graph_type'].apply(clean_graph_name)

        # Create horizontal bar plot
        y_pos = np.arange(len(enrichment_df))

        # Color by significance (using green from torchcell.mplstyle)
        torchcell_green = '#4A9C60'
        colors = [torchcell_green if p < 0.05 else 'gray' for p in enrichment_df['p_value']]

        bars = ax.barh(y_pos, enrichment_df['fold_enrichment'], color=colors, alpha=0.7)

        # Add vertical line at fold enrichment = 1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Add value labels - skip NaN and suspicious cases
        for i, (bar, row) in enumerate(zip(bars, enrichment_df.itertuples())):
            width = bar.get_width()
            # Skip NaN values (undefined enrichment)
            if np.isnan(width):
                continue
            # Skip suspicious cases where enrichment is 0 but p-value is significant
            if width == 0 and row.p_value < 0.05:
                continue
            # Handle infinite enrichment
            if np.isinf(width):
                label = 'Inf'
                if row.p_value < 0.05:
                    label += '*'
                # Place label at a reasonable position
                ax.text(ax.get_xlim()[1] * 0.95, bar.get_y() + bar.get_height()/2,
                       label, ha='right', va='center', fontsize=12)
            else:
                label = f'{width:.2f}'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=12)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(enrichment_df['graph_clean'], fontsize=13)
        ax.set_xlabel('Fold Enrichment\n(Significant / Non-significant)', fontsize=14)
        ax.set_title(f'{model_labels[model]}\nTrigenic Interactions - {metric_label[metric]}',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=12)

        # Set x-axis limits
        max_val = enrichment_df['fold_enrichment'].replace([np.inf, -np.inf], np.nan).max()
        if not np.isnan(max_val):
            ax.set_xlim(0, max_val * 1.2)

    plt.suptitle(f'Graph Enrichment in Significant Trigenic Interactions ({metric_label[metric]})\n(* p < 0.05)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save
    filename = f'trigenic_{metric}_graph_enrichment.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trigenic {metric} enrichment plot to: {output_path}")


def plot_percentage_comparison(models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create percentage comparison plots showing sig vs non-sig overlap."""

    model_labels = {
        'multiplicative': 'Mult',
        'additive': 'Add',
        'log_ols': 'OLS',
        'glm_log_link': 'GLM'
    }

    # Collect data for all models
    all_data = []

    for model in models:
        enrichment_df = load_latest_enrichment_csv(model, 'digenic', 'enrichment')
        if enrichment_df is not None:
            for _, row in enrichment_df.iterrows():
                graph_clean = clean_graph_name(row['graph_type'])
                all_data.append({
                    'Model': model_labels[model],
                    'Graph': graph_clean,
                    'Significant (%)': row['sig_pct'],
                    'Non-significant (%)': row['nonsig_pct'],
                    'p_value': row['p_value']
                })

    if not all_data:
        print("No data available for percentage comparison")
        return

    df = pd.DataFrame(all_data)

    # Get standardized graph order and clean names
    graph_order_base = get_graph_order()
    graph_names_ordered = [clean_graph_name(g) for g in graph_order_base]

    # Filter df to only include graphs in our order
    df = df[df['Graph'].isin(graph_names_ordered)]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(18, 11))

    # Use ordered graphs
    graphs = graph_names_ordered
    x = np.arange(len(graphs))
    width = 0.15

    models_list = list(model_labels.values())

    for i, model in enumerate(models_list):
        model_data = df[df['Model'] == model]

        sig_values = []
        nonsig_values = []

        for graph in graphs:
            graph_data = model_data[model_data['Graph'] == graph]
            if len(graph_data) > 0:
                sig_values.append(graph_data['Significant (%)'].values[0])
                nonsig_values.append(graph_data['Non-significant (%)'].values[0])
            else:
                sig_values.append(0)
                nonsig_values.append(0)

        # Plot bars
        offset = (i - len(models_list)/2 + 0.5) * width
        ax.bar(x + offset, sig_values, width, label=f'{model} (Sig)',
              alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Graph Type', fontsize=15)
    ax.set_ylabel('% with Graph Overlap', fontsize=15)
    ax.set_title('Percentage of Interactions with Graph Overlap\n(Digenic Interactions)',
                fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = 'graph_overlap_percentage_comparison.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved percentage comparison plot to: {output_path}")


def plot_trigenic_percentage_comparison(models=['multiplicative', 'additive', 'log_ols', 'glm_log_link'],
                                         metric='triangle'):
    """Create percentage comparison plots for trigenic interactions."""

    model_labels = {
        'multiplicative': 'Mult',
        'additive': 'Add',
        'log_ols': 'OLS',
        'glm_log_link': 'GLM'
    }

    metric_label = {
        'triangle': 'Triangle',
        'connected': 'Connected'
    }

    # Collect data for all models
    all_data = []

    for model in models:
        enrichment_df = load_latest_enrichment_csv(model, f'trigenic_{metric}', 'enrichment')
        if enrichment_df is not None:
            for _, row in enrichment_df.iterrows():
                graph_clean = clean_graph_name(row['graph_type'])
                all_data.append({
                    'Model': model_labels[model],
                    'Graph': graph_clean,
                    'Significant (%)': row['sig_pct'],
                    'Non-significant (%)': row['nonsig_pct'],
                    'p_value': row['p_value']
                })

    if not all_data:
        print(f"No data available for trigenic {metric} percentage comparison")
        return

    df = pd.DataFrame(all_data)

    # Get standardized graph order and clean names
    graph_order_base = get_graph_order()
    suffix = '_triangle' if metric == 'triangle' else '_connected'
    graph_order = [g.replace('_edge', suffix) for g in graph_order_base]
    graph_names_ordered = [clean_graph_name(g) for g in graph_order]

    # Filter df to only include graphs in our order
    df = df[df['Graph'].isin(graph_names_ordered)]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(18, 11))

    # Use ordered graphs
    graphs = graph_names_ordered
    x = np.arange(len(graphs))
    width = 0.15

    models_list = list(model_labels.values())

    for i, model in enumerate(models_list):
        model_data = df[df['Model'] == model]

        sig_values = []
        nonsig_values = []

        for graph in graphs:
            graph_data = model_data[model_data['Graph'] == graph]
            if len(graph_data) > 0:
                sig_values.append(graph_data['Significant (%)'].values[0])
                nonsig_values.append(graph_data['Non-significant (%)'].values[0])
            else:
                sig_values.append(0)
                nonsig_values.append(0)

        # Plot bars
        offset = (i - len(models_list)/2 + 0.5) * width
        ax.bar(x + offset, sig_values, width, label=f'{model} (Sig)',
              alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Graph Type', fontsize=15)
    ax.set_ylabel(f'% with Graph {metric_label[metric]}', fontsize=15)
    ax.set_title(f'Percentage of Interactions with Graph {metric_label[metric]}\n(Trigenic Interactions)',
                fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = f'trigenic_{metric}_percentage_comparison.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trigenic {metric} percentage comparison plot to: {output_path}")


def plot_digenic_enrichment_by_sign(sign: str = 'positive',
                                     models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create enrichment plots for digenic interactions partitioned by sign."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    model_labels = {
        'multiplicative': 'Multiplicative',
        'additive': 'Additive',
        'log_ols': 'WT-diff Log-OLS',
        'glm_log_link': 'GLM Gamma'
    }

    sign_label = sign.capitalize()
    sign_suffix = f'_{sign}'

    # Get standardized graph order
    graph_order = get_graph_order()

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Load enrichment data
        enrichment_df = load_latest_enrichment_csv(model, 'digenic', 'enrichment', sign=sign_suffix)

        if enrichment_df is None:
            ax.text(0.5, 0.5, f'No data for {model}',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue

        # Reorder by standardized graph order
        enrichment_df = reorder_enrichment_df(enrichment_df, graph_order)

        # Clean graph names
        enrichment_df['graph_clean'] = enrichment_df['graph_type'].apply(clean_graph_name)

        # Create horizontal bar plot
        y_pos = np.arange(len(enrichment_df))

        # Color by significance
        torchcell_green = '#4A9C60'
        colors = [torchcell_green if p < 0.05 else 'gray' for p in enrichment_df['p_value']]

        bars = ax.barh(y_pos, enrichment_df['fold_enrichment'], color=colors, alpha=0.7)

        # Add vertical line at fold enrichment = 1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Add value labels - skip NaN and suspicious cases
        for i, (bar, row) in enumerate(zip(bars, enrichment_df.itertuples())):
            width = bar.get_width()
            if np.isnan(width):
                continue
            if width == 0 and row.p_value < 0.05:
                continue
            if np.isinf(width):
                label = 'Inf'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(ax.get_xlim()[1] * 0.95, bar.get_y() + bar.get_height()/2,
                       label, ha='right', va='center', fontsize=12)
            else:
                label = f'{width:.2f}'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=12)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(enrichment_df['graph_clean'], fontsize=13)
        ax.set_xlabel('Fold Enrichment\n(Significant / Non-significant)', fontsize=14)
        ax.set_title(f'{model_labels[model]}\nDigenic {sign_label} Interactions', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=12)

        # Set x-axis limits
        max_val = enrichment_df['fold_enrichment'].replace([np.inf, -np.inf], np.nan).max()
        if not np.isnan(max_val):
            ax.set_xlim(0, max_val * 1.2)

    plt.suptitle(f'Graph Enrichment in Significant Digenic {sign_label} Interactions\n(* p < 0.05)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save
    filename = f'digenic_{sign}_graph_enrichment.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved digenic {sign} enrichment plot to: {output_path}")


def plot_trigenic_enrichment_by_sign(sign: str = 'positive', metric='triangle',
                                      models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create enrichment plots for trigenic interactions partitioned by sign."""

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    model_labels = {
        'multiplicative': 'Multiplicative',
        'additive': 'Additive',
        'log_ols': 'WT-diff Log-OLS',
        'glm_log_link': 'GLM Gamma'
    }

    metric_label = {
        'triangle': 'Triangle',
        'connected': 'Connected'
    }

    sign_label = sign.capitalize()
    sign_suffix = f'_{sign}'

    # Get standardized graph order
    graph_order_base = get_graph_order()
    suffix = '_triangle' if metric == 'triangle' else '_connected'
    graph_order = [g.replace('_edge', suffix) for g in graph_order_base]

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Load enrichment data
        enrichment_df = load_latest_enrichment_csv(model, f'trigenic_{metric}', 'enrichment', sign=sign_suffix)

        if enrichment_df is None:
            ax.text(0.5, 0.5, f'No data for {model}',
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            continue

        # Reorder by standardized graph order
        enrichment_df = reorder_enrichment_df(enrichment_df, graph_order)

        # Clean graph names
        enrichment_df['graph_clean'] = enrichment_df['graph_type'].apply(clean_graph_name)

        # Create horizontal bar plot
        y_pos = np.arange(len(enrichment_df))

        # Color by significance
        torchcell_green = '#4A9C60'
        colors = [torchcell_green if p < 0.05 else 'gray' for p in enrichment_df['p_value']]

        bars = ax.barh(y_pos, enrichment_df['fold_enrichment'], color=colors, alpha=0.7)

        # Add vertical line at fold enrichment = 1
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

        # Add value labels
        for i, (bar, row) in enumerate(zip(bars, enrichment_df.itertuples())):
            width = bar.get_width()
            if np.isnan(width):
                continue
            if width == 0 and row.p_value < 0.05:
                continue
            if np.isinf(width):
                label = 'Inf'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(ax.get_xlim()[1] * 0.95, bar.get_y() + bar.get_height()/2,
                       label, ha='right', va='center', fontsize=12)
            else:
                label = f'{width:.2f}'
                if row.p_value < 0.05:
                    label += '*'
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                       label, ha='left', va='center', fontsize=12)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(enrichment_df['graph_clean'], fontsize=13)
        ax.set_xlabel('Fold Enrichment\n(Significant / Non-significant)', fontsize=14)
        ax.set_title(f'{model_labels[model]}\nTrigenic {sign_label} Interactions - {metric_label[metric]}',
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.tick_params(axis='x', labelsize=12)

        # Set x-axis limits
        max_val = enrichment_df['fold_enrichment'].replace([np.inf, -np.inf], np.nan).max()
        if not np.isnan(max_val):
            ax.set_xlim(0, max_val * 1.2)

    plt.suptitle(f'Graph Enrichment in Significant Trigenic {sign_label} Interactions ({metric_label[metric]})\n(* p < 0.05)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Save
    filename = f'trigenic_{sign}_{metric}_graph_enrichment.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trigenic {sign} {metric} enrichment plot to: {output_path}")


def plot_percentage_comparison_by_sign(sign: str = 'positive',
                                        models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create percentage comparison plots for digenic interactions partitioned by sign."""

    model_labels = {
        'multiplicative': 'Mult',
        'additive': 'Add',
        'log_ols': 'OLS',
        'glm_log_link': 'GLM'
    }

    sign_label = sign.capitalize()
    sign_suffix = f'_{sign}'

    # Collect data for all models
    all_data = []

    for model in models:
        enrichment_df = load_latest_enrichment_csv(model, 'digenic', 'enrichment', sign=sign_suffix)
        if enrichment_df is not None:
            for _, row in enrichment_df.iterrows():
                graph_clean = clean_graph_name(row['graph_type'])
                all_data.append({
                    'Model': model_labels[model],
                    'Graph': graph_clean,
                    'Significant (%)': row['sig_pct'],
                    'Non-significant (%)': row['nonsig_pct'],
                    'p_value': row['p_value']
                })

    if not all_data:
        print(f"No data available for digenic {sign} percentage comparison")
        return

    df = pd.DataFrame(all_data)

    # Get standardized graph order and clean names
    graph_order_base = get_graph_order()
    graph_names_ordered = [clean_graph_name(g) for g in graph_order_base]

    # Filter df to only include graphs in our order
    df = df[df['Graph'].isin(graph_names_ordered)]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(18, 11))

    # Use ordered graphs
    graphs = graph_names_ordered
    x = np.arange(len(graphs))
    width = 0.15

    models_list = list(model_labels.values())

    for i, model in enumerate(models_list):
        model_data = df[df['Model'] == model]

        sig_values = []

        for graph in graphs:
            graph_data = model_data[model_data['Graph'] == graph]
            if len(graph_data) > 0:
                sig_values.append(graph_data['Significant (%)'].values[0])
            else:
                sig_values.append(0)

        # Plot bars
        offset = (i - len(models_list)/2 + 0.5) * width
        ax.bar(x + offset, sig_values, width, label=f'{model} (Sig)',
              alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Graph Type', fontsize=15)
    ax.set_ylabel('% with Graph Overlap', fontsize=15)
    ax.set_title(f'Percentage of Interactions with Graph Overlap\n(Digenic {sign_label} Interactions)',
                fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = f'digenic_{sign}_percentage_comparison.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved digenic {sign} percentage comparison plot to: {output_path}")


def plot_trigenic_percentage_comparison_by_sign(sign: str = 'positive', metric='triangle',
                                                 models=['multiplicative', 'additive', 'log_ols', 'glm_log_link']):
    """Create percentage comparison plots for trigenic interactions partitioned by sign."""

    model_labels = {
        'multiplicative': 'Mult',
        'additive': 'Add',
        'log_ols': 'OLS',
        'glm_log_link': 'GLM'
    }

    metric_label = {
        'triangle': 'Triangle',
        'connected': 'Connected'
    }

    sign_label = sign.capitalize()
    sign_suffix = f'_{sign}'

    # Collect data for all models
    all_data = []

    for model in models:
        enrichment_df = load_latest_enrichment_csv(model, f'trigenic_{metric}', 'enrichment', sign=sign_suffix)
        if enrichment_df is not None:
            for _, row in enrichment_df.iterrows():
                graph_clean = clean_graph_name(row['graph_type'])
                all_data.append({
                    'Model': model_labels[model],
                    'Graph': graph_clean,
                    'Significant (%)': row['sig_pct'],
                    'Non-significant (%)': row['nonsig_pct'],
                    'p_value': row['p_value']
                })

    if not all_data:
        print(f"No data available for trigenic {sign} {metric} percentage comparison")
        return

    df = pd.DataFrame(all_data)

    # Get standardized graph order and clean names
    graph_order_base = get_graph_order()
    suffix = '_triangle' if metric == 'triangle' else '_connected'
    graph_order = [g.replace('_edge', suffix) for g in graph_order_base]
    graph_names_ordered = [clean_graph_name(g) for g in graph_order]

    # Filter df to only include graphs in our order
    df = df[df['Graph'].isin(graph_names_ordered)]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(18, 11))

    # Use ordered graphs
    graphs = graph_names_ordered
    x = np.arange(len(graphs))
    width = 0.15

    models_list = list(model_labels.values())

    for i, model in enumerate(models_list):
        model_data = df[df['Model'] == model]

        sig_values = []

        for graph in graphs:
            graph_data = model_data[model_data['Graph'] == graph]
            if len(graph_data) > 0:
                sig_values.append(graph_data['Significant (%)'].values[0])
            else:
                sig_values.append(0)

        # Plot bars
        offset = (i - len(models_list)/2 + 0.5) * width
        ax.bar(x + offset, sig_values, width, label=f'{model} (Sig)',
              alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Graph Type', fontsize=15)
    ax.set_ylabel(f'% with Graph {metric_label[metric]}', fontsize=15)
    ax.set_title(f'Percentage of Interactions with Graph {metric_label[metric]}\n(Trigenic {sign_label} Interactions)',
                fontsize=17, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(fontsize=12, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = f'trigenic_{sign}_{metric}_percentage_comparison.png'
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trigenic {sign} {metric} percentage comparison plot to: {output_path}")


def main():
    """Generate all enrichment plots."""
    print("="*80)
    print("GRAPH ENRICHMENT VISUALIZATION")
    print("="*80)

    print("\n1. Creating digenic enrichment plots (all interactions)...")
    plot_digenic_enrichment()

    print("\n2. Creating digenic enrichment plots (positive interactions)...")
    plot_digenic_enrichment_by_sign(sign='positive')

    print("\n3. Creating digenic enrichment plots (negative interactions)...")
    plot_digenic_enrichment_by_sign(sign='negative')

    print("\n4. Creating trigenic triangle enrichment plots (all)...")
    plot_trigenic_enrichment(metric='triangle')

    print("\n5. Creating trigenic triangle enrichment plots (positive)...")
    plot_trigenic_enrichment_by_sign(sign='positive', metric='triangle')

    print("\n6. Creating trigenic triangle enrichment plots (negative)...")
    plot_trigenic_enrichment_by_sign(sign='negative', metric='triangle')

    print("\n7. Creating trigenic connected enrichment plots (all)...")
    plot_trigenic_enrichment(metric='connected')

    print("\n8. Creating trigenic connected enrichment plots (positive)...")
    plot_trigenic_enrichment_by_sign(sign='positive', metric='connected')

    print("\n9. Creating trigenic connected enrichment plots (negative)...")
    plot_trigenic_enrichment_by_sign(sign='negative', metric='connected')

    print("\n10. Creating digenic percentage comparison plot (all)...")
    plot_percentage_comparison()

    print("\n11. Creating digenic percentage comparison plot (positive)...")
    plot_percentage_comparison_by_sign(sign='positive')

    print("\n12. Creating digenic percentage comparison plot (negative)...")
    plot_percentage_comparison_by_sign(sign='negative')

    print("\n13. Creating trigenic triangle percentage comparison plot (all)...")
    plot_trigenic_percentage_comparison(metric='triangle')

    print("\n14. Creating trigenic triangle percentage comparison plot (positive)...")
    plot_trigenic_percentage_comparison_by_sign(sign='positive', metric='triangle')

    print("\n15. Creating trigenic triangle percentage comparison plot (negative)...")
    plot_trigenic_percentage_comparison_by_sign(sign='negative', metric='triangle')

    print("\n16. Creating trigenic connected percentage comparison plot (all)...")
    plot_trigenic_percentage_comparison(metric='connected')

    print("\n17. Creating trigenic connected percentage comparison plot (positive)...")
    plot_trigenic_percentage_comparison_by_sign(sign='positive', metric='connected')

    print("\n18. Creating trigenic connected percentage comparison plot (negative)...")
    plot_trigenic_percentage_comparison_by_sign(sign='negative', metric='connected')

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nAll plots saved to: {ASSET_IMAGES_DIR}")


if __name__ == "__main__":
    main()
