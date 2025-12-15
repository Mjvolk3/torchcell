# experiments/011-kuzmin-tmi/scripts/query-comparison-006-009-010-011
# [[experiments.011-kuzmin-tmi.scripts.query-comparison-006-009-010-011]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/011-kuzmin-tmi/scripts/query-comparison-006-009-010-011
# Test file: experiments/011-kuzmin-tmi/scripts/test_query-comparison-006-009-010-011.py

import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import gaussian_kde

from torchcell.data import (
    Neo4jCellDataset,
    GenotypeAggregator,
    MeanExperimentDeduplicator,
)
from torchcell.data.graph_processor import SubgraphRepresentation
from torchcell.datasets import CodonFrequencyDataset
from torchcell.datasets.fungal_up_down_transformer import (
    FungalUpDownTransformerDataset,
)
from torchcell.metabolism.yeast_GEM import YeastGEM
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome
from torchcell.timestamp import timestamp


def load_dataset(
    exp_name, data_root, genome, codon_frequency, fudt_3prime, fudt_5prime
):
    """Load dataset and return gene_interaction labels"""
    print(f"Loading experiment {exp_name}...")

    query_path = (
        f"experiments/{exp_name}-kuzmin-tmi/queries/001_small_build.cql"
    )
    with open(query_path, "r") as f:
        query = f.read()

    dataset_root = osp.join(
        data_root,
        f"data/torchcell/experiments/{exp_name}-kuzmin-tmi/001-small-build",
    )

    dataset = Neo4jCellDataset(
        root=dataset_root,
        query=query,
        gene_set=genome.gene_set,
        uri="bolt://torchcell-database.ncsa.illinois.edu:7687",
        username="readonly",
        password="ReadOnly",
        incidence_graphs={
            "metabolism_bipartite": YeastGEM().bipartite_graph
        },
        node_embeddings={
            "codon_frequency": codon_frequency,
            "fudt_3prime": fudt_3prime,
            "fudt_5prime": fudt_5prime,
        },
        deduplicator=MeanExperimentDeduplicator,
        aggregator=GenotypeAggregator,  # type: ignore
        graph_processor=SubgraphRepresentation(),
    )

    print(f"  {exp_name}: {len(dataset)} experiments loaded")
    return dataset.label_df["gene_interaction"].values


def calculate_outliers_iqr(data):
    """Calculate outliers using IQR method"""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (data < lower) | (data > upper)
    return outliers, lower, upper


def calculate_statistics(data_dict):
    """Calculate summary statistics for all datasets"""
    stats = []
    for exp_name, data in data_dict.items():
        outliers, lower, upper = calculate_outliers_iqr(data)
        n_outliers = np.sum(outliers)
        pct_outliers = 100 * n_outliers / len(data)

        stats.append({
            'Dataset': f'Exp {exp_name}',
            'N': len(data),
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std': np.std(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Q1': np.percentile(data, 25),
            'Q3': np.percentile(data, 75),
            'Outliers': n_outliers,
            'Outliers %': pct_outliers,
        })

    return pd.DataFrame(stats)


def plot_distributions(data_dict, save_dir, mplstyle_path):
    """Create Figure 1: Distribution Comparison (KDE + Box plots)"""

    # Apply torchcell style
    plt.style.use(mplstyle_path)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Define colors from torchcell palette
    colors = {
        '006': '#000000',  # Black
        '009': '#D86E2F',  # Orange
        '010': '#7191A9',  # Blue
        '011': '#6B8D3A',  # Green
    }

    exp_names = ['006', '009', '010', '011']

    # Panel A: Overlapping KDE plots
    for exp_name in exp_names:
        data = data_dict[exp_name]
        ax1.hist(
            data,
            bins=100,
            density=True,
            alpha=0.3,
            color=colors[exp_name],
            label=f'{exp_name} (n={len(data):,})',
        )
        # Add KDE
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 500)
        ax1.plot(
            x_range, kde(x_range), color=colors[exp_name], linewidth=2.5
        )

    ax1.set_xlabel('Gene Interaction (Fitness)', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title('Distribution Comparison', fontsize=18, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.grid(True, alpha=0.3)

    # Panel B: Box plots with outliers
    box_data = [data_dict[exp] for exp in exp_names]
    bp = ax2.boxplot(
        box_data,
        labels=exp_names,
        patch_artist=True,
        showfliers=True,
        widths=0.6,
        flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.3),
    )

    # Color the boxes
    for patch, exp_name in zip(bp['boxes'], exp_names):
        patch.set_facecolor(colors[exp_name])
        patch.set_alpha(0.7)
        patch.set_linewidth(2)

    # Enhance whiskers and medians
    for whisker in bp['whiskers']:
        whisker.set(linewidth=2)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2.5)

    ax2.set_xlabel('Experiment', fontsize=16)
    ax2.set_ylabel('Gene Interaction (Fitness)', fontsize=16)
    ax2.set_title('Box Plot Comparison', fontsize=18, fontweight='bold')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax2.grid(True, axis='y', alpha=0.3)

    # Add outlier count annotations
    for i, exp_name in enumerate(exp_names):
        outliers, _, _ = calculate_outliers_iqr(data_dict[exp_name])
        n_outliers = np.sum(outliers)
        pct_outliers = 100 * n_outliers / len(data_dict[exp_name])
        ax2.text(i + 1, ax2.get_ylim()[1] * 0.95,
                f'{n_outliers}\n({pct_outliers:.1f}%)',
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure
    save_path = osp.join(
        save_dir,
        f'distribution_comparison_006_009_010_011_{timestamp()}.png'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure 1 saved to: {save_path}")
    plt.close()


def plot_outlier_deep_dive(data_dict, save_dir, mplstyle_path):
    """Create Figure 2: Outlier Deep Dive - Focus on extreme values"""

    # Apply torchcell style
    plt.style.use(mplstyle_path)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Define colors from torchcell palette
    colors = {
        '006': '#000000',  # Black
        '009': '#D86E2F',  # Orange
        '010': '#7191A9',  # Blue
        '011': '#6B8D3A',  # Green
    }

    exp_names = ['006', '009', '010', '011']

    # Panel A: Histogram focusing on tails (extreme values)
    for exp_name in exp_names:
        data = data_dict[exp_name]
        # Count extreme negative values (< -0.15)
        extreme_neg = np.sum(data < -0.15)
        # Count extreme positive values (> 0.15)
        extreme_pos = np.sum(data > 0.15)

        # Plot histogram with focus on tails
        ax1.hist(
            data,
            bins=150,
            alpha=0.5,
            color=colors[exp_name],
            label=(
                f"{exp_name}: neg<-0.15: {extreme_neg} "
                f"({100*extreme_neg/len(data):.2f}%), "
                f"pos>0.15: {extreme_pos} "
                f"({100*extreme_pos/len(data):.2f}%)"
            ),
            edgecolor='black',
            linewidth=0.5,
        )

    ax1.set_xlabel('Gene Interaction (Fitness)', fontsize=16)
    ax1.set_ylabel('Count', fontsize=16)
    ax1.set_title('Extreme Value Distribution', fontsize=18, weight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.axvline(
        x=-0.15, color='red', linestyle='--', alpha=0.7, linewidth=2
    )
    ax1.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.grid(True, alpha=0.3)

    # Panel B: Cumulative count of outliers by threshold
    thresholds = np.linspace(0, 0.5, 100)
    for exp_name in exp_names:
        data = data_dict[exp_name]
        abs_data = np.abs(data)
        outlier_counts = [np.sum(abs_data > t) for t in thresholds]
        outlier_pcts = [100 * c / len(data) for c in outlier_counts]

        ax2.plot(
            thresholds,
            outlier_pcts,
            color=colors[exp_name],
            linewidth=3,
            label=f'{exp_name}',
            marker='o',
            markersize=4,
            markevery=10,
        )

    ax2.set_xlabel('Absolute Value Threshold', fontsize=16)
    ax2.set_ylabel('Percentage of Data (%)', fontsize=16)
    ax2.set_title(
        'Outlier Sensitivity to Threshold', fontsize=18, weight='bold'
    )
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.5])

    # Annotate IQR threshold region
    ax2.axhline(
        y=5,
        color='red',
        linestyle=':',
        alpha=0.5,
        linewidth=2,
        label='~5% (typical IQR outliers)',
    )

    plt.tight_layout()

    # Save figure
    save_path = osp.join(
        save_dir,
        f'outlier_deep_dive_006_009_010_011_{timestamp()}.png'
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved to: {save_path}")
    plt.close()


def main():
    load_dotenv()
    DATA_ROOT = os.getenv("DATA_ROOT")
    EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
    ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

    # Type guards for environment variables
    if DATA_ROOT is None:
        raise ValueError("DATA_ROOT environment variable not set")
    if EXPERIMENT_ROOT is None:
        raise ValueError("EXPERIMENT_ROOT environment variable not set")
    if ASSET_IMAGES_DIR is None:
        raise ValueError("ASSET_IMAGES_DIR environment variable not set")

    MPLSTYLE_PATH = osp.join(
        os.getcwd(), "torchcell", "torchcell.mplstyle"
    )

    # Create subdirectories for this analysis
    # Images go to ASSET_IMAGES_DIR
    images_dir = osp.join(
        ASSET_IMAGES_DIR,
        "query-comparison-006-009-010-011"
    )
    os.makedirs(images_dir, exist_ok=True)

    # CSV results go to EXPERIMENT_ROOT
    results_dir = osp.join(
        EXPERIMENT_ROOT,
        "011-kuzmin-tmi",
        "results",
        "query-comparison-006-009-010-011"
    )
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Dataset Comparison: Experiments 006, 009, 010, 011")
    print("=" * 60)
    print(f"Images directory: {images_dir}")
    print(f"Results directory: {results_dir}\n")

    # Load genome and embeddings (shared across all datasets)
    print("\nLoading genome and embeddings...")
    genome = SCerevisiaeGenome(osp.join(DATA_ROOT, "data/sgd/genome"))
    print(f"Gene set size: {len(genome.gene_set)}")

    fudt_3prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    fudt_5prime_dataset = FungalUpDownTransformerDataset(
        root="data/scerevisiae/fudt_embedding",
        genome=genome,
        model_name="species_downstream",
    )
    codon_frequency = CodonFrequencyDataset(
        root=osp.join(DATA_ROOT, "data/scerevisiae/codon_frequency_embedding"),
        genome=genome,
    )

    # Load all 4 datasets (just labels)
    exp_names = ['006', '009', '010', '011']
    data = {}

    print("\n" + "="*60)
    print("Loading Datasets...")
    print("="*60)

    for exp in exp_names:
        data[exp] = load_dataset(
            exp, DATA_ROOT, genome,
            codon_frequency, fudt_3prime_dataset, fudt_5prime_dataset
        )

    # Verify hypothesis
    print("\n" + "="*60)
    print("Hypothesis Check: len(009) + len(011) ≈ len(010)")
    print("="*60)
    len_009_011 = len(data['009']) + len(data['011'])
    len_010 = len(data['010'])
    difference = len_010 - len_009_011

    print(f"len(009) = {len(data['009']):,}")
    print(f"len(011) = {len(data['011']):,}")
    print(f"len(009) + len(011) = {len_009_011:,}")
    print(f"len(010) = {len_010:,}")
    print(f"Difference = {difference:,}")
    print(f"Ratio = {len_010 / len_009_011:.4f}")

    # Calculate statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    stats_df = calculate_statistics(data)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(stats_df.to_string(index=False))

    # Save statistics to CSV (to results directory)
    stats_path = osp.join(results_dir, f'comparison_statistics_{timestamp()}.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStatistics saved to: {stats_path}")

    # Generate Figure 1 (to images directory)
    print("\n" + "=" * 60)
    print("Generating Publication Figures...")
    print("=" * 60)
    plot_distributions(data, images_dir, MPLSTYLE_PATH)

    # Generate Figure 2: Outlier Deep Dive (to images directory)
    plot_outlier_deep_dive(data, images_dir, MPLSTYLE_PATH)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
