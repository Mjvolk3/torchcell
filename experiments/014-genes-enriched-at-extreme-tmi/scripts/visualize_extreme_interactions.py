# experiments/014-genes-enriched-at-extreme-tmi/scripts/visualize_extreme_interactions.py
# [[experiments.014-genes-enriched-at-extreme-tmi.scripts.visualize_extreme_interactions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/014-genes-enriched-at-extreme-tmi/scripts/visualize_extreme_interactions

"""
Visualization script for extreme TMI analysis.
Loads pre-computed results from CSV files and creates publication-quality plots.
This is the fast step - can be re-run many times to tweak visualizations.
"""

import os
import os.path as osp
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

if EXPERIMENT_ROOT is None:
    raise ValueError("EXPERIMENT_ROOT environment variable not set")
if ASSET_IMAGES_DIR is None:
    raise ValueError("ASSET_IMAGES_DIR environment variable not set")

# Analysis parameters (must match those used in analyze_extreme_interactions.py)
SCORE_THRESHOLD = 0.1
P_VALUE_THRESHOLD = 0.05

# Custom colors from torchcell.mplstyle
CUSTOM_BLUE = "#7191A9"
CUSTOM_GREEN = "#6B8D3A"


def load_results():
    """Load pre-computed results from CSV files."""
    assert EXPERIMENT_ROOT is not None, "EXPERIMENT_ROOT must be set"

    results_dir = osp.join(
        EXPERIMENT_ROOT, "014-genes-enriched-at-extreme-tmi", "results"
    )

    print("Loading pre-computed results...")
    print(f"From: {results_dir}\n")

    all_df = pd.read_csv(osp.join(results_dir, "uncharacterized_extreme_interactions_all.csv"))
    high_conf_df = pd.read_csv(osp.join(results_dir, "uncharacterized_extreme_interactions_high_conf.csv"))
    all_gene_counts = pd.read_csv(osp.join(results_dir, "uncharacterized_gene_counts_all.csv"))
    high_conf_gene_counts = pd.read_csv(osp.join(results_dir, "uncharacterized_gene_counts_high_conf.csv"))
    all_enrichment = pd.read_csv(osp.join(results_dir, "uncharacterized_enrichment_fractions_all.csv"))

    # Try to load high-conf enrichment (may not exist if no high-conf interactions)
    high_conf_enrichment_file = osp.join(results_dir, "uncharacterized_enrichment_fractions_high_conf.csv")
    high_conf_enrichment = None
    if osp.exists(high_conf_enrichment_file):
        high_conf_enrichment = pd.read_csv(high_conf_enrichment_file)

    print(f"Loaded {len(all_df)} extreme interactions")
    print(f"Loaded {len(high_conf_df)} high-confidence interactions")
    print(f"Loaded {len(all_gene_counts)} genes with extreme interactions")
    print(f"Loaded {len(all_enrichment)} gene enrichment fractions\n")

    return all_df, high_conf_df, all_gene_counts, high_conf_gene_counts, all_enrichment, high_conf_enrichment


def create_enrichment_visualization(enrichment_df: pd.DataFrame, all_df: pd.DataFrame):
    """Create visualization for gene enrichment fractions with positive/negative split."""
    assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set"

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram of enrichment fractions (all)
    ax = axes[0, 0]
    ax.hist(
        enrichment_df["enrichment_fraction"],
        bins=50,
        color=CUSTOM_BLUE,
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Enrichment Fraction (Extreme / Total Interactions)")
    ax.set_ylabel("Number of Uncharacterized Genes")
    ax.set_title(f"Distribution of Enrichment Fractions\n" + r"$(|\tau| > " + f"{SCORE_THRESHOLD})$")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.95,
        0.95,
        f"Total genes: {len(enrichment_df)}\n"
        f"Mean: {enrichment_df['enrichment_fraction'].mean():.3f}\n"
        f"Median: {enrichment_df['enrichment_fraction'].median():.3f}\n"
        f"Max: {enrichment_df['enrichment_fraction'].max():.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Top 30 genes by enrichment fraction (all)
    ax = axes[0, 1]
    top_30 = enrichment_df.head(30)
    ax.barh(range(len(top_30)), top_30["enrichment_fraction"], color=CUSTOM_BLUE, alpha=0.7)
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30["gene"], fontsize=8)
    ax.set_xlabel("Enrichment Fraction (Extreme / Total)")
    ax.set_title(f"Top 30 Genes by Enrichment Fraction\n" + r"$(|\tau| > " + f"{SCORE_THRESHOLD})$")
    ax.set_xlim([0.0, 1.0])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Add fraction labels on the bars
    for i, (_, row) in enumerate(top_30.iterrows()):
        ax.text(
            min(row["enrichment_fraction"] + 0.02, 0.98),
            i,
            f"{row['enrichment_fraction']:.2f}",
            va="center",
            fontsize=7,
        )

    # Calculate enrichment for positive interactions
    positive_df = all_df[all_df["gene_interaction"] > SCORE_THRESHOLD].copy()

    # Count positive interactions per gene
    pos_gene_counter = Counter()
    for genes in positive_df["uncharacterized_genes"]:
        for gene in genes.split(","):
            if gene:
                pos_gene_counter[gene] += 1
    pos_gene_counts = pd.DataFrame([
        {"gene": gene, "positive_count": count}
        for gene, count in pos_gene_counter.items()
    ])

    # Calculate enrichment for negative interactions
    negative_df = all_df[all_df["gene_interaction"] < -SCORE_THRESHOLD].copy()

    # Count negative interactions per gene
    neg_gene_counter = Counter()
    for genes in negative_df["uncharacterized_genes"]:
        for gene in genes.split(","):
            if gene:
                neg_gene_counter[gene] += 1
    neg_gene_counts = pd.DataFrame([
        {"gene": gene, "negative_count": count}
        for gene, count in neg_gene_counter.items()
    ])

    # Merge with total counts and calculate positive enrichment
    pos_enrichment = enrichment_df.copy()
    pos_enrichment = pos_enrichment.merge(
        pos_gene_counts.rename(columns={"uncharacterized_genes": "gene"}),
        on="gene",
        how="left"
    )
    pos_enrichment["positive_count"] = pos_enrichment["positive_count"].fillna(0)
    pos_enrichment["pos_enrichment_fraction"] = pos_enrichment["positive_count"] / pos_enrichment["total_interactions"]
    pos_enrichment = pos_enrichment.sort_values("pos_enrichment_fraction", ascending=False)

    # Merge with total counts and calculate negative enrichment
    neg_enrichment = enrichment_df.copy()
    neg_enrichment = neg_enrichment.merge(
        neg_gene_counts.rename(columns={"uncharacterized_genes": "gene"}),
        on="gene",
        how="left"
    )
    neg_enrichment["negative_count"] = neg_enrichment["negative_count"].fillna(0)
    neg_enrichment["neg_enrichment_fraction"] = neg_enrichment["negative_count"] / neg_enrichment["total_interactions"]
    neg_enrichment = neg_enrichment.sort_values("neg_enrichment_fraction", ascending=False)

    # Top 30 genes by positive enrichment
    ax = axes[1, 0]
    top_30_pos = pos_enrichment.head(30)
    ax.barh(range(len(top_30_pos)), top_30_pos["pos_enrichment_fraction"], color=CUSTOM_GREEN, alpha=0.7)
    ax.set_yticks(range(len(top_30_pos)))
    ax.set_yticklabels(top_30_pos["gene"], fontsize=8)
    ax.set_xlabel("Enrichment Fraction (Positive / Total)")
    ax.set_title(f"Top 30 Genes by Positive Enrichment\n" + r"$(\tau > " + f"{SCORE_THRESHOLD})$")
    ax.set_xlim([0.0, 1.0])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    for i, (_, row) in enumerate(top_30_pos.iterrows()):
        ax.text(
            min(row["pos_enrichment_fraction"] + 0.02, 0.98),
            i,
            f"{row['pos_enrichment_fraction']:.2f}",
            va="center",
            fontsize=7,
        )

    # Top 30 genes by negative enrichment
    ax = axes[1, 1]
    top_30_neg = neg_enrichment.head(30)
    ax.barh(range(len(top_30_neg)), top_30_neg["neg_enrichment_fraction"], color="#A97171", alpha=0.7)
    ax.set_yticks(range(len(top_30_neg)))
    ax.set_yticklabels(top_30_neg["gene"], fontsize=8)
    ax.set_xlabel("Enrichment Fraction (Negative / Total)")
    ax.set_title(f"Top 30 Genes by Negative Enrichment\n" + r"$(\tau < -" + f"{SCORE_THRESHOLD})$")
    ax.set_xlim([0.0, 1.0])
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    for i, (_, row) in enumerate(top_30_neg.iterrows()):
        ax.text(
            min(row["neg_enrichment_fraction"] + 0.02, 0.98),
            i,
            f"{row['neg_enrichment_fraction']:.2f}",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()

    # Save to experiment-specific subdirectory
    output_dir = osp.join(ASSET_IMAGES_DIR, "014-genes-enriched-at-extreme-tmi")
    os.makedirs(output_dir, exist_ok=True)

    fig_path = osp.join(
        output_dir,
        "gene_enrichment_fractions.png",
    )
    plt.savefig(fig_path)
    plt.close()
    print(f"  Saved: {fig_path}")


def create_visualizations(
    all_df: pd.DataFrame,
    high_conf_df: pd.DataFrame,
    all_gene_counts: pd.DataFrame,
    high_conf_gene_counts: pd.DataFrame,
):
    """Create comprehensive visualizations for extreme interaction analysis."""
    assert ASSET_IMAGES_DIR is not None, "ASSET_IMAGES_DIR must be set"

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    # Create experiment-specific subdirectory
    output_dir = osp.join(ASSET_IMAGES_DIR, "014-genes-enriched-at-extreme-tmi")
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Interaction score distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # All extreme interactions - score distribution
    ax = axes[0, 0]
    ax.hist(
        all_df["gene_interaction"],
        bins=50,
        color=CUSTOM_BLUE,
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(SCORE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold: ±{SCORE_THRESHOLD}")
    ax.axvline(-SCORE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Triple Mutant Interaction Score $(\tau)$")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Extreme Interactions with Uncharacterized Genes\n" + r"$(|\tau| > " + f"{SCORE_THRESHOLD})$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # High-confidence interactions - score distribution
    ax = axes[0, 1]
    if len(high_conf_df) > 0:
        ax.hist(
            high_conf_df["gene_interaction"],
            bins=50,
            color=CUSTOM_BLUE,
            edgecolor="black",
            alpha=0.7,
        )
        ax.axvline(SCORE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvline(-SCORE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"Triple Mutant Interaction Score $(\tau)$")
    ax.set_ylabel("Frequency")
    ax.set_title(r"High-Confidence $(|\tau| > " + f"{SCORE_THRESHOLD}, p < {P_VALUE_THRESHOLD})$")
    ax.grid(True, alpha=0.3)

    # P-value distribution for all extreme interactions
    ax = axes[1, 0]
    p_values = all_df["p_value"].dropna()
    if len(p_values) > 0:
        ax.hist(p_values, bins=50, color=CUSTOM_BLUE, edgecolor="black", alpha=0.7)
        ax.axvline(P_VALUE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"p = {P_VALUE_THRESHOLD}")
    ax.set_xlabel("P-value")
    ax.set_ylabel("Frequency")
    ax.set_title("P-value Distribution (All Extreme Interactions)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Absolute score distribution comparison
    ax = axes[1, 1]
    ax.hist(
        all_df["abs_score"],
        bins=50,
        alpha=0.5,
        color=CUSTOM_BLUE,
        edgecolor="black",
        label=f"All (n={len(all_df)})",
    )
    if len(high_conf_df) > 0:
        ax.hist(
            high_conf_df["abs_score"],
            bins=50,
            alpha=0.5,
            color=CUSTOM_GREEN,
            edgecolor="black",
            label=f"High-conf (n={len(high_conf_df)})",
        )
    ax.axvline(SCORE_THRESHOLD, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel(r"Absolute Interaction Score $|\tau|$")
    ax.set_ylabel("Frequency")
    ax.set_title("Absolute Score Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = osp.join(
        output_dir,
        "extreme_interaction_distributions.png",
    )
    plt.savefig(fig1_path)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Gene participation counts - All criterion
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of interaction counts (all criterion)
    ax = axes[0]
    ax.hist(
        all_gene_counts["count"],
        bins=50,
        color=CUSTOM_BLUE,
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Number of Extreme Interactions per Uncharacterized Gene")
    ax.set_ylabel("Number of Uncharacterized Genes")
    ax.set_title(f"Uncharacterized Gene Participation in Extreme TMI\n" + r"$(|\tau| > " + f"{SCORE_THRESHOLD})$")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.95,
        0.95,
        f"Total genes: {len(all_gene_counts)}\n"
        f"Mean: {all_gene_counts['count'].mean():.1f}\n"
        f"Median: {all_gene_counts['count'].median():.0f}\n"
        f"Max: {all_gene_counts['count'].max():.0f}",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Top 30 most enriched genes (all criterion)
    ax = axes[1]
    top_30 = all_gene_counts.head(30)
    ax.barh(range(len(top_30)), top_30["count"], color=CUSTOM_BLUE, alpha=0.7)
    ax.set_yticks(range(len(top_30)))
    ax.set_yticklabels(top_30["gene"], fontsize=8)
    ax.set_xlabel("Number of Extreme Interactions")
    ax.set_title(r"Top 30 Genes $(|\tau| > " + f"{SCORE_THRESHOLD})$")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig2_path = osp.join(
        output_dir,
        "gene_enrichment_all_criterion.png",
    )
    plt.savefig(fig2_path)
    plt.close()
    print(f"  Saved: {fig2_path}")

    # Figure 3: Gene participation counts - High-confidence criterion
    if len(high_conf_gene_counts) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram of interaction counts (high-confidence criterion)
        ax = axes[0]
        ax.hist(
            high_conf_gene_counts["count"],
            bins=50,
            color=CUSTOM_BLUE,
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_xlabel("Number of High-Confidence Extreme Interactions per Gene")
        ax.set_ylabel("Number of Genes")
        ax.set_title(r"Gene Participation $(|\tau| > " + f"{SCORE_THRESHOLD}, p < {P_VALUE_THRESHOLD})$")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.95,
            0.95,
            f"Total genes: {len(high_conf_gene_counts)}\n"
            f"Mean: {high_conf_gene_counts['count'].mean():.1f}\n"
            f"Median: {high_conf_gene_counts['count'].median():.0f}\n"
            f"Max: {high_conf_gene_counts['count'].max():.0f}",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Top 30 most enriched genes (high-confidence criterion)
        ax = axes[1]
        top_30 = high_conf_gene_counts.head(30)
        ax.barh(range(len(top_30)), top_30["count"], color=CUSTOM_BLUE, alpha=0.7)
        ax.set_yticks(range(len(top_30)))
        ax.set_yticklabels(top_30["gene"], fontsize=8)
        ax.set_xlabel("Number of High-Confidence Extreme Interactions")
        ax.set_title(r"Top 30 Genes $(|\tau| > " + f"{SCORE_THRESHOLD}, p < {P_VALUE_THRESHOLD})$")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        fig3_path = osp.join(
            output_dir,
            "gene_enrichment_high_conf_criterion.png",
        )
        plt.savefig(fig3_path)
        plt.close()
        print(f"  Saved: {fig3_path}")


def main():
    """Main visualization function."""
    print("=" * 60)
    print("UNCHARACTERIZED GENES IN EXTREME TMI ANALYSIS")
    print("VISUALIZATION STEP")
    print("=" * 60)

    # Load pre-computed results
    all_df, high_conf_df, all_gene_counts, high_conf_gene_counts, all_enrichment, high_conf_enrichment = load_results()

    # Create visualizations
    print("Creating visualizations:")
    create_visualizations(
        all_df,
        high_conf_df,
        all_gene_counts,
        high_conf_gene_counts,
    )

    # Create enrichment fraction visualizations
    print("\nCreating enrichment fraction visualizations:")
    create_enrichment_visualization(all_enrichment, all_df)

    if high_conf_enrichment is not None:
        # Could add high-conf enrichment visualization here if desired
        pass

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
