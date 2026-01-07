# experiments/010-kuzmin-tmi/scripts/12_panel_gene_list_overlap
# [[experiments.010-kuzmin-tmi.scripts.12_panel_gene_list_overlap]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/12_panel_gene_list_overlap
# Test file: experiments/010-kuzmin-tmi/scripts/test_12_panel_gene_list_overlap.py

"""
Visualize overlap between 12-gene panel and various gene lists using UpSet plot.

Gene lists from inference_preprocessing_expansion:
- betaxanthin_genes.txt
- expanded_metabolic_genes.txt
- kemmeren_responsive_genes.txt
- ohya_morphology_genes.txt
- sameith_doubles_genes.txt
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from upsetplot import UpSet, from_memberships

from torchcell.timestamp import timestamp

load_dotenv()
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

assert EXPERIMENT_ROOT is not None
assert ASSET_IMAGES_DIR is not None

# Gene list directory
GENE_LISTS_DIR = osp.join(
    EXPERIMENT_ROOT,
    "006-kuzmin-tmi/results/inference_preprocessing_expansion",
)

# Gene list files to check
GENE_LIST_FILES = {
    "betaxanthin": "betaxanthin_genes.txt",
    "metabolic": "expanded_metabolic_genes.txt",
    "kemmeren": "kemmeren_responsive_genes.txt",
    "ohya": "ohya_morphology_genes.txt",
    "sameith": "sameith_doubles_genes.txt",
}


def load_panel_genes() -> set[str]:
    """Load the 12-gene panel (k=200) from gene_selection_results.csv."""
    results_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/results/gene_selection_results.csv"
    )
    df = pd.read_csv(results_path)
    panel_row = df[(df["panel_size"] == 12) & (df["k"] == 200)].iloc[0]
    genes = set(g.strip() for g in panel_row["selected_genes"].split(","))
    print(f"12-gene panel (k=200): {sorted(genes)}")
    return genes


def load_gene_list(filename: str) -> set[str]:
    """Load a gene list from a txt file."""
    filepath = osp.join(GENE_LISTS_DIR, filename)
    with open(filepath, "r") as f:
        genes = {line.strip() for line in f if line.strip()}
    return genes


def main():
    """Main execution function."""
    print("=" * 60)
    print("12-Gene Panel Overlap with Gene Lists")
    print("=" * 60)

    # Load panel genes
    panel_genes = load_panel_genes()

    # Load all gene lists
    gene_lists = {}
    for name, filename in GENE_LIST_FILES.items():
        gene_lists[name] = load_gene_list(filename)
        print(f"Loaded {name}: {len(gene_lists[name])} genes")

    # Check overlap for each gene in panel
    print("\n" + "=" * 60)
    print("Per-gene membership:")
    print("=" * 60)

    memberships = []
    for gene in sorted(panel_genes):
        gene_memberships = []
        for name, gene_set in gene_lists.items():
            if gene in gene_set:
                gene_memberships.append(name)
        memberships.append(gene_memberships)
        membership_str = ", ".join(gene_memberships) if gene_memberships else "(none)"
        print(f"  {gene}: {membership_str}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, gene_set in gene_lists.items():
        overlap = panel_genes & gene_set
        print(f"  {name}: {len(overlap)}/12 panel genes ({len(overlap)/12*100:.0f}%)")
        if overlap:
            print(f"    → {sorted(overlap)}")

    # Create UpSet plot
    print("\n" + "=" * 60)
    print("Creating UpSet plot...")
    print("=" * 60)

    # Prepare data for upsetplot - include genes and their memberships
    gene_membership_data = []
    for gene in sorted(panel_genes):
        gene_sets = tuple(
            sorted([name for name, gene_set in gene_lists.items() if gene in gene_set])
        )
        gene_membership_data.append(gene_sets)

    # Create membership series
    membership_series = from_memberships(
        gene_membership_data,
        data=list(sorted(panel_genes)),
    )

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    upset = UpSet(
        membership_series,
        subset_size="count",
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="cardinality",
    )
    upset.plot(fig=fig)

    plt.suptitle(
        "12-Gene Panel (k=200) Membership in Gene Lists",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Save plot
    output_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(output_dir, f"12_panel_gene_list_overlap_{timestamp()}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also create a simple bar chart showing overlap counts
    fig2, ax = plt.subplots(figsize=(10, 6))

    names = list(gene_lists.keys())
    overlaps = [len(panel_genes & gene_lists[name]) for name in names]

    bars = ax.barh(names, overlaps, color="#2166ac", edgecolor="black")
    ax.set_xlabel("Number of Panel Genes", fontsize=12)
    ax.set_ylabel("Gene List", fontsize=12)
    ax.set_title("12-Gene Panel Overlap with Gene Lists", fontsize=14)
    ax.set_xlim(0, 12)

    # Add count labels
    for bar, count in zip(bars, overlaps):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{count}/12",
            va="center",
            fontsize=10,
        )

    # Add vertical line at 12
    ax.axvline(x=12, color="red", linestyle="--", alpha=0.5, label="Panel size (12)")

    plt.tight_layout()
    output_path2 = osp.join(output_dir, f"12_panel_gene_list_overlap_bar_{timestamp()}.png")
    fig2.savefig(output_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {output_path2}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()