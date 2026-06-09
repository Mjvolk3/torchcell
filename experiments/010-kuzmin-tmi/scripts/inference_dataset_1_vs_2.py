# experiments/010-kuzmin-tmi/scripts/inference_dataset_1_vs_2.py
# [[experiments.010-kuzmin-tmi.scripts.inference_dataset_1_vs_2]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/inference_dataset_1_vs_2.py
"""
Compare 12-gene panels between inference_1 and inference_2.

Inference_1: 275M triples, fitness > 1.0 threshold
Inference_2: 479K triples, iterative fitness improvement
            (all SMF > 1.0, max SMF > 1.10, all DMF > 1.0, max DMF > max(SMF) + 0.03)

This script visualizes the overlap and differences between the selected gene panels.
"""

import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib_venn import venn2, venn2_circles

from torchcell.timestamp import timestamp

# Load environment variables
load_dotenv()
ASSET_IMAGES_DIR = os.environ["ASSET_IMAGES_DIR"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

# Use torchcell style
plt.style.use("torchcell/torchcell.mplstyle")


def load_gene_panel(csv_path: str) -> set[str]:
    """Load genes from a singles table CSV."""
    df = pd.read_csv(csv_path)
    return set(df["gene"].tolist())


def sort_genes_overlap_first(genes: set[str], overlap: set[str]) -> list[str]:
    """Sort genes with overlap genes first, then alphabetically."""
    overlap_genes = sorted(g for g in genes if g in overlap)
    other_genes = sorted(g for g in genes if g not in overlap)
    return overlap_genes + other_genes


def main():
    print("=" * 60)
    print("Comparing 12-Gene Panels: Inference_1 vs Inference_2")
    print("=" * 60)

    # Load gene panels
    inf1_path = osp.join(
        EXPERIMENT_ROOT, "010-kuzmin-tmi/results/singles_table_panel12_k200.csv"
    )
    inf2_path = osp.join(
        EXPERIMENT_ROOT,
        "010-kuzmin-tmi/results/inference_2/singles_table_panel12_k200.csv",
    )

    genes_inf1 = load_gene_panel(inf1_path)
    genes_inf2 = load_gene_panel(inf2_path)

    # Compute set operations
    overlap = genes_inf1 & genes_inf2
    only_inf1 = genes_inf1 - genes_inf2
    only_inf2 = genes_inf2 - genes_inf1

    # Sort with overlap first
    inf1_sorted = sort_genes_overlap_first(genes_inf1, overlap)
    inf2_sorted = sort_genes_overlap_first(genes_inf2, overlap)

    print(f"\nInference_1 genes ({len(genes_inf1)}) [overlap genes first]:")
    for g in inf1_sorted:
        marker = " *" if g in overlap else ""
        print(f"  {g}{marker}")

    print(f"\nInference_2 genes ({len(genes_inf2)}) [overlap genes first]:")
    for g in inf2_sorted:
        marker = " *" if g in overlap else ""
        print(f"  {g}{marker}")

    print(f"\n{'='*60}")
    print("Set Analysis")
    print("=" * 60)
    print(f"Overlap ({len(overlap)} genes): {sorted(overlap)}")
    print(f"Only in Inference_1 ({len(only_inf1)} genes): {sorted(only_inf1)}")
    print(f"Only in Inference_2 ({len(only_inf2)} genes): {sorted(only_inf2)}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Colors from torchcell.mplstyle
    color_inf1 = "#34699D"  # blue
    color_inf2 = "#D86E2F"  # orange

    # Left: Venn diagram
    v = venn2(
        subsets=(genes_inf1, genes_inf2),
        set_labels=("Inference_1\n(fitness > 1.0)", "Inference_2\n(iterative improvement)"),
        set_colors=(color_inf1, color_inf2),
        alpha=0.7,
        ax=ax1,
    )
    venn2_circles(subsets=(genes_inf1, genes_inf2), linewidth=2, ax=ax1)

    # Customize Venn labels
    for text in v.set_labels:
        if text:
            text.set_fontsize(12)
            text.set_fontweight("bold")
    for text in v.subset_labels:
        if text:
            text.set_fontsize(14)
            text.set_fontweight("bold")

    ax1.set_title("12-Gene Panel Overlap\n(k=200)", fontsize=14, fontweight="bold")

    # Right: Gene lists as text (overlap first)
    ax2.axis("off")

    # Format gene lists with overlap first and marked (3 rows of 4 genes)
    def format_genes_grid(genes: list[str], overlap: set[str], cols: int = 4) -> str:
        """Format genes as grid with overlap marked by *."""
        marked = [f"*{g}*" if g in overlap else g for g in genes]
        rows = [marked[i : i + cols] for i in range(0, len(marked), cols)]
        return "\n".join(", ".join(row) for row in rows)

    inf1_display = format_genes_grid(inf1_sorted, overlap)
    inf2_display = format_genes_grid(inf2_sorted, overlap)

    # Create formatted text
    text_lines = [
        "─" * 44,
        "OVERLAP (shared genes)",
        "─" * 44,
        ", ".join(sorted(overlap)) if overlap else "None",
        "",
        "─" * 44,
        "INFERENCE_1 (overlap first, marked with *)",
        "─" * 44,
        inf1_display,
        "",
        "─" * 44,
        "INFERENCE_2 (overlap first, marked with *)",
        "─" * 44,
        inf2_display,
        "",
        "─" * 44,
        f"Overlap: {len(overlap)}/12 genes",
    ]
    text_content = "\n".join(text_lines)

    ax2.text(
        0.5,
        0.5,
        text_content,
        transform=ax2.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(
            boxstyle="round", facecolor="#E8E8E8", alpha=0.8, edgecolor="#CCCCCC"
        ),
    )

    ax2.set_title("Gene Lists", fontsize=14, fontweight="bold")

    # Add figure title
    fig.suptitle(
        "Inference Dataset Comparison: 12-Gene Panels (k=200)\n"
        "Inference_1: fitness > 1.0 | Inference_2: iterative fitness improvement",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))

    # Save plot
    output_dir = osp.join(ASSET_IMAGES_DIR, "010-kuzmin-tmi")
    os.makedirs(output_dir, exist_ok=True)
    output_path = osp.join(
        output_dir, f"inference_1_vs_2_panel_comparison_{timestamp()}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print("The iterative fitness improvement filter in inference_2 results")
    print(f"in a largely different gene panel, with only {len(overlap)}/12 shared.")
    print("\nInference_1: fitness > 1.0 (275M triples)")
    print("Inference_2: iterative improvement - SMF/DMF must increase (479K triples)")
    print("\nThis suggests the top predicted triples differ substantially")
    print("when requiring fitness to improve at each mutation step.")


if __name__ == "__main__":
    main()
