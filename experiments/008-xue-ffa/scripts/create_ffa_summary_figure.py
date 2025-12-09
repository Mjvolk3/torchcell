# experiments/008-xue-ffa/scripts/create_ffa_summary_figure.py
# [[experiments.008-xue-ffa.scripts.create_ffa_summary_figure]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/create_ffa_summary_figure
# Test file: experiments/008-xue-ffa/scripts/test_create_ffa_summary_figure.py

import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import glob
import matplotlib
matplotlib.use('Agg')

# Load environment
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
RESULTS_DIR = Path("/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/results")


def find_latest_image(pattern):
    """Find the most recent image matching the pattern."""
    files = glob.glob(osp.join(ASSET_IMAGES_DIR, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def create_ffa_summary_figure():
    """
    Create a comprehensive summary figure combining all FFA visualizations.

    Layout:
    - Panel A: FFA bipartite network (metabolic pathway)
    - Panels B-G: Each graph type with TF interaction overlays
    """
    print("=" * 80)
    print("FFA VISUALIZATION SUMMARY FIGURE")
    print("=" * 80)

    # Find the latest images
    print("\nFinding latest visualization images...")

    images = {
        'bipartite': find_latest_image("ffa_bipartite_network_*.png"),
        'physical': find_latest_image("ffa_multigraph_Physical_Interactions_*.png"),
        'regulatory': find_latest_image("ffa_multigraph_Regulatory_Interactions_*.png"),
        'genetic': find_latest_image("ffa_multigraph_Genetic_Interactions_*.png"),
        'tflink': find_latest_image("ffa_multigraph_TFLink_*.png"),
        'coexpression': find_latest_image("ffa_multigraph_STRING_12_0_Coexpression_*.png"),
        'experimental': find_latest_image("ffa_multigraph_STRING_12_0_Experimental_*.png"),
    }

    # Check which images were found
    found_images = {k: v for k, v in images.items() if v is not None}
    missing_images = {k: v for k, v in images.items() if v is None}

    print(f"\nFound {len(found_images)} images:")
    for key, path in found_images.items():
        print(f"  {key}: {os.path.basename(path)}")

    if missing_images:
        print(f"\nMissing {len(missing_images)} images:")
        for key in missing_images:
            print(f"  {key}")

    # Create figure with multiple panels
    # Layout: 2 columns x 4 rows (7 panels + 1 for text/legend)
    fig = plt.figure(figsize=(28, 40))

    panel_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    panel_titles = {
        'bipartite': 'FFA Metabolic Pathway Network',
        'physical': 'Physical Interactions',
        'regulatory': 'Regulatory Interactions',
        'genetic': 'Genetic Interactions',
        'tflink': 'TFLink',
        'coexpression': 'STRING 12.0 Coexpression',
        'experimental': 'STRING 12.0 Experimental',
    }

    # Panel positions (row, col)
    panel_positions = {
        'bipartite': (0, 0),
        'physical': (0, 1),
        'regulatory': (1, 0),
        'genetic': (1, 1),
        'tflink': (2, 0),
        'coexpression': (2, 1),
        'experimental': (3, 0),
    }

    idx = 0
    for key, (row, col) in panel_positions.items():
        if key in found_images:
            # Create subplot
            ax = plt.subplot2grid((4, 2), (row, col))

            # Load and display image
            img = mpimg.imread(found_images[key])
            ax.imshow(img)
            ax.axis('off')

            # Add panel label
            label = panel_labels[idx]
            title = panel_titles[key]
            ax.text(0.02, 0.98, f"{label}. {title}",
                   transform=ax.transAxes,
                   fontsize=20, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            idx += 1

    # Add summary text panel
    ax_text = plt.subplot2grid((4, 2), (3, 1))
    ax_text.axis('off')

    summary_text = """
FFA Metabolic Network Visualization Summary

This figure integrates free fatty acid (FFA) metabolism with
transcription factor (TF) genetic interactions.

Panel A: Core FFA metabolic pathway
  • 17 genes (13 core pathway genes)
  • 64 reactions
  • 148 metabolites (31 target FFAs)
  • Genes → Reactions → Metabolites

Panels B-G: TF epistatic interactions overlaid on metabolism
  • Purple nodes: 10 TF genes from experiment
  • Green edges: Positive epistatic interactions (p<0.05)
  • Grey dashed edges: Baseline network connections
  • Each panel shows a different gene interaction network

Key Findings:
  • Genetic interactions show most enrichment (26 interactions)
  • STRING Experimental shows 21 positive interactions
  • Regulatory and TFLink networks show TF-specific patterns
  • TF interactions influence FFA production pathways

Models: Multiplicative epistatic model, digenic interactions
Graph enrichment: Fisher's exact test, p<0.05 significance
    """

    ax_text.text(0.1, 0.9, summary_text,
                transform=ax_text.transAxes,
                fontsize=14, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9))

    # Main title
    fig.suptitle('Free Fatty Acid Metabolism and Transcription Factor Epistatic Interactions',
                fontsize=24, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    filename = "ffa_summary_figure.png"
    ffa_dir = osp.join(ASSET_IMAGES_DIR, "008-xue-ffa")
    os.makedirs(ffa_dir, exist_ok=True)
    output_path = osp.join(ffa_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"SUMMARY FIGURE SAVED")
    print(f"{'='*80}")
    print(f"\nOutput: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    plt.close()

    # Also create a simplified version with just the main panels (no text)
    fig_simple = plt.figure(figsize=(28, 32))

    idx = 0
    for key, (row, col) in panel_positions.items():
        if key in found_images and key != 'bipartite':  # Skip bipartite for simple version
            # Create subplot (3x2 grid)
            ax = plt.subplot2grid((3, 2), (row - 1 if row > 0 else 0, col))

            # Load and display image
            img = mpimg.imread(found_images[key])
            ax.imshow(img)
            ax.axis('off')

            # Add panel label
            label = panel_labels[idx + 1]  # Start from B
            title = panel_titles[key]
            ax.text(0.02, 0.98, f"{label}. {title}",
                   transform=ax.transAxes,
                   fontsize=20, fontweight='bold',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            idx += 1

    fig_simple.suptitle('TF Epistatic Interactions by Network Type',
                       fontsize=24, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    filename_simple = "ffa_tf_networks_comparison.png"
    output_path_simple = osp.join(ffa_dir, filename_simple)
    plt.savefig(output_path_simple, dpi=300, bbox_inches='tight')
    print(f"\nSimplified comparison: {output_path_simple}")
    print(f"Size: {os.path.getsize(output_path_simple) / 1e6:.1f} MB")
    plt.close()

    print(f"\n{'='*80}")
    print("FFA SUMMARY FIGURE CREATION COMPLETE!")
    print(f"{'='*80}")

    return output_path, output_path_simple


if __name__ == "__main__":
    main_path, comparison_path = create_ffa_summary_figure()
