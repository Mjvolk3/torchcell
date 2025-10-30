# experiments/008-xue-ffa/scripts/multiplicative_vs_additive_comparison.py
# [[experiments.008-xue-ffa.scripts.multiplicative_vs_additive_comparison]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/008-xue-ffa/scripts/multiplicative_vs_additive_comparison
# Test file: experiments/008-xue-ffa/scripts/test_multiplicative_vs_additive_comparison.py

import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from dotenv import load_dotenv
from torchcell.timestamp import timestamp
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")
if not ASSET_IMAGES_DIR:
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    ASSET_IMAGES_DIR = osp.join(PROJECT_ROOT, "notes/assets/images")
os.makedirs(ASSET_IMAGES_DIR, exist_ok=True)

# Apply torchcell style
STYLE_PATH = (
    "/Users/michaelvolk/Documents/projects/torchcell/torchcell/torchcell.mplstyle"
)
if osp.exists(STYLE_PATH):
    plt.style.use(STYLE_PATH)

# Import functions from both model scripts
import sys
sys.path.append(
    "/Users/michaelvolk/Documents/projects/torchcell/experiments/008-xue-ffa/scripts"
)
from free_fatty_acid_interactions import (
    load_ffa_data,
    normalize_by_reference,
    compute_interactions_with_error_propagation as compute_multiplicative,
)
from additive_free_fatty_acid_interactions import (
    compute_additive_interactions_with_error_propagation as compute_additive,
)


def create_model_comparison_heatmaps(mult_data=None, add_data=None):
    """Create heatmaps showing expected f_ij values under each model (no interaction).

    This shows what each model expects when there's NO genetic interaction,
    helping visualize the fundamental difference between additive and multiplicative models.
    """

    # Create a grid of f_i and f_j values
    f_values = np.linspace(0.2, 2.0, 50)
    f_i_grid, f_j_grid = np.meshgrid(f_values, f_values)

    # Calculate expected double mutant fitness when NO interaction (δ=0 or ε=0)
    # Multiplicative model expects: f_ij = f_i * f_j
    f_ij_mult_expected = f_i_grid * f_j_grid

    # Additive model (CORRECTED - standard genetic null): f_ij = f_i + f_j - 1
    # This is the inclusion-exclusion principle for additive effects
    f_ij_add_expected = f_i_grid + f_j_grid - 1

    # Difference between expected values
    difference = f_ij_add_expected - f_ij_mult_expected

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Common colormap settings for f_ij values - use same range for both
    vmin = min(f_ij_mult_expected.min(), f_ij_add_expected.min())
    vmax = max(f_ij_mult_expected.max(), f_ij_add_expected.max())
    # Create common norm for consistent color scaling
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot 1: Additive model expected f_ij
    ax = axes[0]
    im1 = ax.contourf(f_i_grid, f_j_grid, f_ij_add_expected, levels=20,
                      cmap='viridis', norm=norm)
    # Add contour lines at specific values
    cs1 = ax.contour(f_i_grid, f_j_grid, f_ij_add_expected,
                     levels=[0.5, 1.0, 1.5], colors='white', linewidths=1.5)
    ax.clabel(cs1, inline=True, fontsize=8)

    ax.set_xlabel(r'Single Mutant Fitness ($f_i$)', fontsize=11)
    ax.set_ylabel(r'Single Mutant Fitness ($f_j$)', fontsize=11)
    ax.set_title('Additive Model (No Interaction)\n' +
                 r'Expected: $f_{ij} = f_i + f_j - 1$', fontsize=13)
    cbar1 = plt.colorbar(im1, ax=ax)
    cbar1.set_label(r'Expected $f_{ij}$', fontsize=10)
    cbar1.mappable.set_clim(vmin, vmax)  # Ensure colorbar shows full range

    # Plot 2: Multiplicative model expected f_ij
    ax = axes[1]
    im2 = ax.contourf(f_i_grid, f_j_grid, f_ij_mult_expected, levels=20,
                      cmap='viridis', norm=norm)
    # Add contour lines at specific values
    cs2 = ax.contour(f_i_grid, f_j_grid, f_ij_mult_expected,
                     levels=[0.5, 1.0, 1.5], colors='white', linewidths=1.5)
    ax.clabel(cs2, inline=True, fontsize=8)

    ax.set_xlabel(r'Single Mutant Fitness ($f_i$)', fontsize=11)
    ax.set_ylabel(r'Single Mutant Fitness ($f_j$)', fontsize=11)
    ax.set_title('Multiplicative Model (No Interaction)\n' +
                 r'Expected: $f_{ij} = f_i \times f_j$', fontsize=13)
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label(r'Expected $f_{ij}$', fontsize=10)
    cbar2.mappable.set_clim(vmin, vmax)  # Ensure colorbar shows full range

    # Plot 3: Difference between expected values
    ax = axes[2]
    diff_norm = TwoSlopeNorm(vmin=difference.min(), vcenter=0,
                             vmax=difference.max())
    im3 = ax.contourf(f_i_grid, f_j_grid, difference, levels=20,
                      cmap='RdBu_r', norm=diff_norm)
    ax.contour(f_i_grid, f_j_grid, difference, levels=[0],
               colors='black', linewidths=2)
    ax.set_xlabel(r'Single Mutant Fitness ($f_i$)', fontsize=11)
    ax.set_ylabel(r'Single Mutant Fitness ($f_j$)', fontsize=11)
    ax.set_title('Difference in Expected Values\n' +
                 r'Additive $-$ Multiplicative', fontsize=13)
    cbar3 = plt.colorbar(im3, ax=ax)
    cbar3.set_label(r'$f_{ij}^{add} - f_{ij}^{mult}$', fontsize=10)

    # Plot 4: Cross-section at f_i = f_j (diagonal) showing expected f_ij
    ax = axes[3]
    # Extend the range to include 0
    f_diagonal_extended = np.linspace(0, 2.0, 100)
    # Calculate expected values for the extended range
    f_ij_mult_diag_extended = f_diagonal_extended * f_diagonal_extended  # f_i * f_i
    f_ij_add_diag_extended = 2 * f_diagonal_extended - 1  # f_i + f_i - 1 = 2*f_i - 1

    ax.plot(f_diagonal_extended, f_ij_mult_diag_extended, 'b-', linewidth=2.5,
            label='Multiplicative')
    ax.plot(f_diagonal_extended, f_ij_add_diag_extended, 'r-', linewidth=2.5,
            label='Additive')

    # Set same range for both axes
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    ax.set_xlabel(r'Single Mutant Fitness ($f_i = f_j$)', fontsize=11)
    ax.set_ylabel(r'Expected $f_{ij}$', fontsize=11)
    ax.set_title('Expected Double Mutant Fitness\n(symmetric case: same single mutants)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Highlight key regions - flipped colors to match difference plot
    ax.axvspan(0, 1.0, alpha=0.1, color='red')  # Red for fitness < 1
    ax.axvspan(1.0, 2.0, alpha=0.1, color='blue')  # Blue for fitness > 1
    # Add vertical line at f=1 for reference
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
    # Add diagonal reference line (f_ij = f_i)
    ax.plot([0, 2], [0, 2], 'k--', linewidth=1, alpha=0.3)

    plt.suptitle('Expected Double Mutant Fitness: Multiplicative vs Additive Models\n' +
                 '(When genetic interaction = 0)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = f"model_comparison_heatmaps_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print("Model comparison heatmaps saved to:")
    print(f"  {filepath}")


def compare_significant_interactions(mult_data, add_data, columns, p_threshold=0.05):
    """Compare the number of significant interactions between models."""

    # Extract data from both models
    (mult_digenic_int, mult_digenic_sd, mult_digenic_se, mult_digenic_pval,
     mult_trigenic_int, mult_trigenic_sd, mult_trigenic_se, mult_trigenic_pval,
     _, _, _, _, _, _, _, _, _) = mult_data

    (add_digenic_int, add_digenic_sd, add_digenic_se, add_digenic_pval,
     add_trigenic_int, add_trigenic_sd, add_trigenic_se, add_trigenic_pval,
     _, _, _, _, _, _, _, _, _) = add_data

    # Count significant interactions for each FFA
    ffa_types = columns
    comparison_data = []

    for ffa_idx, ffa in enumerate(ffa_types):
        # Count significant digenic interactions
        mult_dig_sig = 0
        add_dig_sig = 0

        for key, pvals in mult_digenic_pval.items():
            if ffa_idx < len(pvals) and not np.isnan(pvals[ffa_idx]):
                if pvals[ffa_idx] < p_threshold:
                    mult_dig_sig += 1

        for key, pvals in add_digenic_pval.items():
            if ffa_idx < len(pvals) and not np.isnan(pvals[ffa_idx]):
                if pvals[ffa_idx] < p_threshold:
                    add_dig_sig += 1

        # Count significant trigenic interactions
        mult_tri_sig = 0
        add_tri_sig = 0

        for key, pvals in mult_trigenic_pval.items():
            if ffa_idx < len(pvals) and not np.isnan(pvals[ffa_idx]):
                if pvals[ffa_idx] < p_threshold:
                    mult_tri_sig += 1

        for key, pvals in add_trigenic_pval.items():
            if ffa_idx < len(pvals) and not np.isnan(pvals[ffa_idx]):
                if pvals[ffa_idx] < p_threshold:
                    add_tri_sig += 1

        comparison_data.append({
            'FFA': ffa,
            'Mult_Digenic': mult_dig_sig,
            'Add_Digenic': add_dig_sig,
            'Mult_Trigenic': mult_tri_sig,
            'Add_Trigenic': add_tri_sig
        })

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = pd.DataFrame(comparison_data)
    x = np.arange(len(ffa_types))
    width = 0.35

    # Plot 1: Digenic interactions comparison
    ax = axes[0, 0]
    ax.bar(x - width/2, df['Mult_Digenic'], width, label='Multiplicative',
           color='#34699D', alpha=0.8)
    ax.bar(x + width/2, df['Add_Digenic'], width, label='Additive',
           color='#CC8250', alpha=0.8)
    ax.set_xlabel('FFA Type', fontsize=12)
    ax.set_ylabel('Number of Significant Interactions', fontsize=12)
    ax.set_title(f'Significant Digenic Interactions (p < {p_threshold})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Trigenic interactions comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, df['Mult_Trigenic'], width, label='Multiplicative',
           color='#34699D', alpha=0.8)
    ax.bar(x + width/2, df['Add_Trigenic'], width, label='Additive',
           color='#CC8250', alpha=0.8)
    ax.set_xlabel('FFA Type', fontsize=12)
    ax.set_ylabel('Number of Significant Interactions', fontsize=12)
    ax.set_title(f'Significant Trigenic Interactions (p < {p_threshold})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Ratio of significant interactions (Additive/Multiplicative)
    ax = axes[1, 0]
    # Avoid division by zero
    digenic_ratio = df['Add_Digenic'] / (df['Mult_Digenic'] + 1e-10)
    trigenic_ratio = df['Add_Trigenic'] / (df['Mult_Trigenic'] + 1e-10)

    ax.bar(x - width/2, digenic_ratio, width, label='Digenic',
           color='#7191A9', alpha=0.8)  # Light blue for digenic
    ax.bar(x + width/2, trigenic_ratio, width, label='Trigenic',
           color='#B73C39', alpha=0.8)  # Red for trigenic
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('FFA Type', fontsize=12)
    ax.set_ylabel('Ratio (Additive / Multiplicative)', fontsize=12)
    ax.set_title('Ratio of Significant Interactions', fontsize=14,
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ffa_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Summary statistics
    ax = axes[1, 1]

    # Calculate totals
    total_mult_dig = df['Mult_Digenic'].sum()
    total_add_dig = df['Add_Digenic'].sum()
    total_mult_tri = df['Mult_Trigenic'].sum()
    total_add_tri = df['Add_Trigenic'].sum()

    # Create summary bar plot
    categories = ['Digenic\n(All FFAs)', 'Trigenic\n(All FFAs)']
    mult_values = [total_mult_dig, total_mult_tri]
    add_values = [total_add_dig, total_add_tri]

    x_summary = np.arange(len(categories))
    ax.bar(x_summary - width/2, mult_values, width, label='Multiplicative',
           color='#34699D', alpha=0.8)
    ax.bar(x_summary + width/2, add_values, width, label='Additive',
           color='#CC8250', alpha=0.8)

    # Expand y-axis to prevent text overlap
    y_max = max(max(mult_values), max(add_values))
    ax.set_ylim(0, y_max * 1.15)

    # Add value labels on bars
    for i, v in enumerate(mult_values):
        ax.text(i - width/2, v + 5, str(int(v)), ha='center', fontsize=10)
    for i, v in enumerate(add_values):
        ax.text(i + width/2, v + 5, str(int(v)), ha='center', fontsize=10)

    ax.set_xlabel('Interaction Type', fontsize=12)
    ax.set_ylabel('Total Significant Interactions', fontsize=12)
    ax.set_title('Overall Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x_summary)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Comparison of Significant Interactions: ' +
                 'Multiplicative vs Additive Models',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = f"significant_interactions_comparison_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print("Significant interactions comparison saved to:")
    print(f"  {filepath}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Significant Digenic - Multiplicative: {total_mult_dig}")
    print(f"Total Significant Digenic - Additive: {total_add_dig}")
    print(f"Total Significant Trigenic - Multiplicative: {total_mult_tri}")
    print(f"Total Significant Trigenic - Additive: {total_add_tri}")
    print("\nRatio (Additive/Multiplicative):")
    print(f"  Digenic: {total_add_dig/max(total_mult_dig, 1):.2f}")
    print(f"  Trigenic: {total_add_tri/max(total_mult_tri, 1):.2f}")


def create_interaction_distribution_comparison(mult_data, add_data, columns):
    """Compare the distributions of interaction scores between models."""

    # Extract interactions
    mult_digenic_int = mult_data[0]
    mult_trigenic_int = mult_data[4]
    add_digenic_int = add_data[0]
    add_trigenic_int = add_data[4]

    ffa_types = columns

    # Create figure with subplots for each FFA
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ffa_idx, ffa in enumerate(ffa_types):
        ax = axes[ffa_idx]

        # Collect interaction values
        mult_dig_values = []
        add_dig_values = []
        mult_tri_values = []
        add_tri_values = []

        for interactions in mult_digenic_int.values():
            if ffa_idx < len(interactions) and not np.isnan(interactions[ffa_idx]):
                mult_dig_values.append(interactions[ffa_idx])

        for interactions in add_digenic_int.values():
            if ffa_idx < len(interactions) and not np.isnan(interactions[ffa_idx]):
                add_dig_values.append(interactions[ffa_idx])

        for interactions in mult_trigenic_int.values():
            if ffa_idx < len(interactions) and not np.isnan(interactions[ffa_idx]):
                mult_tri_values.append(interactions[ffa_idx])

        for interactions in add_trigenic_int.values():
            if ffa_idx < len(interactions) and not np.isnan(interactions[ffa_idx]):
                add_tri_values.append(interactions[ffa_idx])

        # Create violin plots
        positions = [1, 2, 4, 5]
        parts = ax.violinplot([mult_dig_values, add_dig_values,
                              mult_tri_values, add_tri_values],
                              positions=positions, widths=0.8,
                              showmeans=False, showmedians=False)

        # Customize colors for violin plots - distinguish by model type
        # Blue for Multiplicative, Orange/Brown for Additive
        colors = ['#34699D', '#CC8250', '#34699D', '#CC8250']  # Blue for mult, orange for add
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.4)  # Semi-transparent for violin

        # Add separate box plots on top with clear distinction
        bp1 = ax.boxplot([mult_dig_values], positions=[1], widths=0.4,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2),
                         boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5))
        bp2 = ax.boxplot([add_dig_values], positions=[2], widths=0.4,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2),
                         boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5))
        bp3 = ax.boxplot([mult_tri_values], positions=[4], widths=0.4,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2),
                         boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5))
        bp4 = ax.boxplot([add_tri_values], positions=[5], widths=0.4,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='black', linewidth=2),
                         boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.5))

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels([r'Mult' + '\n' + r'Digenic ($\varepsilon$)',
                            r'Add' + '\n' + r'Digenic ($\epsilon$)',
                            r'Mult' + '\n' + r'Trigenic ($\tau$)',
                            r'Add' + '\n' + r'Trigenic ($E$)'], fontsize=9)
        ax.set_ylabel('Interaction Score', fontsize=12)
        ax.set_title(f'{ffa}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add a subtle legend for the mean line
        if ffa_idx == 0:  # Only add legend to first subplot
            ax.plot([], [], color='red', linewidth=2, label='Mean')
            ax.legend(loc='upper left', fontsize=8)

        # Add mean values as text annotations inside the plot
        means = [np.mean(mult_dig_values), np.mean(add_dig_values),
                 np.mean(mult_tri_values), np.mean(add_tri_values)]

        # Expand y-axis limits to give more space
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)

        # Add mean markers as horizontal lines
        for i, (pos, mean_val) in enumerate(zip(positions, means)):
            # Add a small horizontal line for the mean
            ax.plot([pos - 0.2, pos + 0.2], [mean_val, mean_val],
                   color='red', linewidth=2, zorder=10)

            # Add text label for mean value inside the plot area at top
            y_max = ax.get_ylim()[1]
            y_min = ax.get_ylim()[0]
            text_y = y_max - (y_max - y_min) * 0.12  # 12% from top (moved down more)
            ax.text(pos, text_y, f'$\mu$={mean_val:.3f}',
                    ha='center', fontsize=8, color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle('Distribution of Interaction Scores: ' +
                 'Multiplicative vs Additive Models',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    filename = f"interaction_distribution_comparison_{timestamp()}.png"
    filepath = osp.join(ASSET_IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print("Interaction distribution comparison saved to:")
    print(f"  {filepath}")


def main():
    """Main function for model comparison analysis."""

    print("Loading FFA data...")
    file_path = ("/Users/michaelvolk/Documents/projects/torchcell/data/"
                 "torchcell/ffa_xue2025/raw/Supplementary Data 1_Raw titers.xlsx")
    raw_df, abbreviations, replicate_dict = load_ffa_data(file_path)

    print("Normalizing by positive control (+ve Ctrl)...")
    normalized_df, normalized_replicates = normalize_by_reference(
        raw_df, replicate_dict)

    print("Computing multiplicative model interactions...")
    mult_data = compute_multiplicative(
        normalized_df, normalized_replicates, abbreviations)

    print("Computing additive model interactions...")
    add_data = compute_additive(normalized_df, normalized_replicates, abbreviations)

    # Get column names (FFA types)
    columns = list(raw_df.columns[1:])

    print("\n" + "="*60)
    print("Creating model comparison visualizations...")
    print("="*60)

    # Create theoretical model comparison heatmaps with actual data ranges
    print("\n1. Creating model comparison heatmaps...")
    create_model_comparison_heatmaps(mult_data, add_data)

    # Compare significant interactions
    print("\n2. Comparing significant interactions...")
    compare_significant_interactions(mult_data, add_data, columns)

    # Compare interaction distributions
    print("\n3. Comparing interaction distributions...")
    create_interaction_distribution_comparison(mult_data, add_data, columns)

    print("\n" + "="*60)
    print("All model comparison analyses complete!")
    print("Check the file paths above to view the plots.")
    print("="*60)


if __name__ == "__main__":
    main()
