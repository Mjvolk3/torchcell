# experiments/012-sameith-kemmeren/scripts/kemmeren_volcano.py
# [[experiments.012-sameith-kemmeren.scripts.kemmeren_volcano]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/012-sameith-kemmeren/scripts/kemmeren_volcano

import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dotenv import load_dotenv
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset

# Load environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

# Create output directory for this experiment
exp_image_dir = osp.join(ASSET_IMAGES_DIR, "012-sameith-kemmeren")
os.makedirs(exp_image_dir, exist_ok=True)

print("Loading Kemmeren2014 dataset...")
dataset = MicroarrayKemmeren2014Dataset(
    root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
    io_workers=10,
    process_workers=10
)
print(f"Dataset loaded: {len(dataset)} gene deletions")

# Collect data from all gene deletions
print("\nExtracting expression data for volcano plot...")
volcano_data = []

for idx in range(len(dataset)):
    data = dataset[idx]
    experiment = data["experiment"]
    phenotype = experiment["phenotype"]

    # Extract gene name
    systematic_gene = experiment["genotype"]["perturbations"][0]["systematic_gene_name"]

    # Get expression data
    log2_ratios = phenotype["expression_log2_ratio"]
    log2_se = phenotype["expression_log2_ratio_se"]
    n_samples = phenotype["n_samples"]

    # For each measured gene in this deletion strain
    for gene, log2_fc in log2_ratios.items():
        se = log2_se.get(gene, np.nan)
        n = n_samples.get(gene, 1)

        # Compute t-statistic and p-value (two-tailed test: is log2_fc significantly different from 0?)
        if not np.isnan(se) and se > 0 and n > 1:
            t_stat = log2_fc / se
            # Two-tailed t-test with n-1 degrees of freedom
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        else:
            # No p-value when n=1 or SE is invalid
            p_value = np.nan

        volcano_data.append({
            "deletion_strain": systematic_gene,
            "measured_gene": gene,
            "log2_fold_change": log2_fc,
            "se": se,
            "n_samples": n,
            "p_value": p_value
        })

# Convert to DataFrame
df = pd.DataFrame(volcano_data)
print(f"\nTotal data points: {len(df)}")
print(f"Data points with valid p-values: {(~df['p_value'].isna()).sum()}")

# Filter to only include genes with valid p-values for plotting
df_valid = df[~df['p_value'].isna()].copy()
df_valid["-log10_p"] = -np.log10(df_valid["p_value"])

# Thresholds for significance
LOG2FC_THRESHOLD = 1.0  # 2-fold change
PVALUE_THRESHOLD = 0.01  # p < 0.01
NEG_LOG10_P_THRESHOLD = -np.log10(PVALUE_THRESHOLD)

# Classify points
def classify_point(row):
    if abs(row["log2_fold_change"]) < LOG2FC_THRESHOLD or row["p_value"] > PVALUE_THRESHOLD:
        return "Not significant"
    elif row["log2_fold_change"] > LOG2FC_THRESHOLD:
        return "Upregulated"
    else:
        return "Downregulated"

df_valid["classification"] = df_valid.apply(classify_point, axis=1)

# Count classifications
print("\nClassification summary:")
print(df_valid["classification"].value_counts())

# Create volcano plot
fig, ax = plt.subplots(figsize=(12, 10))

# Color scheme
colors = {
    "Not significant": "#CCCCCC",
    "Upregulated": "#4472C4",  # Blue
    "Downregulated": "#C55A5A"  # Red
}

# Plot each category
for category, color in colors.items():
    mask = df_valid["classification"] == category
    ax.scatter(
        df_valid.loc[mask, "log2_fold_change"],
        df_valid.loc[mask, "-log10_p"],
        c=color,
        alpha=0.6,
        s=10,
        label=category,
        edgecolors="none"
    )

# Add threshold lines
ax.axhline(y=NEG_LOG10_P_THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(x=LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(x=-LOG2FC_THRESHOLD, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.axvline(x=0, color="black", linewidth=1.5, alpha=0.7)

# Labels and annotations
ax.set_xlabel("Fold Change (log₂)", fontsize=14, fontweight="bold")
ax.set_ylabel("-log₁₀ p-value", fontsize=14, fontweight="bold")
ax.set_title("Kemmeren2014 Expression Volcano Plot\n(All Gene Deletions, All Measured Genes)",
             fontsize=16, fontweight="bold", pad=20)

# Add text annotations - positioned in center of each pane
# Red text: center of left pane (x from -6 to 0, so center = -3)
ax.text(-3, 10, "Negative Change in\ngene expression\ncompared to control",
        ha="center", va="center", fontsize=11, color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))

# Blue text: center of right pane (x from 0 to 6, so center = 3)
ax.text(3, 10, "Positive Change in\ngene expression\ncompared to control",
        ha="center", va="center", fontsize=11, color="#00008B",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))

# Threshold label - one line, just inside y-axis at threshold height
ax.text(-5.7, NEG_LOG10_P_THRESHOLD, "Threshold (p=0.01)",
        ha="right", va="center", fontsize=9, color="gray",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"))

# Not significant label - centered at x=0, just above x-axis
ax.text(0, 0.3, "Not significant (p>0.01 or |log2FC|<1)",
        ha="center", va="bottom", fontsize=9, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray", linewidth=0.5))

# Legend
ax.legend(loc="upper right", frameon=True, fontsize=11)

# Set axis limits
ax.set_xlim(-6, 6)
ax.set_ylim(-0.5, 20)

# Grid
ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
ax.set_axisbelow(True)

# Tight layout
plt.tight_layout()

# Save figure
output_path = osp.join(exp_image_dir, "kemmeren_volcano.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)  # Close figure to free memory
print(f"\nVolcano plot saved to: {output_path}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total gene deletion strains: {df['deletion_strain'].nunique()}")
print(f"Total unique genes measured: {df['measured_gene'].nunique()}")
print(f"Total data points: {len(df)}")
print(f"Valid p-values: {len(df_valid)}")
print(f"\nSignificant upregulated: {(df_valid['classification'] == 'Upregulated').sum()}")
print(f"Significant downregulated: {(df_valid['classification'] == 'Downregulated').sum()}")
print(f"Not significant: {(df_valid['classification'] == 'Not significant').sum()}")
print(f"\nThresholds used:")
print(f"  |log2FC| > {LOG2FC_THRESHOLD}")
print(f"  p-value < {PVALUE_THRESHOLD}")
