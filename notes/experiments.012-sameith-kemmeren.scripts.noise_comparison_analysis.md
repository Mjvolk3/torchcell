---
id: ho4sbupdzsyv9mnku5mrx54
title: Noise Comparison Analysis - Experiment 012
desc: Compares technical replicate variability between Kemmeren and Sameith datasets
updated: 1768520433622
created: 1767758104961
---

# Technical Replicate Noise Comparison

## Purpose

Quantifies and compares technical noise between Kemmeren2014 and Sameith2015 datasets by analyzing standard deviation of log2 expression changes across technical replicates.

**Key Question**: Which dataset has lower technical variability (higher precision)?

## Implementation

**Script**: `experiments/012-sameith-kemmeren/scripts/noise_comparison_analysis.py`

## Methodology

Both datasets now use `expression_log2_ratio_std` for fair comparison:

- **Kemmeren2014**: 4 technical replicates (2 cultures × 2 arrays)
- **Sameith2015**: 2 technical replicates (dye-swap pairs per deletion)

Compares:

1. Mean std across all genes for each deletion
2. Distribution of std values
3. Per-gene noise levels

## Outputs

### Images

#TODO ![Noise Scatter Plot](./assets/images/012-sameith-kemmeren-expression/noise_scatter.png)

*Figure 1: Scatter plot comparing median technical standard deviation (log2_ratio_std) between Kemmeren2014 and Sameith2015 for 82 overlapping gene deletions. Each point represents one gene deletion. Shows whether noise is gene-intrinsic (high correlation) or study-specific (low correlation).*

#TODO ![Noise Distribution Box Plot](./assets/images/012-sameith-kemmeren-expression/noise_boxplot.png)

*Figure 2: Box plot comparison of technical noise distributions using log2_ratio_std for both datasets. Shows median std and IQR for each dataset with Wilcoxon signed-rank test p-value.*

#TODO ![Noise Distribution Histogram](./assets/images/012-sameith-kemmeren-expression/noise_histogram.png)

*Figure 3: Histogram comparison of technical noise distributions. Shows overlapping histograms with median lines for visual comparison of the standard deviation distributions between studies.*

### Data Files

- `results/noise_comparison.csv` - Per-gene technical noise comparison with statistical metrics

## Key Findings

Kemmeren2014 generally shows **lower technical variability** than Sameith2015, likely due to:

- More technical replicates (4 vs 2)
- More mature platform/protocol

This justifies using Kemmeren as the primary expression dataset for model training.

## Usage

```bash
python experiments/012-sameith-kemmeren/scripts/noise_comparison_analysis.py
# Or: bash experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren.sh
```

## Related Notes

- [[experiments.012-sameith-kemmeren.scripts.verify_metadata]] - QC performed first
- [[experiments.012-sameith-kemmeren.scripts.012-sameith-kemmeren]] - Main pipeline
