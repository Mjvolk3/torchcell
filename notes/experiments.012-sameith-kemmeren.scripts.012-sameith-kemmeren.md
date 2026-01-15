---
id: 3bek4vcfhaul6gxgve5rcdg
title: 012 Sameith Kemmeren
desc: ''
updated: 1768502785713
created: 1768498717932
---

## Overview

Master pipeline for Kemmeren & Sameith microarray dataset analysis. Executes all 6 analysis tasks in sequence to verify data quality, generate visualizations, and analyze cross-study correlations. This pipeline processes corrected post-fix data with proper sign conventions.

## Data Sources

### Kemmeren 2014 Dataset

- **Class**: `MicroarrayKemmeren2014Dataset`
- **Publication**: DOI 10.1016/j.cell.2014.10.018
- **Type**: Single deletion mutants + double mutants
- **Organism**: *Saccharomyces cerevisiae*
- **Data**: Log2 expression ratios (deletion vs. wild-type reference pool)

### Sameith 2015 Dataset

- **Class**: `MicroarrayDoubleKOSameith2015Dataset`
- **Publication**: DOI 10.15252/msb.20145174
- **Type**: Double deletion mutants only
- **Organism**: *Saccharomyces cerevisiae*
- **Data**: Log2 expression ratios (double mutant vs. wild-type reference pool)

## Pipeline Tasks

### Task 1: Metadata Verification

**Script**: [[experiments.012-sameith-kemmeren.scripts.verify_metadata]]

Performs QC checks on all three datasets (Kemmeren single, Kemmeren double, Sameith double):

- Validates perturbation counts (1 for single, 2 for double mutants)
- Checks for null/missing expression values
- Detects extreme outliers (>6 std from mean)
- Verifies systematic names match SGD format
- Outputs summary CSV and anomaly report

### Task 2: Single Mutant Expression Distributions

**Script**: [[experiments.012-sameith-kemmeren.scripts.single_mutant_expression_distributions]]

Generates box plots showing log2 expression ratio distributions:

- Kemmeren single mutants: distribution across all genes
- Sameith reference comparison (if available)
- Identifies median, quartiles, and outliers
- Saves stable filenames (no timestamps) for version control

**Outputs**: `single_mutant_kemmeren.png`, `single_mutant_sameith.png`

### Task 3: Double Mutant Combined Heatmap

**Script**: [[experiments.012-sameith-kemmeren.scripts.double_mutant_combined_heatmap]]

Creates triangular heatmap combining both datasets:

- Upper triangle: Kemmeren double mutants
- Lower triangle: Sameith double mutants
- Diagonal: Gene labels
- Color scale: log2 expression ratios
- Reveals systematic differences between datasets

**Output**: `double_mutant_combined_heatmap.png`

### Task 4: Gene-by-Gene Expression Correlation

**Script**: [[experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]]

Analyzes cross-study correlation for genes measured in both datasets:

- Calculates Pearson and Spearman correlations per gene
- Generates distribution histograms
- Identifies genes with high/low cross-study agreement
- Key metric: Median correlation (+0.599 after sign fix)

**Outputs**: `gene_expression_correlation_dist_pearson.png`, `gene_expression_correlation_dist_spearman.png`, `gene_expression_correlations.csv`

### Task 5: Kemmeren-Sameith Overlap Analysis

**Script**: [[experiments.012-sameith-kemmeren.scripts.kemmeren_sameith_overlap_analysis]]

Compares mean expression values for overlapping double mutants:

- Scatter plot: Kemmeren vs. Sameith mean expression
- Distribution comparisons by dataset
- Quantifies agreement/discrepancy
- Identifies systematic biases

**Outputs**: `kemmeren_sameith_overlap_scatter.png`, `kemmeren_sameith_overlap_distributions.png`

### Task 6: Noise Comparison Analysis

**Script**: [[experiments.012-sameith-kemmeren.scripts.noise_comparison_analysis]]

Analyzes technical replicate noise levels:

- Compares within-replicate variance across datasets
- Scatter plot: Kemmeren vs. Sameith noise
- Box plots: noise distributions by dataset
- Histograms: noise value distributions
- Identifies which dataset has cleaner measurements

**Outputs**: `noise_scatter.png`, `noise_boxplot.png`, `noise_histogram.png`, `noise_comparison.csv`

## Usage

```bash
# From torchcell root directory
bash experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren.sh
```

**Requirements**:

- Run from torchcell root (not from scripts/ directory)
- `.env` file with `DATA_ROOT`, `ASSET_IMAGES_DIR`, `EXPERIMENT_ROOT`
- Datasets already downloaded and processed

**Behavior**:

- Fails fast: stops on first error (`set -e`)
- No interactive prompts
- Runs all 6 tasks sequentially
- Each task must succeed for pipeline to continue

## Outputs

### Images

**Location**: `notes/assets/images/012-sameith-kemmeren-expression/`

1. `single_mutant_kemmeren.png` - Box plot of single mutant distributions
2. `single_mutant_sameith.png` - Box plot of Sameith distributions
3. `double_mutant_combined_heatmap.png` - Triangular combined heatmap
4. `gene_expression_correlation_dist_pearson.png` - Pearson correlation distribution
5. `gene_expression_correlation_dist_spearman.png` - Spearman correlation distribution
6. `kemmeren_sameith_overlap_scatter.png` - Mean expression scatter plot
7. `kemmeren_sameith_overlap_distributions.png` - Distribution comparisons
8. `noise_scatter.png` - Cross-dataset noise comparison
9. `noise_boxplot.png` - Noise distribution box plots
10. `noise_histogram.png` - Noise value histograms

### Data Files

**Location**: `experiments/012-sameith-kemmeren/results/`

1. `metadata_verification_summary.csv` - QC statistics
2. `metadata_verification_anomalies.txt` - Flagged issues
3. `gene_expression_correlations.csv` - Per-gene correlation values
4. `noise_comparison.csv` - Replicate noise statistics

All outputs use stable filenames (no timestamps) for reproducibility and version control.

## Key Results

From latest production run (2026-01-06, after sign inversion fix):

- **Median Pearson correlation**: +0.599 (positive, confirming fix)
- **Cross-study agreement**: Moderate positive correlation
- **Sign convention**: Corrected (deletion mutants show positive ratios when upregulated)

## When to Re-run

Re-run this pipeline after:

- Bug fixes to dataset processing code
- Schema changes to data structure
- Sign convention updates
- Changes to core dataset classes (`MicroarrayKemmeren2014Dataset`, `MicroarrayDoubleKOSameith2015Dataset`)

## Implementation Notes

- **Execution order**: Tasks run sequentially; verification catches issues early
- **Fail-fast**: `set -e` stops on first error for easier debugging
- **Stable filenames**: Outputs overwrite each run (no timestamps)
