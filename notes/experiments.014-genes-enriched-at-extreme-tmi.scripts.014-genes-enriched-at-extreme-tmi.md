---
id: master_script_014
title: 014-genes-enriched-at-extreme-tmi
desc: ''
updated: 1768418379017
created: 1737825600000
---

## Overview

Master orchestration script that runs the complete experiment 014 pipeline: data processing followed by visualization.

## Purpose

Provides a single command to reproduce the entire analysis from scratch. Ensures steps run in correct order with proper dependencies.

## Pipeline Steps

### Step 1: Data Processing (Slow)

**Script:** `analyze_extreme_interactions.py`

**What it does:**

- Loads 668 uncharacterized genes from experiment 013
- Loads 376,732 TMI interactions from experiment 010 dataset
- Filters for extreme interactions (|τ| > 0.1)
- Counts gene participation in extreme interactions
- Calculates enrichment fractions for each gene
- Saves 7 CSV/TXT result files

**Runtime:** ~5-10 minutes (dataset loading is slow)

**Output location:** `experiments/014-genes-enriched-at-extreme-tmi/results/`

### Step 2: Visualization (Fast)

**Script:** `visualize_extreme_interactions.py`

**What it does:**

- Loads pre-computed CSVs from Step 1
- Creates 4 multi-panel publication-quality figures
- Saves PNG images at 300 DPI

**Runtime:** ~10-20 seconds

**Output location:** `notes/assets/images/014-genes-enriched-at-extreme-tmi/`

## Usage

```bash
# Run complete pipeline from torchcell root
bash experiments/014-genes-enriched-at-extreme-tmi/scripts/014-genes-enriched-at-extreme-tmi.sh
```

## Modular Execution

If you only need to re-run part of the pipeline, use the convenience scripts:

```bash
# Re-run analysis only (e.g., changed parameters)
bash experiments/014-genes-enriched-at-extreme-tmi/scripts/run_analysis_only.sh

# Re-run visualization only (e.g., changed plot styling)
bash experiments/014-genes-enriched-at-extreme-tmi/scripts/run_visualization_only.sh
```

## Outputs Generated

### Data Files (7 files)

1. `uncharacterized_extreme_interactions_all.csv` (2,852 interactions)
2. `uncharacterized_extreme_interactions_high_conf.csv` (239 interactions)
3. `uncharacterized_gene_counts_all.csv` (178 genes)
4. `uncharacterized_gene_counts_high_conf.csv` (68 genes)
5. `uncharacterized_enrichment_fractions_all.csv` (178 genes)
6. `uncharacterized_enrichment_fractions_high_conf.csv` (68 genes)
7. `uncharacterized_extreme_tmi_summary.txt`

### Image Files (4 figures)

1. `extreme_interaction_distributions.png`
2. `gene_enrichment_all_criterion.png`
3. `gene_enrichment_high_conf_criterion.png`
4. `gene_enrichment_fractions.png`

## Key Results

### All Extreme Interactions (|τ| > 0.1)

- 2,852 interactions involving uncharacterized genes
- 178 unique uncharacterized genes participate
- Top gene: **YMR310C** (208 extreme interactions)

### High-Confidence Subset (|τ| > 0.1, p < 0.05)

- 239 statistically significant interactions
- 68 unique uncharacterized genes
- Top gene: **YJR115W** (51 high-confidence interactions)

### Enrichment Insights

- Some uncharacterized genes are specifically enriched in extreme TMI
- Directional patterns: genes preferentially aggravating vs alleviating
- Suggests potential functional roles based on interaction patterns

## Implementation Notes

### Dependencies

- Requires experiment 013 results (`uncharacterized_genes.json`)
- Requires experiment 010 dataset (Kuzmin TMI processed data)
- Environment variables: `DATA_ROOT`, `EXPERIMENT_ROOT`, `ASSET_IMAGES_DIR`

### Design Philosophy

- **Separation of concerns:** Data processing separate from visualization
- **Fast iteration:** Visualization can be tweaked without re-running slow analysis
- **Modular execution:** Can run steps independently via convenience scripts
- **Reproducibility:** Single command reproduces entire analysis

### Analysis Parameters

Hard-coded in both scripts (must match):

- `SCORE_THRESHOLD = 0.1` (defines "extreme")
- `P_VALUE_THRESHOLD = 0.05` (defines "high-confidence")

## Related Notes

- [[experiments.014-genes-enriched-at-extreme-tmi.scripts.analyze_extreme_interactions]] - Data processing details
- [[experiments.014-genes-enriched-at-extreme-tmi.scripts.visualize_extreme_interactions]] - Visualization details
- [[experiments.013-uncharacterized-genes.scripts.count_dubious_and_uncharacterized_genes]] - Source of uncharacterized gene list
