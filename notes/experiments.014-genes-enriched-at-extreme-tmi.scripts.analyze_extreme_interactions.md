---
id: xakibzffrvlos9dsjmxh5xs
title: Analyze_extreme_interactions
desc: ''
updated: 1768419727489
created: 1768416026640
---

## Overview

Data processing script that identifies uncharacterized genes enriched in extreme triple mutant interactions (TMI). Loads the Kuzmin TMI dataset, filters for interactions with |τ| > 0.1, and calculates enrichment statistics.

## Purpose

**This is the slow step** - only needs to be run once or when analysis parameters change. Separates computationally expensive dataset loading and interaction filtering from fast visualization iteration.

## Data Sources

### Uncharacterized Genes

- Loads from experiment 013: `uncharacterized_genes.json` (668 genes)

### TMI Dataset

- Uses existing processed dataset from experiment 010
- Path: `DATA_ROOT/data/torchcell/experiments/010-kuzmin-tmi/001-small-build`
- Contains 376,732 total interactions (Kuzmin 2018 + 2020)
- Query from: `experiments/009-kuzmin-tmi/queries/001_small_build.cql`

## Analysis Pipeline

### 1. Extreme Interaction Detection

**Function:** `analyze_extreme_interactions()`

- Uses `is_any_perturbed_gene_index` for efficient lookup
- Filters interactions where |τ| > 0.1 (score_threshold)
- Tracks two criteria:
  - **All extreme:** |τ| > 0.1
  - **High-confidence:** |τ| > 0.1 AND p < 0.05

### 2. Gene Participation Counting

**Function:** `count_gene_participation()`

- Counts how many extreme interactions each uncharacterized gene participates in
- Only tracks uncharacterized genes (ignores characterized genes in the interaction)
- Reports statistics: total, mean, median, max

### 3. Enrichment Fraction Calculation

**Function:** `calculate_enrichment_fractions()`

**Formula:**

```
enrichment_fraction = (# extreme interactions) / (# total interactions for that gene)
```

**Interpretation:**

- **1.0:** All interactions involving this gene are extreme
- **0.5:** Half of this gene's interactions are extreme
- **0.0:** Gene has no extreme interactions

## Outputs

All files saved to: `experiments/014-genes-enriched-at-extreme-tmi/results/`

### Interaction Data (CSV)

1. **`uncharacterized_extreme_interactions_all.csv`** (2,852 interactions)
   - Columns: idx, num_genes, genes, num_uncharacterized, uncharacterized_genes, gene_interaction, p_value, abs_score

2. **`uncharacterized_extreme_interactions_high_conf.csv`** (239 interactions)
   - Same structure, filtered for p < 0.05

### Gene Statistics (CSV)

3. **`uncharacterized_gene_counts_all.csv`** (178 genes)
   - Columns: gene, count (sorted by count descending)

4. **`uncharacterized_gene_counts_high_conf.csv`** (68 genes)
   - Same structure for high-confidence subset

### Enrichment Fractions (CSV)

5. **`uncharacterized_enrichment_fractions_all.csv`** (178 genes)
   - Columns: gene, total_interactions, extreme_interactions, enrichment_fraction

6. **`uncharacterized_enrichment_fractions_high_conf.csv`** (68 genes)
   - Same structure for high-confidence subset

### Summary (TXT)

7. **`uncharacterized_extreme_tmi_summary.txt`**
   - Parameters, dataset sizes, top 10 genes for each criterion

## Key Results

### All Extreme Interactions (|τ| > 0.1)

- 2,852 interactions involving 178 unique uncharacterized genes
- Top gene: YMR310C (208 extreme interactions)

### High-Confidence (|τ| > 0.1, p < 0.05)

- 239 interactions involving 68 unique uncharacterized genes
- Top gene: YJR115W (51 high-confidence extreme interactions)

### Enrichment Statistics

- Mean enrichment fraction: varies by gene
- Shows which uncharacterized genes are specifically enriched in extreme interactions

## Usage

```bash
# Run analysis (slow - dataset loading takes time)
python experiments/014-genes-enriched-at-extreme-tmi/scripts/analyze_extreme_interactions.py

# Or use convenience script
bash experiments/014-genes-enriched-at-extreme-tmi/scripts/run_analysis_only.sh
```

## Implementation Notes

- Uses `is_any_perturbed_gene_index` for O(1) gene lookup instead of O(n) iteration
- Tracks `seen_indices` to avoid duplicate counting when multiple uncharacterized genes in same interaction
- Handles missing p-values gracefully (sets to NaN)
- Separates data processing from visualization for fast iteration
- No plotting code - purely data processing and CSV generation

## Related Scripts

- **Visualization:** `visualize_extreme_interactions.py` (creates plots from these CSVs)
- **Master script:** `014-genes-enriched-at-extreme-tmi.sh` (runs both in sequence)
