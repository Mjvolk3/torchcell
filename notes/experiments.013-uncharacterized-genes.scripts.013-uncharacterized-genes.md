---
id: ckv9jkajvgf3drym928x1bx
title: 013 Uncharacterized Genes
desc: ''
updated: 1768422367776
created: 1768422367776
---

## Overview

Master orchestration script that runs the complete experiment 013 pipeline: gene collection, triple interaction analysis, and essential gene overlap detection.

## Purpose

Provides a single command to reproduce the entire uncharacterized gene analysis from scratch. Ensures steps run in correct order with proper dependencies.

## Pipeline Steps

### Step 1: Gene Collection

**Script:** `count_dubious_and_uncharacterized_genes.py`

**What it does:**

- Queries SGD genome (GFF3) for `orf_classification` field
- Queries SGD graph API for `qualifier` field
- Identifies 668 uncharacterized genes (99.85% agreement between sources)
- Identifies 684 dubious genes
- Saves union, intersection, and source-specific gene lists

**Runtime:** ~30 seconds (API queries)

**Output location:** `experiments/013-uncharacterized-genes/results/`

### Step 2: Triple Interaction Analysis

**Script:** `triple_interaction_enrichment_of_uncharacterized_genes.py`

**What it does:**

- Loads 668 uncharacterized genes from Step 1
- Counts triple mutant interactions (TMI) involving these genes
- Identifies 44,619 interactions where ≥1 gene is uncharacterized
- Creates distribution histogram showing interaction counts per gene

**Runtime:** ~1 minute (dataset access)

**Output location:**

- Data: `experiments/013-uncharacterized-genes/results/`
- Images: `notes/assets/images/013-uncharacterized-genes/`

### Step 3: Essential Gene Overlap

**Script:** `uncharacterized_essential_overlap.py`

**What it does:**

- Compares uncharacterized genes (genome & graph sources) with essential genes
- Essential = inviable phenotype in null mutants (S288C strain)
- Finds 3 genes in Essential ∩ Uncharacterized intersection
- Creates UpSet plot showing overlap between sets
- Identifies 1 annotation discrepancy (YCR016W)

**Runtime:** ~30 seconds (API queries)

**Output location:** `notes/assets/images/013-uncharacterized-genes/`

## Usage

```bash
# Run complete pipeline from torchcell root
bash experiments/013-uncharacterized-genes/scripts/013-uncharacterized-genes.sh
```

## Outputs Generated

### Data Files (10 JSON/CSV files)

1. `uncharacterized_genes.json` (668 genes)
2. `dubious_genes.json` (684 genes)
3. `only_uncharacterized_genes.json` (genome-specific)
4. `only_dubious_genes.json` (genome-specific)
5. `union_all_genes.json` (1,352 genes)
6. `gene_interaction_counts.csv` (uncharacterized gene counts)
7. `uncharacterized_triple_interactions.csv` (44,619 interactions)

### Image Files (3 figures)

1. `gene_interaction_counts.png` (distribution histogram)
2. `uncharacterized_distribution.png` (genome vs graph comparison)
3. `upset_essential_uncharacterized_all.png` (set overlap visualization)

## Key Results

### Gene Collection

- 668 uncharacterized genes identified
- 684 dubious genes identified
- 99.85% agreement between genome (GFF3) and graph (SGD API) sources
- 1 discrepancy: YCR016W (recently characterized, annotation lag)

### Triple Interaction Analysis

- 44,619 TMI interactions involve ≥1 uncharacterized gene
- Distribution highly skewed: few genes have many interactions
- Enables downstream enrichment analysis (experiment 014)

### Essential Gene Overlap

- Only 3 genes are both Essential AND Uncharacterized:
  - YEL035C (false positive - deletion affects neighbor's TATA box)
  - YJL119C (true essential, uncharacterized)
  - YOR296W (true essential, uncharacterized)
- Shows annotation completeness: most essential genes are characterized

## Implementation Notes

### Dependencies

- Requires SGD genome (GFF3 file) access
- Requires SGD graph database (neo4j) running
- Environment variables: `DATA_ROOT`, `EXPERIMENT_ROOT`, `ASSET_IMAGES_DIR`

### Design Philosophy

- **Sequential pipeline:** Each step depends on previous results
- **Data reusability:** JSON outputs feed into downstream experiments
- **Reproducibility:** Single command reproduces entire analysis
- **Publication-quality:** UpSet plots for multi-way set visualization

### Data Sources

- **Genome:** GFF3 `orf_classification` field
- **Graph:** SGD API `qualifier` field
- **Essential genes:** Phenotype records (inviable null mutants, S288C)
- **Triple interactions:** Kuzmin 2018 + 2020 datasets

## Related Notes

- [[experiments.013-uncharacterized-genes.scripts.count_dubious_and_uncharacterized_genes]] - Gene collection details
- [[experiments.013-uncharacterized-genes.scripts.triple_interaction_enrichment_of_uncharacterized_genes]] - Interaction analysis details
- [[experiments.013-uncharacterized-genes.scripts.uncharacterized_essential_overlap]] - Essential gene overlap details
- [[experiments.014-genes-enriched-at-extreme-tmi.scripts.014-genes-enriched-at-extreme-tmi]] - Downstream enrichment analysis
