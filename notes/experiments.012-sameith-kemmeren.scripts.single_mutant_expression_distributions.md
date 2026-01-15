---
id: 2i4so9py9300xtoc5z24fx0
title: Single Mutant Expression Distributions - Experiment 012
desc: >-
  Box plot visualizations of genome-wide expression changes for single gene
  deletions
updated: 1768502832530
created: 1767758979960
---

# Single Mutant Expression Distribution Visualizations

## Purpose

Creates box plot visualizations showing the distribution of genome-wide log2 expression changes for each single gene deletion strain. Provides an overview of expression variability across:

- **Kemmeren2014**: ~1,484 deletion mutants
- **Sameith2015**: 82 GSTF deletion mutants

Helps identify:

- Genes whose deletion causes widespread transcriptional changes (high variance)
- Genes with minimal transcriptional impact (low variance)
- Outlier genes with extreme expression changes

## Implementation

**Script**: `experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions.py`

**Related Code**:

- [[torchcell.datasets.scerevisiae.kemmeren2014]]
- [[torchcell.datasets.scerevisiae.sameith2015]]

## Visualization

Box plots show log2 expression change distributions per deletion strain:

- **Wide boxes**: Broad transcriptional reprogramming
- **Narrow boxes**: Minimal genome-wide impact
- **Outliers**: Highly specific gene effects

## Outputs

### Images

![Single Mutant Kemmeren](./assets/images/012-sameith-kemmeren-expression/single_mutant_kemmeren.png)

*Figure 1: Log2 expression changes across ~1,484 Kemmeren2014 single gene deletions. Each vertical box plot shows the distribution of expression changes across ~6K measured genes for one deletion strain. Statistics show **average per-strain** counts: e.g., "±0.25 log2 FC: (631/6168) 10.23%" means on average, 631 out of 6168 measured genes per deletion strain exceed ±0.25 log2 fold-change.*

![Single Mutant Sameith](./assets/images/012-sameith-kemmeren-expression/single_mutant_sameith.png)

*Figure 2: Log2 expression changes across 82 Sameith2015 GSTF deletions. Similar structure to Kemmeren plot but focused on transcription factor deletions. Statistics show **average per-strain** counts: numerator and denominator represent averages across the 82 deletion strains, not pooled totals.*

## Key Findings

- **Kemmeren2014**: Most deletions show median ~0, IQR 0.5-1.0; few global regulators cause broad changes
- **Sameith2015**: Transcription factor deletions show variable impacts despite expected broad effects
- **Cross-study**: Similar distribution shapes and variance after sign inversion fix (±6 log2 FC range)

## Related Notes

- [[experiments.012-sameith-kemmeren.scripts.verify_metadata]] - QC checks performed first
- [[experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]] - Cross-study correlation

## Usage

```bash
python experiments/012-sameith-kemmeren/scripts/single_mutant_expression_distributions.py
# Or: bash experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren.sh
```
