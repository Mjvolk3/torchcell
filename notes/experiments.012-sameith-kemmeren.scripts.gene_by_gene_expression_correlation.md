---
id: 24595cmt9snxeu6pv5165fl
title: Gene-by-Gene Expression Correlation - Experiment 012
desc: >-
  Cross-study correlation analysis demonstrating biological consistency after
  sign convention fix
updated: 1768520223905
created: 1767758069863
---

# Gene-by-Gene Expression Profile Correlation

## Purpose

Assesses cross-study reproducibility between Kemmeren2014 and Sameith2015 by correlating full genome-wide expression profiles for 82 overlapping gene deletions.

**Key Question**: Do the same gene deletions produce similar transcriptional responses across independent studies?

**Method**: For each of the 82 overlapping genes:

1. Extract full expression vector (~6K genes) from Kemmeren dataset
2. Extract full expression vector (~6K genes) from Sameith dataset
3. Calculate Pearson and Spearman correlation between vectors
4. Analyze distribution of correlation coefficients

## Implementation

**Script**: `experiments/012-sameith-kemmeren/scripts/gene_by_gene_expression_correlation.py`

**Related Code**:

- [[torchcell.datasets.scerevisiae.kemmeren2014]]
- [[torchcell.datasets.scerevisiae.sameith2015]]
- [[torchcell.datamodels.schema]] - Canonical log2 ratio convention

## Outputs

### Images

![Correlation Distribution Pearson](./assets/images/012-sameith-kemmeren-expression/gene_expression_correlation_dist_pearson.png)

*Figure 1: Cross-study expression profile correlation distribution (Pearson). Shows **median r = 0.599** with 57.3% of genes having r > 0.5, confirming biological consistency after sign convention fix. The positive correlation demonstrates that identical gene deletions produce similar transcriptional responses across independent studies.*

![Correlation Distribution Spearman](./assets/images/012-sameith-kemmeren-expression/gene_expression_correlation_dist_spearman.png)

*Figure 2: Cross-study expression profile correlation distribution (Spearman). Shows **median r = 0.593**, consistent with Pearson results. The similarity between Pearson and Spearman correlations indicates linear relationships without major rank-order distortions.*

### Data Files

- `results/gene_expression_correlations.csv` - Per-gene Pearson and Spearman correlation coefficients

## Key Findings (Post-Fix)

### GOOD Cross-Study Reproducibility Confirmed

After fixing the sign inversion bug (2026-01-06), the results show **strong biological consistency**:

| Metric                 | Value             | Interpretation                                 |
|------------------------|-------------------|------------------------------------------------|
| **Median Pearson r**   | **+0.599**        | Moderate-to-strong positive correlation        |
| **Median Spearman r**  | **+0.593**        | Consistent with Pearson (linear relationships) |
| **Genes with r > 0.5** | **57.3% (47/82)** | Majority show good reproducibility             |
| **Genes with r > 0.7** | **37.8% (31/82)** | Strong agreement for ~1/3 of genes             |
| **Genes with r < 0**   | **~20%**          | Minority with poor cross-study agreement       |

### Biological Interpretation

**Positive Correlation = Biological Validation**

The positive median correlation confirms:

1. **Sign convention fix was correct**: Both datasets now use canonical log2(sample/reference)
2. **Biological signal is reproducible**: Same deletions → similar transcriptional responses
3. **Technical noise is manageable**: Despite different platforms, biological signal dominates
4. **Datasets are usable together**: Can train models on one, validate on the other (with domain adaptation)

### Variance Explained

Median r = 0.599 is good for cross-platform comparisons, with variance from:

- Platform differences (probe designs, sensitivities)
- Batch effects and biological noise
- Strain drift over time

### Historical Note & Implications

**Sign bug fix (2026-01-06)**: Corrected dye-swap handling in `kemmeren2014.py` flipped correlation from -0.599 to +0.599, confirming biological reproducibility.

**Dataset compatibility**: Kemmeren2014 and Sameith2015 can be integrated with domain adaptation (transfer learning, CORAL/MMD).

## Related Notes

- [[torchcell.datamodels.schema]] - Canonical sign convention documentation
- [[experiments.012-sameith-kemmeren.scripts.verify_metadata]] - QC checks

## Usage

```bash
python experiments/012-sameith-kemmeren/scripts/gene_by_gene_expression_correlation.py
# Or: bash experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren.sh
```
