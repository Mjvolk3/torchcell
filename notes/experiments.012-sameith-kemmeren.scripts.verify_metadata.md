---
id: wunjdk50aiwvvedttef9v1s
title: Metadata Verification - Experiment 012
desc: Quality control verification of Kemmeren2014 and Sameith2015 datasets
updated: 1768515889306
created: 1767758984749
---

# Metadata Verification Script

## Purpose

Quality control script that verifies metadata consistency across three microarray datasets used in Experiment 012:

- **Kemmeren2014**: ~1,484 single deletion mutants
- **SmMicroarraySameith2015**: 82 single deletion mutants (GSTFs only)
- **DmMicroarraySameith2015**: ~72 double deletion mutants (GSTF pairs)

Ensures data integrity before cross-study comparisons.

## Implementation

**Script**: `experiments/012-sameith-kemmeren/scripts/verify_metadata.py`

**Related Code**:

- [[torchcell.datasets.scerevisiae.kemmeren2014]]
- [[torchcell.datasets.scerevisiae.sameith2015]]

## Checks Performed

### 1. Reference Log2 Ratios

- ✅ All reference samples should have log2(sample/reference) = 0.0 ± 1e-6
- Detects data processing errors or incorrect reference handling

### 2. Environmental Conditions

- ✅ All experiments: SC liquid media at 30°C
- Ensures biological comparability across datasets

### 3. Strain Distribution

- ✅ Kemmeren: Predominantly BY4741/BY4742
- ✅ Sameith: Correct strain assignments
- Verifies genetic background consistency

### 4. Expression Data Completeness

- ✅ No NaN values in expression_log2_ratio dictionaries
- ✅ ~6,000-6,600 genes measured per experiment
- Ensures downstream analyses won't encounter missing data

### 5. Data Structure Integrity

- ✅ Required keys present: experiment, reference, publication
- ✅ Nested structure correct: genotype, phenotype, environment
- Catches schema violations early

## Outputs

### Data Files

- `results/metadata_verification_summary.csv` - Summary statistics per dataset
- `results/metadata_verification_anomalies.txt` - Detailed anomaly reports
- `results/verification.log` - Detailed log of verification run

## Key Findings

### All Datasets Pass Quality Checks ✅

| Dataset       | Samples | Reference Violations | Environment Issues | Missing Values |
|---------------|---------|----------------------|--------------------|----------------|
| Kemmeren2014  | 1,484   | 0                    | 0                  | 0              |
| SmSameith2015 | 82      | 0                    | 0                  | 0              |
| DmSameith2015 | 72      | 0                    | 0                  | 0              |

**Interpretation**: All three datasets have clean metadata and are suitable for cross-study comparisons. No preprocessing artifacts detected.

## Historical Context

This verification was performed **before** the sign inversion bug was discovered. The metadata checks passed, confirming the bug was in the **expression value calculation**, not in metadata handling.

## Related Notes

- [[experiments.012-sameith-kemmeren.scripts.gene_by_gene_expression_correlation]] - Uses verified datasets

## Usage

```bash
python experiments/012-sameith-kemmeren/scripts/verify_metadata.py
# Or: bash experiments/012-sameith-kemmeren/scripts/012-sameith-kemmeren.sh
```
