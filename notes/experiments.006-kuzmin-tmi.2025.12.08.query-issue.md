---
id: fyhpat5anqs1tcye8o4o6pi
title: Query Issue
desc: ''
updated: 1765218692631
created: 1765218441405
---
# TmiKuzmin Dataset Filtering Analysis

## Analysis Provenance

This analysis was produced by investigating query inconsistencies between TmiKuzmin2018 and TmiKuzmin2020 datasets in experiment 006.

**Key Files:**

Analysis Scripts:

- `experiments/006-kuzmin-tmi/scripts/analyze_filtering_tradeoffs.py` - Comprehensive analysis of raw datasets, categorizes experiments by perturbation types, computes all filtering scenarios
- `experiments/006-kuzmin-tmi/scripts/check_perturbation_types.py` - Analyzes queried 006 dataset, reveals inconsistent filtering between 2018/2020

Results Data:

- `experiments/006-kuzmin-tmi/results/filtering_tradeoffs_analysis.json` - Raw data for all tables in sections 1-3
- `experiments/006-kuzmin-tmi/results/perturbation_types_analysis.json` - Proves TmiKuzmin2018 has non-deletions, TmiKuzmin2020 doesn't

Query Fix:

- `experiments/007-kuzmin-tm/queries/002_small_build.cql` - Corrected query with consistent deletion filtering for both datasets
- `experiments/007-kuzmin-tm/queries/001_small_build.cql` - Original inconsistent query (for comparison)

Related Code:

- `experiments/007-kuzmin-tm/scripts/query.py` - Updated to use 002 query and new dataset root

---

## Executive Summary

Analysis of raw TmiKuzmin2018 and TmiKuzmin2020 datasets to understand the impact of different perturbation filtering strategies.

**Key Finding**: The current 006 query has INCONSISTENT filtering:

- TmiKuzmin2018: Missing deletion filter → includes 33,228 experiments with TS alleles/alleles (36.5%)
- TmiKuzmin2020: Has deletion filter → excludes 70,103 experiments with TS alleles (23.2%)

---

## 1. Raw Dataset Composition

### Perturbation Type Breakdown

| Dataset | Total Experiments | Deletion Perts | TS Allele Perts | Allele Perts |
|---------|-------------------|----------------|-----------------|--------------|
| **TmiKuzmin2018** | 91,111 | 232,304 (85.0%) | 14,349 (5.3%) | 26,680 (9.8%) |
| **TmiKuzmin2020** | 301,798 | 856,296 (94.6%) | 49,098 (5.4%) | 0 (0.0%) |
| **Combined** | 392,909 | 1,088,600 (89.5%) | 63,447 (5.2%) | 26,680 (2.2%) |

**Note**: Perturbation counts are higher than experiment counts because each triple mutant has 3 perturbations.

### Experiment Composition by Perturbation Mix

| Experiment Type | TmiKuzmin2018 | % of 2018 | TmiKuzmin2020 | % of 2020 |
|-----------------|---------------|-----------|---------------|-----------|
| **Deletion only** | 57,883 | 63.5% | 252,700 | 83.7% |
| **Deletion + TS allele** | 10,852 | 11.9% | 49,098 | 16.3% |
| **Deletion + Allele** | 18,879 | 20.7% | 0 | 0.0% |
| **Deletion + TS + Allele** | 2,814 | 3.1% | 0 | 0.0% |
| **TS allele + Allele only** | 683 | 0.7% | 0 | 0.0% |
| **TS allele only** | 0 | 0.0% | 0 | 0.0% |
| **Allele only** | 0 | 0.0% | 0 | 0.0% |

---

## 2. Filtering Scenario Comparison

| Filtering Strategy | 2018 Kept | 2020 Kept | Combined | % of Total Data |
|-------------------|-----------|-----------|----------|-----------------|
| **Deletion Only** (002 query) | 57,883 | 252,700 | 310,583 | 79.0% |
| **Deletion + Mixed** (relaxed) | 90,428 | 301,798 | 392,226 | 99.8% |
| **All Perturbations** (no filter) | 91,111 | 301,798 | 392,909 | 100.0% |
| **Current 006 Query** (inconsistent) | 90,581 | 231,695 | 322,276 | 82.0% |

---

## 3. Impact Analysis: Strict Deletion-Only Filter

### What We Keep vs Lose

| Dataset | Keep (Deletion Only) | Lose (Has Non-Deletion) | % Lost |
|---------|---------------------|------------------------|---------|
| **TmiKuzmin2018** | 57,883 | 33,228 | 36.5% |
| **TmiKuzmin2020** | 252,700 | 49,098 | 16.3% |
| **Combined** | 310,583 | 82,326 | 21.0% |

### Breakdown of What We Lose (2018 only, since 2020 loses only TS alleles)

| Lost Experiment Type | Count | % of 2018 Dataset |
|---------------------|-------|-------------------|
| Deletion + Allele | 18,879 | 20.7% |
| Deletion + TS allele | 10,852 | 11.9% |
| Deletion + TS + Allele | 2,814 | 3.1% |
| TS allele + Allele only | 683 | 0.7% |
| **Total Lost** | **33,228** | **36.5%** |

---

## 4. Current 006 Query Analysis

### Discrepancy Between Raw Data and Query Results

| Dataset | Raw Total | Query Result | Filtered Out | Filtering |
|---------|-----------|--------------|--------------|-----------|
| **TmiKuzmin2018** | 91,111 | 90,581 | 530 (0.6%) | Almost none (weak filter) |
| **TmiKuzmin2020** | 301,798 | 231,695 | 70,103 (23.2%) | Strong (deletion filter) |

**Current 006 Query Behavior:**

- **TmiKuzmin2018**: Uses `EXISTS` (≥1 gene in gene_set) + NO deletion filter
  - Keeps: 99.4% of data (90,581 / 91,111)
  - Includes: All experiment types (deletion, allele, TS allele mixes)

- **TmiKuzmin2020**: Uses `ALL` (all genes in gene_set) + HAS deletion filter
  - Keeps: 76.8% of data (231,695 / 301,798)
  - Excludes: All experiments with TS alleles (49,098 experiments)

**Mystery**: 2020 query should keep ~252,700 deletion-only experiments, but only keeps 231,695

- Missing: 21,005 deletion-only experiments (7.7% of deletion-only data)
- Likely filtered by gene_set constraint (genes not in filtered genome)

---

## 5. Recommendations

### Option A: Strict Deletion-Only (002 Query)

✅ **Consistent filtering across both datasets**
✅ **Clean dataset with only deletion perturbations**
❌ **Lose 82,326 experiments (21% of data)**
❌ **Lose valuable TS allele data for essential genes**

**Use case**: When you need purely comparable deletion experiments

### Option B: Include All Perturbation Types

✅ **Maximum data retention (392,226 experiments if fixing gene_set issue)**
✅ **Can study TS alleles (essential gene phenotypes)**
✅ **More biological diversity**
❌ **Need to handle different perturbation types in analysis**
❌ **May complicate interpretation**

**Use case**: When you want maximum coverage and can model different perturbation types

### Option C: Separate Analyses

✅ **Can optimize each analysis for its question**
✅ **Keeps all data for different purposes**
❌ **More complex workflow**

**Use case**:

- Deletion-only for baseline comparisons
- Include TS alleles when studying essential gene interactions

---

## 6. Data Coverage by Biological Relevance

### Why TS Alleles Matter

**Temperature-sensitive alleles represent essential genes** that cannot be studied via deletion:

- 2018: 14,349 TS allele perturbations across 13,666 experiments (14.3% of 2018 experiments with TS)
- 2020: 49,098 TS allele perturbations across 49,098 experiments (16.3% of 2020 experiments with TS)

**Biological coverage**:

- Essential genes make up ~17-20% of yeast genome
- TS alleles are the ONLY way to study triple interactions involving these genes
- Excluding them means losing genetic interaction data for critical cellular processes

### Scientific Trade-off

| Metric | Deletion Only | With TS Alleles |
|--------|---------------|-----------------|
| Gene coverage | ~80% of genome | ~100% of genome |
| Data consistency | High (all same type) | Lower (mixed types) |
| Biological completeness | Missing essential genes | Complete |
| Analysis complexity | Simple | Moderate |
| Total experiments | 310,583 | 392,226 |

---

## 7. Recommended Action

**For 007 dataset (Tmf + Tmi matching pairs)**:

Use **002 query** (deletion-only) because:

1. Need matching Tmf (fitness) and Tmi (interaction) data
2. Fitness values may not exist for all TS allele combinations
3. Ensures clean 1:1 correspondence between fitness and interaction

**For future 006 dataset rebuild**:

Consider **relaxed filter** (include TS alleles) because:

1. These are triple mutant INTERACTIONS - fitness matching is less critical
2. Gain 25% more data
3. Include essential gene interactions
4. Can always filter to deletion-only post-hoc if needed

**Immediate fix**: Apply 002 query to both datasets for consistency`
