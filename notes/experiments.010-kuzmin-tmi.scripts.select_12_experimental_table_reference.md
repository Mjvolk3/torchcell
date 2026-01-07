---
id: 2rip11u9t4mxo0oq5vej2n7
title: Select_12_experimental_table_reference
desc: ''
updated: 1767746123966
created: 1767746123966
---

## Overview

Creates an enriched experimental reference table for the 12-gene panel (k=200) with Sameith overlap flags and pair frequency information. This table helps prioritize which double mutants to construct by showing literature overlap.

## Key Features

1. **Gene ordering by frequency**: Most common gene in panel → `gene1`
2. **Sameith overlap flags**: Identifies genes/pairs with existing double expression data
3. **Pair count columns**: Shows how many triples each pair enables

## Outputs

### `experimental_table_12_genes_k200.csv`

Full table with all panel triples.

### `experimental_table_12_genes_k200_constructible.csv`

Filtered to only DMF-constructible triples (where all pairs have DMF fitness ≥ 1.0).

### Column Descriptions

| Column                      | Description                                               |
|-----------------------------|-----------------------------------------------------------|
| `rank`                      | Rank by inferred gene interaction (descending)            |
| `gene1`, `gene2`, `gene3`   | Genes ordered by frequency (gene1 = most frequent)        |
| `inferred_gene_interaction` | Model prediction value                                    |
| `constructible`             | Boolean: are all genes in the panel?                      |
| `gene{1,2,3}_in_sameith`    | Boolean: is this gene in Sameith dataset (82 genes)?      |
| `pair_{12,13,23}_in_sameith`| Boolean: are BOTH genes in the pair in Sameith?           |
| `gene1_gene2_count`         | How many panel triples contain this pair                  |
| `gene1_gene3_count`         | How many panel triples contain this pair                  |
| `gene2_gene3_count`         | How many panel triples contain this pair                  |

## Why Sameith Overlap Matters

The Sameith et al. dataset contains gene expression profiles for 82 genes' double mutants. If a pair in our panel overlaps with Sameith:

- We can validate our predictions against published expression data
- We have existing double mutant strains available
- Enables reproducibility comparisons with literature

## Dependencies

- Requires `gene_selection_results.csv` from [[select_12_and_24_genes_top_triples|experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples]]
- Requires `sameith_doubles_genes.txt` from experiment 006
- Requires `constructible_triples_panel12_k200.parquet` for DMF filtering

## Related Scripts

- [[select_12_and_24_genes_top_triples|experiments.010-kuzmin-tmi.scripts.select_12_and_24_genes_top_triples]] - Main gene selection algorithm
- [[select_12_k200_tables_hist|experiments.010-kuzmin-tmi.scripts.select_12_k200_tables_hist]] - Basic tables without Sameith annotations
