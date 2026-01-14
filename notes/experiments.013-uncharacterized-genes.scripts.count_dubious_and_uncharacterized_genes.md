---
id: 2m8fw3v98njfggri1v92f35
title: Count_dubious_and_uncharacterized_genes
desc: ''
updated: 1766136413291
created: 1766136413291
---

## Overview

Collects and categorizes all uncharacterized and dubious genes from the S. cerevisiae S288C genome (R64-4-1_20230830).

## Functionality

### Gene Collection

- Uses `SCerevisiaeGenome` to access the complete gene set
- Filters genes by `orf_classification` attribute
- Collects genes with "Uncharacterized" or "Dubious" in their classification

### Data Model

`UncharacterizedGeneData` - Strict Pydantic model containing:

- Systematic gene ID and names/aliases
- ORF classification
- Genomic coordinates (chromosome, start, end, strand)
- Sequences: DNA, protein, CDS
- Annotations: GO terms, ontology terms, notes
- Database cross-references

### Set Analysis

Performs proper set operations to analyze overlaps:

- **Uncharacterized only**: Exclusively "Uncharacterized" genes
- **Dubious only**: Exclusively "Dubious" genes
- **Intersection**: Genes with both classifications (if any exist)
- **Union**: All unique uncharacterized + dubious genes

## Outputs

### JSON Files (in `results/`)

1. `uncharacterized_genes.json` - All genes with "Uncharacterized" classification (668 genes)
2. `dubious_genes.json` - All genes with "Dubious" classification (684 genes)
3. `only_uncharacterized_genes.json` - Exclusively uncharacterized
4. `only_dubious_genes.json` - Exclusively dubious
5. `intersection_genes.json` - Genes with both classifications (if any)
6. `union_all_genes.json` - All unique genes combined

### Console Output

- Gene counts for each category
- Set analysis statistics (intersection, exclusive sets, union)
- Example genes from each category with detailed attributes
- File paths for all saved data

## Usage

```bash
python experiments/013-uncharacterized-genes/scripts/count_dubious_and_uncharacterized_genes.py
```

## Gene Counts

Expected vs Actual:

- **Uncharacterized**: ~571 (official) → 668 (actual)
- **Dubious**: ~660 (official) → 684 (actual)

Higher counts include mitochondrial genes and genes with partial annotations that still carry these classifications.

## Implementation Notes

- Uses `ModelStrict` from `torchcell.datamodels.pydant` for immutable, validated data
- Environment variables `DATA_ROOT` and `EXPERIMENT_ROOT` must be set
- Genome loaded with `overwrite=False` to use existing database
- Set operations prevent naive dictionary merging that would hide overlaps
