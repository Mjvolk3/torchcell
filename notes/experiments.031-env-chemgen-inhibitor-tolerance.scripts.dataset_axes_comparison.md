---
id: 9mbltdlmqw0sni3cjnmwha9
title: Dataset_axes_comparison
desc: ''
updated: 1790375602220
created: 1790375602220
---

## 2026.09.25 - Axis table, environment catalogs, and overlaps

Reads the flattened parquet files and writes, under
`experiments/031-env-chemgen-inhibitor-tolerance/results/`:

- `axes_table.md` / `.csv`: one row per axis, one column per dataset, each cell the
  distinct values with their record counts. Background genes are detected as the genes
  present in every genotype of a dataset (Vanacloig's `pdr1 pdr3 snq2` drug-sensitized
  host), and "queried genes" excludes them.
- `environment_catalog_<dataset>.csv`: one row per distinct environment, where the
  identity is compound + dose + physical factor + temperature + duration + screen, with
  record and gene counts and the median response.
- `gene_overlap.csv`, `compound_overlap.csv`, `shared_compounds.csv` and the combined
  `overlap.md`: pairwise and three-way intersections of queried genes and of compounds
  keyed by InChIKey, plus for every compound shared with Vanacloig the doses each dataset
  used. The HOM vs HET compound overlap is counted but its per-compound rows are not
  listed, because that pair is not the question.

The findings are discussed in [[experiments.031-env-chemgen-inhibitor-tolerance]].
