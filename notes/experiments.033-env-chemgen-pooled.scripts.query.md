---
id: 5cgwvcz8nh19xnizzmv0dx7
title: Query
desc: ''
updated: 1790549467687
created: 1790549467687
---

## 2026.09.27 - Build script

`Neo4jCellDataset` with `converter=None`, `deduplicator=None` and
`GenotypeEnvironmentAggregator` ([[torchcell.data.genotype_environment_aggregate]]). Stages:
raw (main's single-threaded `Neo4jQueryRaw`) -> aggregation by the (genotype, environment)
cell -> processed -> the phenotype, perturbation-count and dataset-name indices and the label
table. A final pass over the processed store histograms the measurements per entry and counts
experiments per dataset, written to `dataset_index_summary.json` beside the served counts, so
both the gene-filter loss and the multi-measurement cells are counted.

`--genes` runs a smoke build on a gene list into `--root`; `scripts/gh_query_smoke.slurm`
does that on eight genes under slurm (every job that touches the served store is a slurm
job). The full build is `scripts/gh_query_build_001.slurm` (16 CPUs, 64 GB, 2 days) from a
detached worktree pinned at the build commit, root
`/db/experiments/033-env-chemgen-pooled-001-pooled-build` symlinked from
`$DATA_ROOT/data/torchcell/experiments/033-env-chemgen-pooled/001-pooled-build`.
