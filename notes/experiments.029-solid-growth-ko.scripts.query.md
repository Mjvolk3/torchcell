---
id: yymsufcorexblrfvcymo4bn
title: Query
desc: ''
updated: 1789739333040
created: 1789739333040
---

## 2026.09.18 - Build script

`Neo4jCellDataset` with `CompositeFitnessConverter`, `deduplicator=None`, `GenotypeAggregator`.
Stages: raw (batched writes, gene set from the stream) -> conversion (byte passthrough for
every record that is not essentiality or synthetic lethality) -> aggregation (byte-keyed
gene-set grouping, guarded) -> processed -> indices and label table. `--genes` runs a smoke
build on a gene list; `--root` overrides the dataset root. Launched by
`scripts/gh_query_build_001.slurm` (40 CPUs, 160 GB, 3 days) from a detached worktree.
