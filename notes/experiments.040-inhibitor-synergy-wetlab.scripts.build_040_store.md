---
id: vpwdtaa173p21epwh668eyc
title: Build_040_store
desc: ''
updated: 1792470006762
created: 1792470006762
---

## 2026.10.19 - One entry per served record, with a manifest

Builds `$DATA_ROOT/data/torchcell/experiments/040-inhibitor-synergy-wetlab/mixture_040` as a
`Neo4jCellDataset` over the 040 query: `converter=None`, `deduplicator=None`,
`RecordAggregator` (every record its own group, keyed by its content hash, the 043 build's
device: `aggregator=None` writes a processed store no reader can open), `Perturbation()`
processor, the trainer's three gene graphs so `dataset[i]` is a cell, `fetch_workers=8` with
two-character `e.id` partitions. The 033 build folded repeated measurements of a (genotype,
environment) cell with its own aggregator; here the fold is the trainer's read-time label policy.

The build runs on the release's own commit: the slurm launcher
([[experiments.040-inhibitor-synergy-wetlab.scripts.gh_build_040_store]]) points `PYTHONPATH`
at the `build/kg-4-0-v1-8-0` worktree (1.8.3), the package KG 4.0 is paired with, and the
script asserts torchcell resolved from `TORCHCELL_SRC`. A `python -c` check in the launcher
cannot do that: run from the submit directory, it resolves that directory's own torchcell
first (slurm 5905).

Writes `processed/store_manifest.json`
([[experiments.040-inhibitor-synergy-wetlab.scripts.store_manifest_040]]): release id and
version from the `KgRelease` node, torchcell commit and version imported, query file and
sha256, gene set size, returned against served per dataset, processed entry count, and the
size and sha256 of `processed/lmdb/data.mdb`. A full build asserts returned equals served for
every dataset. The census `results/build_040_store_census_<tag>.json` adds the index counts
and, per dataset, the first record's compounds, doses and what `dataset[i]` returns.

Smoke: `TC040_GENES` (space- or semicolon-separated; sbatch splits `--export` on commas, which
is how slurm 5905 received one gene of five) and `TC040_STORE_TAG=mixture_040_smoke`.
