---
id: afh408rsahkuw5aq04de74i
title: Make_010build_index_artifacts
desc: ''
updated: 1788922193336
created: 1788922193336
---

## 2026.09.08 - Index artifacts for running the 025 script on the 010 build

Writes three gzipped JSON artifacts into `experiments/025-solid-growth/results/` in 010's own
record-index space (0 to 376,731), from frozen inputs only:

- `subset_010build_all_indices.json.gz`: every record (the "subset" is the whole build).
- `pinned_splits_010build_seed_42.json.gz`: `{pinned: {train 301,386, val 37,673, test 37,673}}`,
  read from the 010 build's `data_module_cache/index_seed_42.json`.
- `query_pair_disjoint_splits_010build_seed_42.json.gz`: `{splits: {train 301,483, val 37,569,
  test 37,680}}`, read from `experiments/010-kuzmin-tmi/results/index_query_pair_disjoint_seed_42.json`,
  the partition behind Table 10 of the additive-baselines report (285/67/68 query pairs).

Each is checked to be an exact partition of the 376,732 records before writing. Consumed by
`cgt_010b_*` configs through `subset.indices` / `subset.split_file` / `subset.split_key`; see
[[experiments.025-solid-growth.scripts.delta_cgt]].
