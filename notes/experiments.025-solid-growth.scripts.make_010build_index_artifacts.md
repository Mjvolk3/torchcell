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

## 2026.09.09 - The Delta disjoint arm was on the wrong partition

Two query-pair-disjoint partitions of the same 376,732 records exist, and the Delta arm read
the wrong one.

| Partition | Query pairs train/val/test | Records train/val/test | Baseline B1 |
| --- | --- | --- | --- |
| 010's own, `index_query_pair_disjoint_seed_42.json` | 285 / 67 / 68 | 301,483 / 37,569 / 37,680 | 0.174 |
| Arm Q, `query_pair_disjoint_splits_025.json.gz` | 331 / 43 / 46 | 301,236 / 37,705 / 37,791 | 0.185 |

Both assign the same 420 recurring pairs, differently. Arm Q's is the partition the six
additive nulls were refit on and the one the GilaHyper disjoint run trains on, so a Delta
result on 010's own split would have compared to neither.

This script now carries arm Q's partition into 010-build index space instead of reading 010's
own. The carry needs no index correspondence and no LMDB pass over the 3.2 TB build, because
the arm Q artifact stores `pair_assignment`, a query-pair-name to fold map that is index free.
The script reads each 010 record's genotype, regroups by query pair with the rule
`subset_definitions.query_pair_split` uses, including the most-frequent-pair tie break, and
reads the fold from that map.

Two assertions make the transfer checkable rather than assumed. The recurring pair set
computed from the 010 gene sets must equal arm Q's 420 pairs, and the resulting fold sizes
must equal arm Q's exactly. Both hold: the run prints 301,236 / 37,705 / 37,791. They must
hold, the two record populations being genotype-identical, so a mismatch would mean the
populations had drifted.

Output renamed to `query_pair_disjoint_splits_010build_armq.json.gz`, since the old name said
only "seed 42" and both partitions carry seed 42. `cgt_010b_q_kl_007.yaml` and
`delta_preflight_025.sh` follow the rename, and the stale artifact is removed.

Nothing was running on the wrong split. The whole Delta sweep, including the three disjoint
jobs 21917136 to 21917138, was cancelled 2026-09-09T18:57:08 with zero elapsed time.

Also fixed here: `gh_cgt.slurm` defaulted to `cgt_s0_r_kl_000` when no config argument
arrived. That turns a dropped argument into a silently different experiment, and the default
was the random-split arm, which is exactly the wrong partition for a job named for the
disjoint one. The launcher now refuses without a config and checks the file exists, matching
`delta_cgt.slurm`.
