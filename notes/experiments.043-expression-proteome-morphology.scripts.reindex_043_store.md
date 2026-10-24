---
id: wbbcf1keaxbkm8jgxxo9t7z
title: Reindex_043_store
desc: ''
updated: 1792426460521
created: 1792426460521
---

## 2026.10.19 - Index recomputation without a rebuild

Recomputes the index files of the built 043 store from its processed LMDB after the
stage 3 fixes. The fixed sites are the segregant reads in the index passes of
`torchcell/data/neo4j_cell.py` and the shared-label read in the processors. The script
then reads the first sample of every dataset to check that the phenotype value count
equals the label length. `Neo4jCellDataset` skips `process()` when `processed/lmdb` exists,
and each index property recomputes only when its json is absent. The stale
`perturbation_count_index.json` was moved to
`/scratch/projects/torchcell-scratch/deprecated/2026.10.19/raw_043-index-stale/` before
the run. Launcher: [[experiments.043-expression-proteome-morphology.scripts.gh_reindex_043_store]].
Output: `results/reindex_043_store.json`.

## 2026.10.19 - Job 5861 result

Slurm job 5861 COMPLETED in 36 min 48 s and wrote `results/reindex_043_store.json`.

- `perturbation_count_index.json` covers 173,617 of 173,617 entries (it covered 91,150
  before the fix). Count 0 holds 82,468 entries: the 82,467 Boocock and Nguessan
  segregants plus the one Muenzner 2024 proteome strain with no perturbation. The other
  counts are 1: 89,285 and 2: 72, then the isolates.
- `dataset_name_index` and `phenotype_label_index` each cover 173,617 entries.
- `is_any_perturbed_gene_index` has 8,112 genes and covers 91,149 entries, which is every
  entry with at least one perturbation (173,617 - 82,468). `is_any_deletion_gene_index`
  has 4,908 genes and covers 18,404 entries. Both map genes to entries, so an entry with
  no perturbed or deleted gene is absent by construction.
- Samples: for the first record of each of the 15 datasets, the number of phenotype values
  equals the label length. Kemmeren (index 0) now has 6,169, where it had 12,338 before
  the shared-label fix. Nadal-Ribelles (index 92,308) has 5,833 of 5,833.
  `phenotype_types` now has 8 unique entries.
