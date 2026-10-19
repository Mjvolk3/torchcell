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
