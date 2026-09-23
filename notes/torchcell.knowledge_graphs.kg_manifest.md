---
id: yw9q0vqh6p42zzi2gvmn33b
title: Kg_manifest
desc: ''
updated: 1790204534064
created: 1790204534064
---

## 2026.09.21 - Admit a served dataset as a proven superset

The admission gate exists so a dataset can join the served graph without a full rebuild, but it blocked any dataset already served. That turned a loader fix that only adds records into a full rebuild: the Kuzmin 2018/2020 dmf query-strain fitnesses (+172 and +201, every existing record byte-identical, main `4c4a4f950`). A served dataset is now re-admissible when it provably only grows.

- The check reads the live store's experiment ids under the dataset's `Dataset` node over bolt through `ExperimentMemberOf` (no property index needed), then walks the dev LMDB through the adapter's own id path (`transform_item`, then the sha256 of the json-dumped `model_dump` that `_experiment_node` uses; a test pins the two).
- Admissible only if every served id is still produced and at least one new id exists. A served id the LMDB no longer produces blocks, because that is a changed record and the full-rebuild case; an identical re-admission blocks as nothing to add.
- Manifest: `KgDatasetEntry.superset_of` records the previous entry and `n_added`; the event kind is `superset_admission`. CLI: `admit --neo4j-uri --database`.
- Measured runs and the resulting served counts (410,571 and 632,998): [[torchcell.knowledge_graphs.incremental-admission]]. Slurm side: [[database.slurm.scripts.gilahyper_increment_kg-slurm_docker]].
