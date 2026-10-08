---
id: 5n8zw71jnghwg2ok32xhu6j
title: Gilahyper_increment_kg Slurm_docker
desc: ''
updated: 1790204542489
created: 1790204542489
---

## 2026.09.21 - Superset admissions, and two launch pitfalls

This script adds one dataset, or a batch, to the served knowledge graph by incremental import, so a new or grown dataset does not force the full rebuild in [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]. The admission rules it enforces live in [[torchcell.knowledge_graphs.incremental-admission]] and [[torchcell.knowledge_graphs.kg_manifest]].

- Superset admission: the script passes the bolt URI to `kg_manifest admit`, so an already-served dataset whose loader now emits more records can be re-admitted. Limits raised to 12 h / 64 G for the first large superset, because the edge filter looks up every served edge (main `b87a5c739`).
- A batch goes through the caller's environment, `DATASET_CLASSES=A,B sbatch --export=ALL ...`, never inside `--export`'s list: sbatch splits that list on commas, so `--export=ALL,DATASET_CLASSES=A,B` handed job 2677 `DATASET_CLASSES=A` and it ran as a one-member batch (main `ab6d8c5d1`).
- The conf and biocypher directories are chowned back to the invoking user through a root container before `directory_setup`; a preceding full rebuild leaves them 7474-owned at mode 700, which killed job 2675 at stage 3 (main `1fbd6f883`).

## 2026.10.08 - Record the artifact pointer set before the stamp

Stage 7 runs `kg_manifest ... artifact-refs --data-root "$DEV_DATA_ROOT"` right before `releases stamp --kind incremental`. It re-walks every served dataset whose closure names `ArtifactRef`, not only the admitted members: a newly admitted (or superset) entry starts unrecorded, and one unrecorded entry would make the release's pointer set None. Cost, estimated and not yet measured on a build: the walk is 0.26 ms per record, and on main at `1b180c7e8` thirteen loaders' closures name `ArtifactRef` (bloom2019, caudal2024, lian2019, mormino2022, smith2016, choe2025, cui2018, rapp2026, rousset2018, wang2018, carruthers2025, menasalvas2025, yunus2026), so the step walks all of their records; Bloom 2019 alone (530,100 records) is about 2.3 min of walking plus the LMDB reads. The step runs after the import, under `set -euo pipefail`: a walk failure (an unregistered class, a missing dev LMDB) aborts before stamp, write-node and the archive, with the store already changed, so the recovery is to fix the dev tree and rerun from the stamp. The served entries are read from their dev-tree LMDBs, which can have moved since admission; the admission gate stops schema drift, not loader-data drift.
