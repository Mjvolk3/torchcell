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
