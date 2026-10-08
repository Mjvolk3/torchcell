---
id: xmmrc8oy4ek7jrn3p9tggky
title: ops-artifact-pointers
desc: ''
updated: 1791439565470
created: 1791439565470
---

## 2026.10.08

- [x] `make ops` reports the flat files the served graph points at: the KG manifest records each dataset's file-level `ArtifactRef` set at the stamp (`kg_manifest artifact-refs`, [[torchcell.knowledge_graphs.kg_manifest]]), the `KgRelease` node and the snapshot carry it ([[torchcell.knowledge_graphs.releases]], [[torchcell.knowledge_graphs.release_snapshot]]), and `releases artifacts` checks every pointer against tc-data's manifests for the `artifacts` line in both host blocks plus a `tc-data` health line for Radiant ([[scripts.ops]]); both slurm build scripts run the recording step before the stamp ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]], [[database.slurm.scripts.gilahyper_increment_kg-slurm_docker]]). Owner items before the line can read ✓ from GilaHyper: the OpenStack rule for 8724 and a `TC_DATA_API_KEY` in GilaHyper's `.env` ([[database.tc-data-endpoint]]).
