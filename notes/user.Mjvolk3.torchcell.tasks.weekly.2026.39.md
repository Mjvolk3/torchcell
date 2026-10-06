---
id: fvevd1vilz1zh43vcg33q8e
title: '39'
desc: ''
updated: 1790204613104
created: 1790204613104
---

## 2026.09.21

- [x] Superset admission: an already-served dataset whose loader only adds records is re-admitted incrementally once every served id is proven still produced, so the Kuzmin 2018/2020 dmf fix reached the graph without a full rebuild [[torchcell.knowledge_graphs.kg_manifest]] [[torchcell.knowledge_graphs.incremental-admission]]
- [x] KG slurm fixes found by those increments: build-tree conf/biocypher dirs chowned back after a rebuild, and a batch passed through the caller's env because `--export` splits on commas [[database.slurm.scripts.gilahyper_increment_kg-slurm_docker]] [[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]
- [x] `dataset-label-names` worktree: every supported-dataset name is `Author Year (descriptor)` [[user.Mjvolk3.torchcell.tasks.weekly.2026.39.dataset-label-names]]
- [x] `datasets-figure-labels` worktree: scatter labels beside their own markers via a deterministic slot placer [[user.Mjvolk3.torchcell.tasks.weekly.2026.39.datasets-figure-labels]]

## 2026.09.23
