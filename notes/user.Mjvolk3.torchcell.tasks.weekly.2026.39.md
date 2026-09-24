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

- [x] `wt-cleanup-sweep` worktree: a sweep that removes worktrees whose work already landed, run after every `/enqueue-merge`, because 28 had piled up beside the primary checkout [[user.Mjvolk3.torchcell.tasks.weekly.2026.39.wt-cleanup-sweep]] [[scripts.wt_cleanup]]
- [x] SM media sourced from the papers (liquid, agar, and Zelezniak's recipe-deferred variant with no grams copied in) replace a zero-component stub, so two papers' synthetic minimal media no longer join on the string "SM" alone (#143) [[torchcell.datamodels.media-components#20260923---sm-media-formulations-issue-143]]
- [x] Caudal 2024: `SACE_` kept as part of the isolate code, since stripping it would silently lose 78 of 943 isolates, and isolate accounting now fails loudly (#73) [[torchcell.datasets.scerevisiae.caudal2024#20260923---sace_-header-prefix-and-loud-isolate-accounting-issue-73]]
- [x] Cachera 2023: the varying deletion stores its common name through the shared genome helper, so one gene keeps one spelling across datasets (#195) [[torchcell.datasets.scerevisiae.cachera2023]]
- [x] Issue 92 closed: the 11 datasets that failed L0 all pass L0-L4 after the 09.14 rebuilds, with a regenerable table from `scripts/verify_datasets.py` [[torchcell.verification.runners#20260923---issue-92-closed-by-rebuilds-all-11-datasets-pass-l0-again]]
- [x] Lint gate green again: two gated-experiment scripts landed unlinted (025 graph regularization, 029 closure), fixed, and pre-commit now lints experiments 016 and up locally, as CI does [[experiments.025-solid-growth.scripts.graph_regularization_how_not_what]]

## 2026.09.24

- [ ] The table note and the SI caption still say "pre-build ... not yet a versioned Neo4j DB build"; all 51 have been served as release 1.0 since 2026.09.17, so that framing needs the author's call [[torchcell.knowledge_graphs.releases]]
- [ ] Radiant VM: request a ~2 T block volume from NCSA; the NFS-backed store faults on every read (system database included) and Neo4j does not support NFS [[plan.kg-releases.2026.09.19]]
- [ ] Delete `/bulk/deprecated/2026.09.19/kg-releases-partial-backup-workdir` (373 G, the killed first backup) and `/db/deprecated/2026.09.19/kg-releases` if still present
