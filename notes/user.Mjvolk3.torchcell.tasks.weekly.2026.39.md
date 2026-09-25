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

- [ ] The table note and the SI caption still say "pre-build ... not yet a versioned Neo4j DB build"; all 51 have been served as release 1.0 since 2026.09.17, so that framing needs the author's call [[torchcell.knowledge_graphs.releases]]
- [ ] Radiant VM: request a ~2 T block volume from NCSA; the NFS-backed store faults on every read (system database included) and Neo4j does not support NFS [[plan.kg-releases.2026.09.19]]
- [ ] Delete `/bulk/deprecated/2026.09.19/kg-releases-partial-backup-workdir` (373 G, the killed first backup) and `/db/deprecated/2026.09.19/kg-releases` if still present

## 2026.09.25

- [x] 031 inhibitor tolerance: Vanacloig 2022 vs Hillenmeyer 2008 HOM and HET compared axis by axis on the served records (background strain, ploidy, medium, pH, temperature, aerobicity, duration, dose basis, statistic), overlaps by gene and InChIKey, reliability ceilings, and the shared response structure; HOM is the partner, MMS is the only compound that transfers, isobutanol is measured only in Vanacloig [[experiments.031-env-chemgen-inhibitor-tolerance]]
- [x] scripts: [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.flatten_records]] [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.dataset_axes_comparison]] [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.cross_dataset_similarity]]
- [ ] 031 training config: Vanacloig alone vs Vanacloig + HOM, compound-held-out split, per-compound reliability beside every score
