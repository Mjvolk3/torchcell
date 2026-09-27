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

## 2026.09.27

- [x] 033: the pooled four-dataset chemogenomic build from the served graph (Vanacloig, Hillenmeyer HET, Hoepfner, Wildenhain; HOM dropped by decision), query, build script and launchers; aggregated by the (genotype, environment) cell with a new aggregator, since the gene-set key would collapse every condition of a gene into one entry [[experiments.033-env-chemgen-pooled]] [[experiments.033-env-chemgen-pooled.queries.001_env_chemgen_pooled]] [[experiments.033-env-chemgen-pooled.scripts.query]] [[torchcell.data.genotype_environment_aggregate]]
- [ ] 033 build 001 on GilaHyper: smoke on eight genes (slurm 2928), then the full build from the detached `033-build` worktree; record the per-dataset returned counts against the served counts and the measurements-per-entry histogram
