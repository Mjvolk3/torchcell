---
id: 873act1eu6pwdjskzv50211
title: 001_env_chemgen_pooled
desc: ''
updated: 1790549459641
created: 1790549459641
---

## 2026.09.27 - Four blocks, one filter

Four `UNION ALL` blocks, one per kept dataset, each the 025 block shape with the phenotype,
environment and media matches removed: `Dataset -> Experiment`, the genotype's perturbations
all in `$gene_set`, at least one perturbation, the experiment reference, and the two
`serialized_data` properties returned rather than nodes. Dataset ids are the loader class
names as the adapter map registers them (`EnvChemgenVanacloig2022Dataset`,
`HetHillenmeyer2008Dataset`, `EnvChemgenHoepfner2014Dataset`,
`EnvChemgenWildenhain2015Dataset`), confirmed against `kg_manifest.json` at release
2026.09.21-ab6d8c5d.

Two things a reader of the query should know:

- A Vanacloig genotype carries four perturbations (the queried deletion plus PDR1 YGL013C,
  PDR3 YBL005W and SNQ2 YDR011W), so a smoke gene set without those three returns no
  Vanacloig record. The smoke launcher includes them.
- The gene filter is the only place a record can be lost: a dataset gene absent from the
  S288C gene set drops its records. The build writes the returned experiments per dataset
  beside the served counts in `dataset_index_summary.json`, so the loss is a number rather
  than an assumption.
