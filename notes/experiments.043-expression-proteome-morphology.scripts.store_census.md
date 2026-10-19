---
id: dgg3r9f90l1t5skoqyvemro
title: Store_census
desc: ''
updated: 1792393319760
created: 1792393319760
---

## 2026.10.19 - Census of the store

`--source raw` reads the built raw LMDB of
[[experiments.043-expression-proteome-morphology.scripts.build_043_store]] (JSON records,
interned constants already spliced in) and writes `results/store_census.json`;
`--source dev` reads the 15 dev-tree loader LMDBs (pickled records, `$ref` pointers
resolved from `processed/interned`) and writes `results/store_census_dev.json`. The dev
census counts what the loaders built, not what KG 4.0 serves.

Keys: genotype = frozenset of `systematic_gene_name` (a `SegregantGenotype` by
`(cross, segregant_id)`); perturbation set = frozenset of per-perturbation content hashes
(added beside the genotype key: it separates genotypes that share gene names, such as the
Vaishnav promoter library whose members all carry the same reporter); environment = media
name + temperature + environmental perturbations.
