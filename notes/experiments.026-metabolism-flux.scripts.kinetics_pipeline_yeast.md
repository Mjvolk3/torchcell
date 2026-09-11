---
id: ubyh0c70uihq78emyqmeh3f
title: Kinetics_pipeline_yeast
desc: ''
updated: 1789100312373
created: 1789100312373
---

## 2026.09.10 - The portable pipeline, run on yeast-GEM 9.0.2

GEM-specific glue for [[torchcell.metabolism.kinetics_pipeline]]: it writes the gene-to-
accession map out of the `swissprot.tsv` yeast-GEM ships and points the config at the model
SBML, the genome protein dump, the MetaNetX mirror, the lipid structures, the Open Enzyme
Database slice and the predictor parquets. Porting to another organism replaces this file,
not the module.

Measured 2026.09.10 with the accession key: 148 measured / 3,570 predicted / 10 default for
both parameters, identical to the flux layer's own path. With `ec_substrate_organism`: 145
measured. With `ec_substrate_any_organism`: 869 measured on the cross-organism homolog tier,
23.3 percent of the model, which is the estimate of what an unassayed organism would get.

Writes `results/kinetic_pipeline_yeast_<key>.json`. Full accounting:
[[experiments.026-metabolism-flux.kinetic-coverage-closure]].

## 2026.09.11 - Reproduces the shipped tables exactly

Default key is now `uniprot_substrate` with an `ec_substrate_any_organism` homolog stage, and
the predictor tables read are the verified version. Measured 2026.09.11: 96 measured_protein
/ 782 measured_ec_other_organism / 2,820 predicted / 30 default for both parameters, default
9.6 1/s and 0.1467 mM, identical on every count to `resolve_kcat_table` and
`resolve_km_table`. With `--measured-key ec_substrate_any_organism --no-homolog-tier`, the
cross-organism tier alone reaches 869 units, which is the estimate for a chassis with no
assays of its own.
