---
id: 2r2uj3yzar7dlf2j1aq2eia
title: Flatten_records
desc: ''
updated: 1790375594797
created: 1790375594797
---

## 2026.09.25 - One row per served record, every cell and environment axis as a column

Reads the dev-tree LMDB builds of `EnvChemgenVanacloig2022Dataset`,
`HomHillenmeyer2008Dataset` and `HetHillenmeyer2008Dataset` through their loaders and
writes `experiments/031-env-chemgen-inhibitor-tolerance/results/records_<dataset>.parquet`.
The served pydantic record is the source, not the raw matrix, so the columns are exactly
the typed axes the schema carries: the genotype (systematic genes, perturbation types,
reference strain, ploidy), the environment (medium name / base / state / synthetic flag /
dropouts, temperature, aerobicity, duration in hours and in generations, every dosed
compound with its InChIKey, PubChem CID, dose value, unit, basis and solvent, every
physical factor), the readout (measurement type, assay type, response, SE, uncertainty
type, `n_samples`, sample unit, `screen_id`, units string) and the same environment axes
for the reference. Multi-valued fields are `|`-joined so one record stays one row.

Runtime on GilaHyper: about 3,300 to 5,300 records per second single-threaded, so the
three stores (143,218 + 1,088,620 + 2,698,797 records) take roughly 15 minutes. `--limit`
gives a smoke run; `--datasets` selects a subset.

The two downstream scripts read only these parquet files:
[[experiments.031-env-chemgen-inhibitor-tolerance.scripts.dataset_axes_comparison]] and
[[experiments.031-env-chemgen-inhibitor-tolerance.scripts.cross_dataset_similarity]].
