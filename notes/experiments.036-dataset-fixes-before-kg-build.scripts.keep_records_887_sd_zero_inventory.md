---
id: lj58smw6z83og4nja96vbtl
title: Keep_records_887_sd_zero_inventory
desc: ''
updated: 1791806038623
created: 1791806038623
---

## 2026.10.12 - What the #887 and SD-of-zero decisions keep

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/keep_records_887_sd_zero_inventory.py`; run under slurm by [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_keep_records_887_sd_zero]] (job 4201, 2026-10-12, 4.9 GB max RSS). Output: `experiments/036-dataset-fixes-before-kg-build/results/keep_records_887_sd_zero_inventory.json`. Reads only pinned bytes and dev stores; writes no store.

It measures, for the owner decisions of 2026-10-12 (never drop a released record; store what the source reports, "the data is the data!"; store the uncertainty a source releases exactly as released):

- **Xue 2025 (#887)**: the workbook's sha256 against the loader pin, every cell of both sheets (20 + 2,783), the genotype statement for RGR1, MED4 and TFC7, an allele search over every cell (RGR1, MED4, TFC7, truncation, DAmP, degron, conditional, hypomorph, partial), records per gene (46 each; 112 with any; 83 with RGR1 or MED4), the SGD null `inviable` entries from the cached locus records, the `gene_essentiality_sgd` keys and Ohya 2005 membership, and the dev store (176 records, 138 `kanmx_deletion` perturbations of the three). It raises if the store and the release disagree per gene.
- **Released SDs of 0 (#894, #895, #897, #930)**: per released table, the zero-SD rows by combined-mutant type with the released SD and P-value text on them (Kuzmin 2018: 4 digenic + 5 trigenic; Kuzmin 2020 S1: 18 + 6, S3: 15 + 7; Costanzo 2016: 115, 230, 1,356, 1,080), how the Kuzmin dev stores hold them (4, 5, 33 records, each `sample_sd`, `n_samples` 4, `fitness_se` 0.0), the verbatim SI column definitions checked against the bytes, and every SI and main-text line matching "standard deviation", "not computed", "not calculated", "undefined", "zero", "single colony" or "pseudocount". `--costanzo-store` also reads the 20.7 M-record `dmf_costanzo2016` store (not run: another branch rebuilds it).

Findings are written up in [[torchcell.datasets.scerevisiae.xue2025]], [[torchcell.datasets.scerevisiae.kuzmin2018]] and [[torchcell.datasets.scerevisiae.kuzmin2020]] (2026.10.12 sections).

`--after` censuses the two dev stores PR #1054 rebuilds (`smf_kuzmin2018`, `dmf_kuzmin2018`: record count, typed gaps by field, uncertainty labels, released zeros) into `results/keep_records_887_sd_zero_stores_after.json`; it runs at the end of [[experiments.036-dataset-fixes-before-kg-build.scripts.gh_keep_records_887_sd_zero_rebuild]].
