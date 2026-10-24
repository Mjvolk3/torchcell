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

## 2026.10.19 - Dev-tree census

`python experiments/043-expression-proteome-morphology/scripts/store_census.py --source dev`
(run 2026.10.19 on GilaHyper, about 6 min), output
`experiments/043-expression-proteome-morphology/results/store_census_dev.json`. The raw
census (`--source raw`) has not run: build job 5851 failed before the raw LMDB was written
([[experiments.043-expression-proteome-morphology.scripts.build_043_store]]).

| dataset | records | genotypes | perturbation sets | environments | media | label | label length min / median / max | perturbations per record |
|---|---|---|---|---|---|---|---|---|
| Boocock2025Dataset | 79,516 | 79,516 | 79,516 | 2 | 2 | umi_count | 182 / 962 / 4758 | 0 to 0 |
| CaudalPanTranscriptome2024Dataset | 943 | 943 | 943 | 1 | 1 | expression_tpm | 5922 / 6050 / 6120 | 195 to 7025 |
| DmMicroarraySameith2015Dataset | 72 | 72 | 72 | 1 | 1 | expression_log2_ratio | 6169 / 6169 / 6169 | 2 to 2 |
| MicroarrayKemmeren2014Dataset | 1,484 | 1,484 | 1,484 | 1 | 1 | expression_log2_ratio | 6169 / 6169 / 6169 | 1 to 1 |
| MicroarrayScoreHu2007Dataset | 263 | 263 | 263 | 1 | 1 | expression_score | 5921 / 6381 / 6429 | 1 to 1 |
| NadalRibellesPerturbSeq2025Dataset | 6,188 | 3,095 | 3,150 | 2 | 1 | expression_log2_ratio | 4223 / 5866 / 6313 | 1 to 1 |
| Nguessan2025Dataset | 2,951 | 2,951 | 2,951 | 1 | 1 | umi_count | 475 / 2592 / 5111 | 0 to 0 |
| PooledPromoterExpressionVaishnav2022Dataset | 71,158 | 1 | 71,158 | 1 | 1 | expression_level | 1 / 1 / 1 | 1 to 1 |
| ProteinTurnoverMuenzner2024Dataset | 55 | 55 | 55 | 1 | 1 | degradation_rate | 919 / 1055 / 1177 | 4903 to 6430 |
| ProteomeMessner2023Dataset | 4,699 | 4,549 | 4,549 | 1 | 1 | protein_abundance | 1441 / 1819 / 1849 | 1 to 1 |
| ProteomeMuenzner2024Dataset | 795 | 795 | 795 | 1 | 1 | protein_abundance | 1336 / 1540 / 1571 | 0 to 7025 |
| ProteomeZelezniak2018Dataset | 97 | 97 | 97 | 1 | 1 | protein_abundance | 726 / 726 / 726 | 1 to 1 |
| ScmdOhya2005Dataset | 4,718 | 4,718 | 4,718 | 1 | 1 | calmorph | 281 / 281 / 281 | 1 to 1 |
| SmMicroarraySameith2015Dataset | 82 | 82 | 82 | 1 | 1 | expression_log2_ratio | 6169 / 6169 / 6169 | 1 to 1 |
| TfDeletionPseudobulkJackson2020Dataset | 784 | 12 | 72 | 10 | 9 | umi_count | 298 / 4028.5 / 5561 | 1 to 1 |

Total 173,805 records, equal to the release's served total. Perturbation types are in the
JSON (isolates: `sequence_variant`, `natural_loss_of_function`, `natural_gene_absence`,
`natural_gene_presence`, `deletion`).

Readings against the reviewer's dev-store census:

- Jackson 2020: 12 genotypes and 9 media names, as expected; 10 environments under this
  key against the reviewer's 11 environment refs. This key (media + temperature +
  environmental perturbations) ignores the other `Environment` fields, so two refs that
  differ only elsewhere count once here (which field differs is not checked).
- Nadal-Ribelles 2025: 3,095 genotypes over 2 environments, as expected.
- Isolates: Caudal 195 to 7,025 perturbations per record, Muenzner turnover 4,903 to 6,430,
  including `sequence_variant`.
- Muenzner 2024 proteome has one record with zero perturbations in the dev store; the
  `any` gene clause is false on an empty list, so hypothesis (untested): the query returns
  794 of its 795 served records.
- Label names: `expression_log2_ratio` spans Kemmeren, Sameith Sm and Dm, and
  Nadal-Ribelles; `umi_count` spans Jackson, Boocock and Nguessan; `protein_abundance`
  spans Messner, Zelezniak and Muenzner proteome.
- Vaishnav 2022 collapses to 1 gene-name genotype (every member carries the same
  reporter); the perturbation-set key separates all 71,158.

## 2026.10.19 - Raw census of the built store

`--source raw` over the job 5855 raw LMDB (173,617 records), written to
`results/store_census.json`. Every dataset matches the dev census in genotypes,
environments, media, labels, label lengths and perturbation types, except the
`all`-clause drops below. Records, then genotypes, raw against dev:

- Kemmeren: 1,482 records against 1,484; genotypes 1,482.
- Hu: 262 records against 263; genotypes 262.
- Nadal-Ribelles: 6,118 records against 6,188; genotypes 3,059 against 3,095.
- Messner: 4,624 records against 4,699; genotypes 4,476 against 4,549.
- Ohya: 4,678 records against 4,718; genotypes 4,678.

Muenzner proteome now returns 795, including the zero-perturbation strain. Each gap is
records whose gene is outside the genome's 6,607-gene set; the count is in
[[experiments.043-expression-proteome-morphology.scripts.build_043_store]].
