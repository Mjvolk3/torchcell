---
id: r6aawe1k0k9tb5k9gtqrqvk
title: Natural_isolate_genotype_identity
desc: ''
updated: 1791808474406
created: 1791808474406
---

## 2026.10.12 - Do the stores build the shared module's genotypes?

`experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py` answers the identity question of #988 for the seven natural-isolate stores (`caudal_pantranscriptome2024`, `isolate_growth_peter2018`, `proteome_muenzner2024`, `protein_turnover_muenzner2024`, `doubling_time_dechiara2022`, `cell_yield_dechiara2022`, `life_cycle_dechiara2022`), reading each store's LMDB (interned genotypes resolved) once per isolate. The content id is the sha256 of the sort-keyed JSON of the python-mode genotype dump, the same bytes for a stored dict and a rebuilt one.

- `before`, on the stores the loaders built before the shared module: is every stored genotype equal to `NaturalIsolateGenotypes.base_perturbations` (the sequence variants, natural presences and natural absences, exactly what the four loaders built), and do two stores that share an isolate store one genotype? Output `results/natural_isolate_genotype_identity_before.json`.
- `after`, on the rebuilt stores: is every stored genotype equal to `NaturalIsolateGenotypes.genotype`, does every shared isolate keep one content id across stores, and what exactly differs from the superseded store's genotype of the same isolate, by perturbation type. Output `results/natural_isolate_genotype_identity_after.json`, with the shared module's ledger over the 1,011 genome isolates.

Run under slurm (`sbatch --wrap`, 4 CPUs, 64 GB): `PYTHONPATH=<worktree> python experiments/036-dataset-fixes-before-kg-build/scripts/natural_isolate_genotype_identity.py before|after`.

## 2026.10.12 - Results: identical before, identical after, and nothing removed

**Before** (slurm 4241, 8 min 46 s, 7.2 GB, on the stores built before the shared module): every isolate of all seven stores stores exactly `base_perturbations` (Caudal 943 of 943, Peter 971 of 971, Muenzner proteome 794 of 794, turnover 55 of 55, De Chiara 906, 906 and 905 of the same), and the 999 isolates held by two or more stores carry one content id each (1,004 isolates in any store). `results/natural_isolate_genotype_identity_before.json`.

**After** (slurm 4420, 23 min 34 s, from a snapshot of the branch at `15488460d`, the genotype code the stores were built with): every stored genotype equals `NaturalIsolateGenotypes.genotype` (all seven stores, every isolate, no mismatch), and the 999 shared isolates still carry one content id each. Against the generation each rebuild retired, per store, isolate for isolate, perturbations were only ADDED, none removed, so every perturbation the loaders stored before is byte-identical in the new genotype:

| store | isolates compared | HO deletions added | loss-of-function calls added | removed |
|---|---|---|---|---|
| caudal_pantranscriptome2024 | 943 | 116 | 361,624 | none |
| isolate_growth_peter2018 | 971 | 122 | 372,792 | none |
| proteome_muenzner2024 | 794 | 85 | 300,040 | none |
| protein_turnover_muenzner2024 | 55 | 4 | 20,525 | none |
| doubling_time_dechiara2022 | 906 | 124 | 349,533 | none |
| cell_yield_dechiara2022 | 906 | 124 | 349,533 | none |
| life_cycle_dechiara2022 | 905 | 48 | 348,265 | none |

The two Muenzner stores were read from their kept generations with `--processed` and `--superseded`, because the #1045 branch's array 4348 rebuilt both dev stores minutes after #1058 did: the #1058 builds are `processed.superseded.20261012-114150` (proteome) and `processed.superseded.20261012-114548` (turnover), compared with main's builds `processed.superseded.20261012-084201` and `processed.superseded.20261012-084426`. A first after run (slurm 4301) read the dev `processed/` of the proteome store while that rebuild was in flight and stopped on a genotype naming no isolate. Over the 1,011 genome isolates the module's ledger reads 124 HO deletions, 240,526 nonsense/SIFT calls on 4,090 genes and 147,310 frameshift calls on 2,455 gene keys (9,006 on rows released without a code). `results/natural_isolate_genotype_identity_after.json`.
