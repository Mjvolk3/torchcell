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
