---
id: 1omwhaq8vk0y50a7hg39xta
title: Test_kuzmin2020_synthetic
desc: ''
updated: 1790482092754
created: 1790482092754
---

## 2026.09.26 - Hermetic builds of the five Kuzmin 2020 loaders from hand-written xlsx tables

Tests [[torchcell.datasets.scerevisiae.kuzmin2020]] end to end without `$DATA_ROOT`, a genome, or the network. Tables S1, S3 and S5 are written with pandas/openpyxl (`to_excel(startrow=1)`, so the header sits on the second row and the loaders' `skiprows=1` lands on it) under `<tmp_path>/<ClassName>/raw/`; PyG skips `download()`, `process()` runs once, and the post-process side files are written. Records are validated back into the loader's `experiment_class` / `reference_class` and compared by `model_dump()` equality.

Fixture: S1 holds the digenic cross `YAL015C+YDL227C_tm461` (`ntg1Δ+hoΔ`) x `YBL007C_dma91` (`sla1Δ`, 0.9695 / 0.0465, epsilon 0.01, p 0.6) and the trigenic row of `YAL015C+YOL043C_tm72` (`ntg1Δ+ntg2Δ`) x `sla1Δ` (0.85 / 0.05, tau -0.08, p 0.02, query fitness 1.0133). S3 holds the same digenic query x `YBR001C_tsa100` (`nth2-5001`, 0.7 / 0.03, -0.02, 0.5) and the trigenic row of `YAL015C+YBR001C_tm99` (`ntg1Δ+nth2-5001`) x `YCR002C_sn1` (`cdc10-1`, an array strain that is neither `dma` nor `tsa`; 0.6 / 0.01, -0.15, 0.001, query fitness 0.77). S5 holds single mutants NTG1 (`delta`, sn123, 0.98 / 0.01), NTH2 (`nth2-5001`, sn124, 0.8 / 0.02) and CDC10 with no fitness (dropped), plus the double mutant `tm72` (1.0133 / 0.008); `tm99` has no S5 row.

Expected values pinned:

- Smf: 2 records; `Allele1 == "delta"` gives a KanMX deletion, otherwise an SGA allele; perturbed gene name is `Gene1` verbatim (`NTG1`, `NTH2`); se 0.005 and 0.01; reference fitness 1.0 with no SD.
- Dmf: 4 records, digenic crosses first (se 0.02325, 0.015, `sample_sd` n = 4) then `tm72` as the query pair ntg1 + ntg2 with S5's 1.0133 / 0.008 labeled `bootstrap_se` n = 12 (se 0.008, undivided), then `tm99` falling back to the S1/S3 column 0.77 with the KanMX + allele pair. `record_kind` in `data.csv` is two `digenic_array_cross` then two `double_mutant_query_strain`. S5 reduced to its single mutants raises `ValueError("... matched no trigenic query strain ...")`.
- Tmf: 2 records (0.85 / 0.05 and 0.6 / 0.01), the unknown array type stored as `SgaAllelePerturbation`; reference fitness 1.0 with `fitness_std = (0.05 + 0.01) / 2` and no uncertainty label.
- Dmi: 2 edge records (0.01 / 0.6, -0.02 / 0.5). Tmi: 2 hyperedge records (-0.08 / 0.02, -0.15 / 0.001) whose genotypes equal Tmf's.
- Side files as for 2018: sorted gene set, one reference-index entry covering `range(n)`, manifest naming the class and `torchcell.datasets.scerevisiae.kuzmin2020`, `processed/interned`, the Kuzmin 2020 publication on every record; a reopen after deleting the three tables reads the same records.

Findings (pinned as the code behaves):

- The `tm99` fallback record stores `fitness_std = nan` (a float), not `None`: `_double_mutant_query_strain_rows` masks the SD with `where(Fitness.notna())`, which yields NaN, and `create_experiment` passes `row["fitness_std"]` straight into `FitnessPhenotype`; only the uncertainty fields go through `pd.isna`. The value never compares equal to itself, so the test pins it with `math.isnan` and the reopen test compares records through sorted JSON.
- `SmfKuzmin2020Dataset` labels the S5 "St.dev." as `sample_sd` over 4 colonies via `_combined_mutant_uncertainty`, while the module's own `N_SAMPLES_QUERY_STRAIN_FITNESS` comment quotes the SI calling the query-strain fitness SD a bootstrap quantity over 12-24 colonies, and `DmfKuzmin2020Dataset` labels the same S5 column `bootstrap_se` n = 12 for its double-mutant rows.
- `DmfKuzmin2020Dataset.preprocess_raw` computes `self.phenotype_reference_std` (0.03825 here) but `create_experiment` never receives it; the stored reference is fitness 1.0 with no SD.
- `TmfKuzmin2020Dataset` and `TmiKuzmin2020Dataset` tag the query perturbations with the SPLIT halves of the query strain id (`YAL015C`, `YOL043C_tm72`), not the full id that the Dmf query-strain records and every 2018 loader use.
- `TmfKuzmin2020Dataset` stores `fitness_std` with no `fitness_uncertainty` labeling, so `fitness_se` is `None` on every record and on the reference, unlike `TmfKuzmin2018Dataset`.

Gates: `ruff format` / `ruff check`, `scripts/run-mypy.sh`, `pytest` under `env -u DATA_ROOT` (sentinel absent afterwards), `scripts/test_quality_check.py`. 9 test functions, 13 test cases with the reopen parametrization. Phase 5 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.10.01 - Issue #533 Findings Retired

- Retired: the Dmf query-strain fallback record's `fitness_std` NaN; it is now asserted None and compared in the full dump. The tm99 trigenic array is now the ts strain `YCR002C_tsa1` (an unknown array is refused, pinned in `test_kuzmin2020.py`).
