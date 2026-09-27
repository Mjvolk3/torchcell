---
id: gv6lgh0a97zxkcuyloncgqk
title: Test_kuzmin2018_synthetic
desc: ''
updated: 1790482085191
created: 1790482085191
---

## 2026.09.26 - Hermetic builds of the five Kuzmin 2018 loaders from a three-row raw table

Tests [[torchcell.datasets.scerevisiae.kuzmin2018]] end to end without `$DATA_ROOT`, a genome, or the network. A hand-written `aao1729_data_s1.tsv` (the eleven columns the loaders read) is placed under `<tmp_path>/<ClassName>/raw/`, so PyG skips `download()`; `process()` runs once and `@post_process` writes `gene_set.json`, `experiment_reference_index.json` and `build_manifest.json`. Every record read back through `ds[i]` is validated into the loader's `experiment_class` / `reference_class` and compared by `model_dump()` equality against pydantic objects built by hand.

Fixture (shared by every test): two digenic rows of query `YAR002W+YDL227C_tm3180` (`nup60Δ+hoΔ`) against `YAL048C_dma5203` (`gem1Δ`, combined 0.8103 / SD 0.0463, epsilon -0.05, p 0.2) and `YBR001C_tsa100` (`nth2-5001`, 0.7 / 0.03, -0.02, 0.5), plus one trigenic row of query `YAR002W+YML107C_tm2550` (`nup60Δ+pml39Δ`) against the same `gem1Δ` array (0.4 / 0.02, tau -0.1, p 0.01, query double-mutant fitness 0.5128). Array single fitness 0.95 and 0.8, digenic query single fitness 0.9.

Expected values pinned:

- Smf: 3 records, array singles first (gem1_delta 0.95, nth2-5001 ts 0.8) then the digenic query single (nup60_delta 0.9, strain id the full query strain); no SD on any single. Reference SD is `pd.Series([0.0463, 0.03, 0.02]).mean()` because the mean is taken before the digenic filter, so it includes the trigenic row.
- Dmf: 2 digenic crosses (se 0.02315, 0.015; `sample_sd`, n = 4, colony) then the query strain `tm2550` as the pair nup60_delta + pml39_delta at 0.5128 with no uncertainty. Reference SD is the digenic mean `(0.0463 + 0.03) / 2 = 0.03815`, se 0.019075. `record_kind` column in `data.csv` is `digenic_array_cross, digenic_array_cross, double_mutant_query_strain`.
- Tmf: 1 record, three perturbations, 0.4 / 0.02 (se 0.01); reference SD 0.02.
- Dmi: 2 edge-level records (-0.05 / 0.2, -0.02 / 0.5), reference interaction 0.0 with `graph_level` edge.
- Tmi: 1 hyperedge record (-0.1 / 0.01).
- Side files: sorted gene set, ONE reference-index entry with `member_indices == range(n)`, manifest `dataset_name` / `loader_class` = class name and `loader_module` = the module, `processed/interned` beside `processed/lmdb`. The raw LMDB record holds `$ref` pointers for the environment (name = media name) and the reference (name = dataset name) while the publication stays inline.
- Reopening a built root after deleting the raw file reads the same records (neither download nor process runs).

Findings (pinned as the code behaves):

- `SmfKuzmin2018Dataset.preprocess_raw` ends with `df[~df["Query single/double mutant fitness"].isna()]` on the concatenated array + query frame, so an array single-mutant record is dropped when the row it was read from has an empty QUERY fitness, even though its own "Array single mutant fitness" is present (`test_smf_drops_array_single_when_the_row_query_fitness_is_missing`).
- The "no ho" columns of a trigenic row glue both query genes together (`YAR002WYML107C`, `nup60_deltapml39_delta`) because the `+` is removed after the `hoΔ` / `YDL227C` deletion; Tmf never reads them, so records are unaffected.
- Smf and Dmf disagree on the reference noise: Smf averages the combined-mutant SD over all rows, Dmf over the digenic rows only.

Gates: `ruff format` / `ruff check`, `scripts/run-mypy.sh` (the pre-commit wrapper, `--follow-imports=silent`), `pytest` under `env -u DATA_ROOT` (the conftest sentinel stays absent), `scripts/test_quality_check.py`. A bare `mypy <file>` additionally reports 15 pre-existing `ndarray` type-argument errors inside imported `torchcell` modules, none in the test file. 12 test functions, 16 test cases with the reopen parametrization. Phase 5 of [[plan.test-suite-buildout.2026.09.25]].
