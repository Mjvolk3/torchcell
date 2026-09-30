---
id: 6isp06dg9ji3nytuifenhau
title: Test_costanzo2016
desc: ''
updated: 1790481841271
created: 1790481841271
---

## 2026.09.26 - Hermetic builds of the SMF, DMF, and DMI loaders

Test file: `tests/torchcell/datasets/scerevisiae/test_costanzo2016.py` (18 test functions, 33 cases with parametrization). Source under test: [[torchcell.datasets.scerevisiae.costanzo2016]]. Every build runs end to end over hand-written raw files placed in `<root>/raw/`, so PyG skips `download()`; `process()` runs once and `@post_process` writes `gene_set.json`, `experiment_reference_index.json`, and `build_manifest.json`. No network, no `$DATA_ROOT`, no genome. The three module-scoped builds plus the two one-off builds take about 4.5 s together.

### SMF fixture and expected values

Six raw rows written with `to_excel` under the seven source column names (degree sign included, `Single mutant fitness (26°)`):

| Strain ID | systematic | allele | f26 | sd26 | f30 | sd30 | role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YAL001C_tsa1 | YAL001C | tfc3-1 | 0.85 | 0.02 | 0.60 | 0.04 | ts allele |
| YAL002W_dma1 | YAL002W | VPS8 | 0.95 | 0.01 | 0.90 | 0.02 | KanMX |
| YAL003W_sn1 | YAL003W | EFB1 | NaN | NaN | 0.70 | 0.03 | NatMX, 26 C row dropped by dropna |
| YAL002W_dma1 | (exact duplicate of row 2) | | | | | | dropped by drop_duplicates |
| YAL005C_damp1 | YAL005C | ssa1-damp | 0.80 | 0.03 | 0.75 | 0.05 | DAmP |
| YAL007C_S1 | YAL007C | erp2-S1 | 1.05 | 0.04 | 1.02 | 0.06 | suppressor |

Derived expectations, all pinned:

- 9 records in the order `concat([26 C rows, 30 C rows])`: indices 0..3 are the 26 C rows (tsa1, dma1, damp1, S1), indices 4..8 the 30 C rows (tsa1, dma1, sn1, damp1, S1).
- Reference std = mean stddev per temperature: 26 C `(0.02+0.01+0.03+0.04)/4 = 0.025`, 30 C `(0.04+0.02+0.03+0.05+0.06)/5 = 0.04`. Both typed `bootstrap_se`, so `fitness_se == fitness_std` on every record and reference; `n_samples = 17` (`N_SAMPLES_QUERY_SMF_SCREENS`), `sample_unit = screen`.
- Record 0 and record 6 are compared to hand-built `FitnessExperiment` / `FitnessExperimentReference` / `Publication` objects by `model_dump` equality (media `SGA_DM_SELECTION`, PMID 27708008, DOI 10.1126/science.aaf1420).
- Side files: `preprocess/` holds exactly `build_manifest.json, data.csv, experiment_reference_index.json, gene_set.json`; gene set is the five names sorted; two references with member indices `[0,1,2,3]` (26 C) and `[4..8]` (30 C); the manifest records `dataset_name = "smf"` (basename of root), the loader class and module, and `socket.gethostname()`; `processed/interned` holds exactly 4 objects (2 environments + 2 references; the publication is under 512 bytes and stays inline). `data.csv` has 9 rows and 7 columns (`Strain_ID_suffix` is not carried into the melted frame).

### SGA (DMF and DMI) fixture and expected values

Five tab-separated rows across the four required files, under the eleven source column names (`Genetic interaction score (ε)` included):

| file | query | array | Arraytype/Temp | eps | p | DMF | DMF sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SGA_DAmP.txt | YAL001C_damp1 (tfc3-damp) | YAL002W_dma1 (vps8) | DMA30 | -0.12 | 0.001 | 0.70 | 0.05 |
| SGA_ExE.txt | YAL003W_tsq1 (efb1-1) | YAL005C_tsa1 (ssa1-1) | TSA26 | 0.08 | 0.02 | 0.80 | 0.03 |
| SGA_ExN_NxE.txt | YAL007C_tsq2 (erp2-1) | YAL008W_dma2 (fun14) | DMA26 | -0.30 | 0.0001 | 0.50 | 0.06 |
| SGA_NxN.txt | YAL009W_sn1 (spo7) | YAL010C_dma3 (mdm10) | DMA30 | 0.02 | 0.5 | 0.99 | 0.01 |
| SGA_NxN.txt | YAL011W_sn2 (swc3) | YAL012W_S1 (cys3-S1) | DMA30 | 0.05 | 0.3 | 1.01 | 0.02 |

Both loaders are built with `io_workers=1, batch_size=2` (`io_workers=0` is impossible: `ThreadPoolExecutor(max_workers=0)` raises), so the gene set goes through the forked `CpuExperimentLoaderMultiprocessing` path and the reference index through the sequential path.

- Temperatures parse from the digits of `Arraytype/Temp`: `[30, 26, 26, 30, 30]`.
- DMF record 0: fitness 0.70, `fitness_std = fitness_uncertainty = 0.05` typed `sample_sd`, `n_samples = 4` colonies, so `fitness_se = 0.05 / 2 = 0.025`. Reference std is the mean DMF sd per temperature: 30 C `(0.05+0.01+0.02)/3 = 0.0267`, 26 C `(0.03+0.06)/2 = 0.045`, SE half of each.
- DMI record 0: `GeneInteractionPhenotype(gene_interaction=-0.12, p=0.001, graph_level="edge")`; reference `0.0` with `p=None`, also `edge`. All five `(eps, p, temperature)` triples pinned in file order.
- Both: gene set = the ten systematic names sorted; two references in first-sighting order, 30 C members `[0, 3, 4]` then 26 C members `[1, 2]`; 4 interned objects; `processed/` is exactly `interned, lmdb, pre_filter.pt, pre_transform.pt`.
- `subset_n=2` on DMF keeps `df.sample(n=2, random_state=42)` = source rows 1 and 4 (efb1-1 x ssa1-1 at 26 C, swc3 x cys3-S1 at 30 C), re-indexed to records 0 and 1.

### Findings (pinned as the code behaves)

- **SMF unknown suffix crashes with `UnboundLocalError`.** A strain id whose suffix matches none of damp/tsa/tsq/dma/sn/S gets `perturbation_type = "unknown"`, and `create_experiment` then raises `UnboundLocalError` because no `elif` branch binds `genotype`. A typed error naming the strain would be the fix.
- **DMF silently stores a one-perturbation genotype.** The same unknown suffix on an array strain appends nothing in the DMF loader, so a double-mutant record with ONE perturbation is written with no error. The DMI loader asserts `len(genotype) == 2` and raises `AssertionError("Genotype must have 2 perturbations.")` on the same row; the two loaders disagree on the same input.
- A NaN interaction score is refused at `GeneInteractionPhenotype` construction (`Gene interaction cannot be NaN`), so a NaN row cannot reach the DMI store.

### Not covered, and why

- `download()` for all three classes (network + zip extraction). Deliberately excluded; the hermetic contract is that raw files are present.
- The DMF build-time failure path when a batch raises inside the `ThreadPoolExecutor` (the DMI unknown-suffix case pinned above is exercised through the static `create_experiment`, not through a full build, to avoid leaving an open LMDB env behind in the test's tmp dir).
- `main()`.

## 2026.09.30 - Phase 14: the remaining SMF, DMF and DMI branches on the synthetic archive

Eighteen to thirty tests, 81 to 99 percent (two branches left, an unknown query suffix in DMF and DMI). The issue #410 deletion twins as whole records 0 (26 C) and 3 (30 C) differing only in temperature, each with its own reference (0.02, 0.0225); a DAmP twin pair beside a TS allele with different values; a blank SMF stddev dropping a measured fitness (`dropna`, line 267) with the reference index [0..2] and [3..6]; DMF with a suppressor query, DAmP array and NatMX array, a blank DMF SD stored as a NaN SE typed `sample_sd` (730), the full KanMX x NatMX record and the shared 30 C reference; DMI whole records; `subset_n=2` keeping the same rows as DMF under seed 42; all three `download` paths on a faked archive; `main`'s seven roots.

Findings: SMF at 22 C raises `UnboundLocalError` (377-381); DMF raises the same on a `TSA22` row (742-746) while DMI stores that row at 22 C; a blank DMF value refuses the whole build with "Fitness cannot be NaN".
