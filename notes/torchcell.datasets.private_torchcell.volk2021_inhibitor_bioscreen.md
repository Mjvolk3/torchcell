---
id: n442asp7k0im1qf7lf4wpec
title: Volk2021_inhibitor_bioscreen
desc: ''
updated: 1791437948717
created: 1791437948717
---

## 2026.10.08 - Sources, mirrors and derivation ahead of the loader

Private dataset in preparation: wild-type growth of strain bAID under dosed, often multi-inhibitor YPD, from the 2021 Bioscreen C runs behind the preliminary exam. This phase writes everything that does not depend on the schema PR (`SourceType` / `preliminary_report`, `IntegratedCassette`, `MeasurementType.relative_growth_rate`, `ResponseCategory.no_growth`, `Visibility`); the dataset class follows once that PR lands.

Code: [[torchcell.datasets.private_torchcell.bioscreen]] (mirrors, layouts, traits) and [[torchcell.datasets.private_torchcell.volk2021_sources]] (sourced constants and typed gaps). Analysis this reuses: [[experiments.039-inhibitor-combinations-wetlab]] (branch `exp/039-inhibitor-combinations-wetlab`).

### The runs

| run | what | wells in layout | started | length of export |
|---|---|---|---|---|
| ex21 | six single-inhibitor titrations, 9 doses + 1 uninhibited well per block, 3 replicate blocks | 180 (162 dosed, 18 WT) | 2021-04-04 | 71.97 h |
| ex23 | the 63 combinations at one dose each, biological triplicate | 197 (189 dosed, 8 WT; 3 blanks excluded) | 2021-04-17 | 84.97 h (aborted by hand at 3 d 13 h) |
| ex26 | furfural x acetic acid isobole, 2 plates of 10 x 10 | 200 (2 WT) | 2021-09-07 | 95.99 h |
| ex27 | formic acid x acetic acid isobole | 200 (2 WT) | 2021-09-11 | 95.98 h |
| ex28 | 5-HMF x acetic acid isobole | 200 (2 WT) | 2021-09-30 | 95.98 h |

Start dates and lengths are the `.bsm` event records and the export's first and last reading. No run is 48 h long; 48 h is the lag the Bioscreen software assigns to a well with no growth in the ex23 and ex26 trait files. ex22 (calibration) is excluded. The isoboles were run after the report (May 2021) and are not in it.

### Strain identity

The archive calls the strain "BY4742-iAID6" and the report "MAGIC strain BY4742-iAID6". Lian 2019 builds bAID by "integrating PmeI-digested pAID6 into the genome of BY4742" and lists `bAID | BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]` in Supplementary Table 9 (`si/si1.md`, sha256 `b2bcfe2e...3d32db`). Every row of `MV_ex23_preprocessed.csv` carries strain `BY4742-iAID6`, and the ex23 preculture was grown in YPD with G418 (`1_inhibitor-screen_OD_calculation_2021_04_14.xlsx` B1), consistent with the KanMX marker.

Observation across different runs, not a result: the mean software generation time of the uninhibited wells is 1.815 h in ex23 (8 wells, bAID; the thesis's first WT well is 1.79 h), while the `WT` wells of the mutant runs, strain column BY4741, average 1.473 (ex16, n = 44), 1.681 (ex17, 44), 1.529 (ex18, 8), 1.504 (ex19, 9), 1.292 (ex24, 10) and 1.265 h (ex25, 3). Different days, plates and strains; no comparison was designed.

### Mirror chain

- Library key `volkPreliminaryExamReport2021` at `$DATA_ROOT/torchcell-library/volkPreliminaryExamReport2021/`: `paper.pdf` = `/bulk/thesis/thesis_archive/reports/Prelim_Report_2021_Michael_Volk.pdf`, sha256 `4c2fcf11e16468b20fbbd589f651821bff5904884a7584ccae27ee2b70ec60b9`; `paper.md` from MinerU 2.7.6 (cpu, 200 dpi), sha256 `61f2b85096f5dce77de8e97061eb44daa1e42e3663c14cc0b8d8f09d4563cf33`. Title "Machine Learning for Engineering Improved Yeast Fitness", doi None, no Zotero item. The PDF's record is a `local_archive` retrieval; `reports/` is not in the archive's `MANIFEST.tsv` (it covers `data/` only), so the original location of the report is not recorded there. Rebuilt by `bioscreen.deposit_report`.
- Raw mirror at `$DATA_ROOT/torchcell-raw/volkPreliminaryExamReport2021/`, 26 files at their archive paths (`data/...`), each checked against `MANIFEST.tsv` and recorded with the original Box path as `source_url` and the new `local_archive` retriever (`torchcell.literature.retrieve.local_archive`, refuses a sha256 mismatch). Rebuilt by `bioscreen.deposit_raw_mirror`. Beyond the requested exports, `.bsm` files, spreadsheets, `MV_ex23_preprocessed.csv` and the ex26 trait file, it holds the files the doses and conditions are sourced from: the ex23 well map and cell-volume tables, the ex23 OD calculation, `1_inhibitor-screen.ipynb`, the isobole design sheet `BioscreenC_FF_AA_Isobole_exp.xlsx`, `Isobole_BioscreenC_plate_setup.xlsx`, the Fluent worklist, the ex21 software traits, and `ex1.txt` + `MV_ex1.bsm` (the labeled temperature log).
- Tolerance of the nightly jobs, read from the code: `lit_sync` (`sync_collection*`) and `lit_annotations` iterate Zotero items, so a key with no Zotero item is never visited. `reocr_si` acts only on keys named with `--key` and would refuse this one (`KeyNotInZoteroError`); the key has no SI PDF anyway. `tc-lit` lists it like any directory. One hazard: `backfill_mirror --force` (manual) would rewrite the manifest offline, dropping the title and the PDF's retrieval (`provenance_complete=False`); `deposit_report` restores it. Without `--force` the key is skipped.

### Doses (two corrections to the 039 constants)

- ex23: FF 1.5, AA 2, HMF 2.522, FA 1, LVA 6, LA 20 g/L = uL of stock in the well map (42.5, 8.5, 17, 17, 51, 85) x stock (60, 400, 252.2, 100, 200, 400 g/L, `inhibitors.xlsx` C12:C17) / 1700 uL. The report's Fig 11b dose table shows the same six values (FF 1.5, AA 2, HMF 2.52, FA 1, LVA 6, LA 20), read by eye from the rendered page because MinerU did not capture it (the Fig 11b region came out as an empty text block). The 039 script's `EX23_G_PER_L` (FF 6, AA 4, LVA 2, LA 40) is the literature `C2` column, the fifth titration step, not what ex23 dispensed.
- ex21: dose = uL of stock per 4 mL tube (`inhibitors_titration_array.xlsx`) x stock / 4000. This reproduces every notebook label except two typos the 039 script carries verbatim: FF "28" is 18 g/L and FA "1.25" is 0.125 g/L.
- Isoboles: the 039 steps (FF 0.3333, FA 0.2, HMF 0.504, AA 0.4 g/L per 10 uL in 200 uL) equal the design sheet's dilution x 10/200 within 1e-3.

### Derivation and validation

`generation_time` is the 039 algorithm unchanged (baseline = median of the first three reads, growth if the rise reaches 0.3, steepest 3 h log2-linear window), returning None for no growth. Relative growth rate = mean generation time of the run's uninhibited wells that grew / the well's. Two trait sources (`TraitSource`): `raw_curve` for every run, `bioscreen_software` for ex21, ex23 and ex26 (the thesis's PRECOG calls, which the report says used "estimated cell counts from calibration curves").

| check (measured 2026-10-08, `tests/torchcell/datasets/private_torchcell/test_bioscreen.py`) | value |
|---|---|
| ex26 grew/no-grew agreement, raw vs software, 200 wells | 0.985 |
| ex26 Spearman of generation time, 45 wells both call grown | 0.804 |
| ex21 agreement / Spearman (180 wells, 117 both grown) | 0.978 / 0.683 |
| ex23 agreement / Spearman (197 wells, 69 both grown) | 0.939 / 0.794 |
| wells grown, raw: ex21 / ex23 / ex26 / ex27 / ex28 | 117 / 69 / 45 / 76 / 53 |
| WT generation time, raw: ex21 / ex23 / ex26 / ex27 / ex28 (h) | 1.729 / 1.345 / 2.144 / 2.183 / 2.163 |
| WT generation time, software: ex21 / ex23 / ex26 (h) | 2.615 / 1.815 / 2.166 |

ex23 single-inhibitor relative growth rate, mean of 3 wells:

| source | AA | FA | LVA | FF | LA | HMF |
|---|---|---|---|---|---|---|
| software (thesis, Fig 11b) | 0.947 | 0.937 | 0.873 | 0.769 | 0.767 | 0.437 |
| raw curve | 0.998 | 0.996 | 0.869 | 0.507 | 0.982 | 0.480 |

The two sources rank ex26 alike but disagree in scale on ex23 and ex21. Hypothesis (untested): the software's OD-to-cell-count calibration changes the steepest-window rate at the higher ODs these runs start at. The loader must pick one source per run and record it; that choice is open.

### What the report states vs what is gapped

Sourced to the report OCR: strain name, dose-selection rule, fitness definition, biological triplicates (ex21 caption, ex23 text), Bioscreen C with 200 wells, PRECOG with calibration, randomized well positions (true for ex23 only). Sourced to the archive: stocks, literature C2 doses and their source columns, ex23 volumes, the 1700 uL tube with the inoculum inside it, YPD as the medium, initial OD 0.2 (ex21, ex23, isoboles), the 200 uL isobole well with 20 uL inoculum, ex21 end-of-run well volumes (102.4 to 194.8 uL), run lengths.

Measured from the instrument log: tray temperature median 30.01 C on every run (1st to 99th percentile 29.99 to 30.01 C, `.bsm` `T` records after warm-up; the field order is fixed by ex1's labeled `ex1.txt`). Read interval 900 s (`S,4`, matching the 15 min export spacing).

Typed gaps (`volk2021_sources.GAPS`): shaking (no decoded source; hypothesis that `S,U` changing 1 -> 2 at ex13, named "amp_med_speed_fast", is the shaking code), temperature set point (`S,2 = 30` undecoded), well volume for ex21 and ex23, ex21 inoculum dilution (doses are the 4 mL tube concentrations; a 1:10 addition would make the culture dose 0.9x, unverified), YPD recipe, and the isobole grid orientation: the design slide says "All the first row contains 90 uL of inhibitor 1" and the Fluent worklist sends 90 uL from trough 1 to well 1, which disagree with the 039 rule. Of the four axis flips only the 039 orientation keeps growth monotone in dose (0 to 2 violations per plate vs 4 to 17), so the layout keeps it; a swap of the two inhibitors cannot be ruled out that way.

### Planned record shape for the loader

One record per inoculated well: `StrainEnvironmentResponseExperiment` with a genotype of zero perturbations on the bAID background (BY4742 + the integrated pAID6 cassette as `IntegratedCassette`), a `CultureEnvironment` of YPD with one `SmallMoleculePerturbation` per inhibitor at its g/L dose, and an `EnvironmentResponsePhenotype` with `measurement_type=relative_growth_rate` and `assay_type=liquid_od_growth`, or `category=no_growth` for a well that did not grow. `screen_id` = run and well (e.g. `ex23:well150`). Biological replicate id from the layout (ex21 block, ex23 table, isobole plate). The dataset carries `Visibility` private and the `Publication` source type `preliminary_report`.

## 2026.10.08 - The dataset as built

`InhibitorBioscreenVolk2021Dataset` (`torchcell/datasets/private_torchcell/volk2021_inhibitor_bioscreen.py`, adapter `torchcell/adapters/volk2021_inhibitor_bioscreen_adapter.py`) builds from the raw mirror alone: 977 records in 5 s under `$DATA_ROOT/data/torchcell/inhibitor_bioscreen_volk2021` (ex21 180, ex23 197, ex26 200, ex27 200, ex28 200). Dropped: ex23's three uninoculated `blank` wells (80, 87, 157) and ex21's unassigned wells 91 to 100 and 191 to 200.

Two departures from the planned shape above. A well that did not grow is `category=severely_reduced` (the enum's own meaning, "growth or signal essentially abolished") with `category_label` "no growth within <h> h"; no `no_growth` member was added because that one already means it. And the loader declares `has_gene_perturbations = False`: every record is the unedited host, so the dataset's gene set is legitimately empty, and `ExperimentDataset.gene_set`'s setter accepts an empty set only under that class-level declaration (it still refuses `None`, and refuses an empty set for every loader that does carry gene edits).

Records read back from the build (the WT well's rate is its own generation time against the run's WT mean, so WT replicates scatter around 1.0):

- ex23 well with acetic acid 2 g/L + formic acid 1 g/L + levulinic acid 6 g/L (index 214): genotype `[]`; 84.97 h, 30.0 C; `relative_growth_rate` 0.335; reference strain `bAID`, background `bAID` with the integration `Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]`.
- ex26 well with furfural 1.666 g/L (index 382): `categorical` / `severely_reduced`, "no growth within 96 h", 95.99 h.
- ex21 WT well (index 0): no perturbation, 71.97 h, `relative_growth_rate` 0.669.

Serving: `visibility = private`. The GilaHyper live-rebuild and increment slurm scripts now default to `INCLUDE_PRIVATE=1` (`--include-private` to the generator and to the admission check); the generator unions `PRIVATE_DATASET_ADAPTER_MAP` only under that flag and refuses a private dataset named without it; `scripts/package_dataset_lmdb.py` (tc-data) refuses it either way. The admission check against the served manifest (store 4b293d34, run 2026-10-08 with `--include-private`) reports the dev LMDB fresh and the dataset BLOCKED for the expected reason: `Publication` and `SourceType` moved the schema closure of all 51 served datasets (PR #778), the `publication` and `crispr construct` graph classes changed, and the compound and media value surfaces changed. So this dataset enters the served store through the next FULL rebuild, not an increment.
