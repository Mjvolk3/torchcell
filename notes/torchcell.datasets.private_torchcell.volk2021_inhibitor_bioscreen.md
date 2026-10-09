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

## 2026.10.09 - The private path into KG 4.0, verified by running it (#827)

Every row below was measured today against the store this branch rebuilt, not read off
the code. Scripts and logs are in the session scratchpad
(`run_generator.py`, `measure_store.py`, `measure_plots.py`, `run_verify.py` under
`/scratch/tmp/claude-1000/.../scratchpad/feat-827-bioscreen/`); the gate itself is
committed as `verify_build` in the loader module and `test_live_rebuild_slurm.py`,
`test_create_scerevisiae_kg_small.py` and the dataset's own test file carry the pins.

### 1. The generator emits the private dataset's CSVs only with `--include-private`

`create_scerevisiae_kg_small` run on the real dev store with
`--config-name kg_uncapped --include-private '+datasets=[InhibitorBioscreenVolk2021Dataset]'`,
writing to a scratch output directory (nothing touched the served `torchcell` database or
`$DATA_ROOT/database/`, and no slurm job was submitted):

| measured | with `--include-private` | without it |
|---|---|---|
| exit | 0 | 1, `PrivateDatasetRefused: ... Pass --include-private` |
| nodes / edges | 9652 / 9654 | none written |
| CSV families | 24 node + edge families | 0 CSVs |
| `Experiment-part000.csv` | 977 rows | absent |
| `ExperimentReference-part000.csv` | 5 rows | absent |
| `EnvironmentResponsePhenotype-part000.csv` | 982 rows (977 + 5) | absent |
| `Dataset-part000.csv`, `Publication-part000.csv` | 1 row each | absent |

Two local-environment notes, neither a defect of the build: the rehearsal redirects
`build_telemetry.CGROUP_ROOT`, because the sampler reads cgroup v2 files that exist in the
build container and not in a login shell, and it runs with `adapters.fast_writer=false`,
because `fast_csv.build_row_specs` calls `BioCypher._initialize_writer`, which the
container's pinned `biocypher==0.15.2` (`env/requirements.txt`) has and the local
conda env's 0.5.43 does not. The fast-CSV writer path is therefore **not** exercised
here; it is exercised by the build itself.

`INCLUDE_PRIVATE` in `database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm`
defaults to `1`, so a FULL rebuild serves this dataset unless someone opts out: measured
by running the script's own dispatch block (`INCLUDE_PRIVATE` unset -> `--include-private`,
`0` -> no flag, `yes` -> exit 1). The freshness preflight and the dev fence both take the
same switch into `build_adapter_map(include_private=...)`, so a public-only preflight
cannot miss a stale private store.

### 2. tc-data refuses it

`python scripts/package_dataset_lmdb.py --dataset-dir $DATA_ROOT/data/torchcell/inhibitor_bioscreen_volk2021 --store <tmp>`
exits 1 with `refused: InhibitorBioscreenVolk2021Dataset is PRIVATE (visibility=private):
in-house data is never published in a tc-data release`, and the store directory is left
empty. The refusal reads `visibility` off the loader class named by the build manifest, so
no spelling of the command publishes it.

### 3. The dev store, and its L0 to L4 table

`--list-stale --include-private` read the store STALE twice today, and both times the
drift was a shared symbol rather than anything of this dataset's: first on `SampleUnit`,
then, after rebasing onto the bacterial-perturbation-leaf land, on
`EnvironmentPerturbationType` and `GenePerturbationType`. 109 of the 112 mapped stores
are stale on the same symbols, which is the expected state before a full rebuild. Rebuilt
with `--retire-existing` (dev tree only; the previous `processed/` and `preprocess/` are
`*.superseded.20261009-093436` siblings): 977 records, 5 references, empty gene set, 5 s.
The store then reads `fresh`.

`verify_build` (`torchcell/datasets/private_torchcell/volk2021_inhibitor_bioscreen.py`,
report written to `preprocess/verification_report.json`):

| level | row | measured |
|---|---|---|
| L0 | structural | 977 records validate as `ExperimentType` |
| L1 | count | 977 == 180 + 197 + 200 + 200 + 200 |
| L1 | completeness | 977 / 977 `<run>:well<n>` keys, no extras |
| L1 | wells_per_run | ex21 180, ex23 197, ex26 200, ex27 200, ex28 200 |
| L2 | value_fidelity | 360 rates, all finite and >= 0 (range 0.247 to 1.350) |
| L2 | readout_split | 360 rates + 617 no-growth calls; per run 63 / 128 / 155 / 124 / 147; no record carries both |
| L3 | reference_one | reference relative growth rate 1.0 for all 977, with the sample SD of 17 / 8 / 2 / 2 / 2 grown WT wells |
| L3 | wild_type_wells | 32 inhibitor-free wells (18 / 8 / 2 / 2 / 2); every other record carries a dosed inhibitor |
| L3 | no_growth_label | severely_reduced, labelled 72 h / 85 h / 96 h / 96 h / 96 h by run |
| L3 | strain_background | all 977 references carry bAID, parent BY4742, integration `Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]` |
| L4 | software_trait_agreement | the served raw-curve grew call equals the Bioscreen software's on ex21 0.9778, ex23 0.9391, ex26 0.9850 |

The gate lives in the loader rather than in `runners.ENVIRONMENT_RESPONSE_DATASETS`
because five of that registry's rules describe a deletion-collection screen and not this
dataset, each measured on these records: `pair_uniqueness` keys on (ORF, compound) and
every ex23 condition has 3 replicate wells with an empty ORF set; `measurement_type_consistent`
requires one type and this readout is 360 rates + 617 calls; `reference_zero` requires 0
and this reference is 1.0, with `reference_centered=False` closed off because
`relative_growth_rate` is deliberately absent from `ABSOLUTE_MEASUREMENT_TYPES`;
`environment_perturbed` counts 32 unperturbed records (the WT wells, which are the
baseline, not defects); and the whole L4 gene layer is vacuous on an empty gene set.
Giving it a registry row would mean five declared reliefs on rules every public screen
shares. **Owner decision open:** either keep the loader-owned gate (what this PR does) or
add a one-strain-environment-panel family to `torchcell/verification/`. One stale comment
was found and left for the owner: `schema.py`'s `ABSOLUTE_MEASUREMENT_TYPES` block says
`relative_growth_rate` "is 0 at the control by construction", which contradicts the
`MeasurementType` docstring 40 lines above it ("1.0 = grows like the wild type"). The
enum doc is right; the comment needs a one-line fix in a schema PR.

### 4. Coverage of every run PR #740 plots

The wells each figure of `experiments/039-inhibitor-combinations-wetlab/scripts/plot_inhibitor_runs.py`
consumes, counted from the raw mirror, against the store:

| run | wells the figures use | records stored | source the figure reads |
|---|---|---|---|
| ex21 | 180 (6 inhibitors x 3 blocks x (1 control + 9 steps)) | 180 | `MV_ex21_inhibitor_titration.tsv` (software traits) |
| ex23 | 197 (189 condition wells + 8 WT; 3 blanks excluded) | 197 | `MV_ex23_preprocessed.csv` (software traits) |
| ex26 | 200 | 200 | `MV_ex26_..._Traits.txt` (software traits) |
| ex27 | 200 | 200 | raw export, this derivation |
| ex28 | 200 | 200 | raw export, this derivation |
| total | 977 | 977 | |

So no run and no well the figures plot is missing from the loader: the 977 records are
exactly the plotted wells. Two differences that are NOT coverage gaps:

- **The 25 grew / 38 did not of the ex23 table is the SOFTWARE's call.** Re-derived from
  `MV_ex23_preprocessed.csv`: 63 combinations, 25 with at least one grown replicate
  (6 singles, 14 pairs, 5 triples), 38 with none. The store's raw-curve derivation gives
  21 (6 singles, 12 pairs, 3 triples) over the same 63 combinations and the same 189
  wells. The two sources disagree on 12 of 197 ex23 wells (agreement 0.9391, the L4 row),
  which is the known scale disagreement of the 2026.10.08 table, not a missing record.
  The served label is the raw curve, per the 2026.10.08 decision.
- **One plot input is not in the raw mirror**: `plot_ex23_curves` draws the OD traces from
  `MV_ex23_Magic_inhibitor_combinations_curves_Processed.tsv`, the software's blanked
  curves. It adds no well (the same 197) and no record value, so the loader does not
  consume it; the raw export it was computed from is mirrored.

### 5. The served label and the strain

The served value is the raw-curve generation-time derivation on all five runs, defined
verbatim in every record's `units`, and the two L3 rows above assert the reference (1.0
with the WT wells' sample SD) and the typed background on all 977 records. The background
oracle is the genotype string Lian 2019's Supplementary Table 11 states for bAID
(`BY4742-Delta::KanMX-[dLbCpf1-VP]-[Csy4]-[dSpCas9-RD1152]-[SaCas9]`), split into parent
and cassette, so the row checks the store against the SOURCE and not against the
`baid_background()` constructor that wrote it. The integration object carries its two
source quotes (Supplementary Table 11 and the Methods construction sentence) with their
mirror sha256s.

### 6. What will list the dataset after the build, and what cannot be checked yet

- **`kg_manifest`**: the build writes a `KgDatasetEntry` per served dataset carrying
  `visibility`, which `dataset_visibility` reads off the loader class, so the manifest
  will name this dataset `private`. Checkable only after the build stamps the manifest.
- **The release snapshot and the docs' served-count fragment**: `SnapshotDataset` has NO
  visibility field, so a snapshot of a store built with `--include-private` lists the
  in-house dataset exactly like a public one, and
  `experiments/034-showcase-datasets/scripts/served_dataset_counts.py` generates
  `docs/source/datasets/_generated/served_counts.md` from the newest snapshot. The row
  will therefore appear on a public page with its record count and no marker saying it
  cannot be downloaded, while `tc-data` refuses the archive (section 2). Pinned as
  today's behavior in `test_release_snapshot.py`. **Owner decision open:** whether the
  snapshot should carry the marker.
- **The supported-queries page**: no supported query selects this dataset
  (`registry.json` has no entry whose Cypher names it), so it gets no `docs_page` column
  and no dataset page. Adding one is a separate curation decision.
- **Cannot be checked before the build**: the manifest entry and its closure, the
  snapshot, the served node counts, the fast-CSV writer path (needs the container's
  `biocypher==0.15.2`), and the neo4j-admin import of these CSVs.
