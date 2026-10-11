---
id: mxzux21dzfp6d95bdsi4gjg
title: Dechiara2022
desc: ''
updated: 1791716143052
created: 1791716143052
---

## 2026.10.11 - Release inventory and the growth-block misalignment

De Chiara et al. 2022 (Nat Ecol Evol 6:448-460, doi 10.1038/s41559-022-01671-9, mirror key `dechiaraDomesticationReprogrammedBudding2022`, Zotero group item N6R6CPDT) phenotyped the 1,011-isolate panel of Peter 2018 for three parts of the life cycle: asexual growth in 43 carbon and nitrogen environments, the sexual cycle (sporulation), and chronological life span (CLS). The DOI in the mirror manifest is `10.1038/s41559-022-01671-9`.

Everything per isolate is in one sheet, Supplementary Table 3 of `si/si4.xlsx` (MOESM4, sha256 `b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e`): 1,011 rows keyed by `Standard name`, 10 metadata columns, and 185 phenotype columns. Supplementary Table 10 names "the 99 phenotypes measured in this study": 5 sporulation, 8 CLS, 43 doubling times and 43 yields. Supplementary Table 3 carries each growth trait twice, once as the control-normalized log2 value the paper analysed and once as an absolute value (hours, cells), so 13 + 4 x 43 = 185.

Measured by [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_release_inventory]] (`results/dechiara2022_release_inventory.json`):

| quantity | value |
|---|---|
| isolates in Supplementary Table 3 | 1,011, names unique |
| isolate names found in Peter's presence/absence matrix | 1,011 of 1,011 |
| phenotype columns whose non-missing count differs from Supplementary Table 10 | 0 |
| `SACE_` isolates with any growth value | 0 of 93 |
| rows whose normalized and absolute growth blocks describe the same isolate | 386 |
| rows whose two blocks describe different isolates | 519 |
| rows with no absolute growth value / too few values to match | 105 / 1 |

**The two growth blocks disagree on 519 rows.** Where the blocks agree, `log2(absolute) - normalized` is a per-environment constant (largest within-environment SD 0.032 log2 units for doubling time, 0.28 for yield), which is the back-transform the authors describe in the peer-review file (`si/si3.md`: "We now also show and report in table S3 the absolute cell doubling time and cell yield, in hours and numbers of cells"). Matching every row's 86-value profile against every row of the other block finds a best partner with Pearson r >= 0.964 for 99% of rows, and the partner of 516 rows is the alphabetically PREVIOUS isolate name (`BDM` through `CRL`): the normalized block is shifted by one row over a contiguous alphabetical run. `BDL` has normalized values but no absolute ones and `CRL` the reverse, the two ends of the run.

**Genotype adjudication says the absolute block is the correctly named one.** In the 386 agreeing rows both blocks carry the textbook sugar associations with Peter 2018's pangenome presence (two-sided Mann-Whitney): SUC2 (`YIL162W`) and sucrose 2% doubling time p = 3.1e-4 / 3.0e-4 (normalized / absolute); MAL13 (`YGR288W`) and maltose 32% yield p = 4.1e-10 / 3.3e-10. In the 519 disagreeing rows only the absolute block keeps them: sucrose 2% p = 0.68 (normalized) against 3.1e-3 (absolute); maltose 32% yield p = 0.30 against 6.0e-14; maltose 32% doubling time p = 0.42 against 2.5e-8.

**The life-cycle columns are correctly named.** Recomputing Extended Data Fig. 7b from the released CLS columns and Peter's pangenome ORF `X560` (HPF1-like) gives n = 89 vs 718 and p = 1.9e-9, 8.6e-9, 1.6e-5 for crSDC days 7, 21, 35, the caption's values exactly. The sporulation and CLS columns also equal, name for name, the per-isolate source-data workbooks for Extended Data Figs. 3b, 4a, 5 and 6.

Decision taken by recommendation: the loader reads the ABSOLUTE growth block only and refuses the normalized block, whose row names are wrong for 519 isolates and which carries no information beyond a per-environment constant where it is right. The misalignment is reported upstream as a finding, not repaired by us (repairing it would mean re-assigning released numbers to names by our own inference).

## 2026.10.11 - Three stores: life cycle, doubling time, cell yield

Module: `torchcell/datasets/scerevisiae/dechiara2022.py`. Adapters: `torchcell/adapters/dechiara2022_adapter.py` with three confs. Verification: the module's `verify_build`, registered in `NATURAL_ISOLATE_DATASETS` (`torchcell/verification/runners.py`, `run_natural_isolate`). Raw mirror: `$DATA_ROOT/torchcell-raw/dechiaraDomesticationReprogrammedBudding2022/si/41559_2022_1671_MOESM4_ESM.xlsx` (sha256 `b2eb2e28cdd2c3037ce14bb1eb8f58ee84609157dd592b3904595b24daf8c99e`, `springer_esm` retrieval restated from the literature manifest), the only file the records are built from. The Peter 2018 tarball and matrices are read from the genomes tier (`peter2018_1011_assemblies`), not copied.

### Trait inventory and where each went

Every per-isolate phenotype is a column of Supplementary Table 3; Supplementary Table 10 names "the 99 phenotypes measured in this study".

| released columns | n columns | unit, as released | replicate structure (quoted below) | store / class |
|---|---|---|---|---|
| `Asci at 24h`, `Asci at 72h` | 2 | percent of cells that are asci | not stated per isolate; >100 cells counted | life cycle, `sporulation_efficiency` |
| `Dyads at 24h`, `Dyads at 72h` | 2 | percent of ASCI that are dyads | as above | life cycle, `dyad_fraction` |
| `Sporulation in water` | 1 | percent of cells sporulated, day 8 | not stated; 10,000 cells per sample | life cycle, `sporulation_efficiency` |
| `CLS in water (day 8, 20)` | 2 | percent viable cells | single replicate; 10,000 cells | life cycle, `stationary_phase_viability` |
| `CLS in SDC (day 7, 21, 35)` | 3 | percent viable, plate-normalized (can exceed 100) | single replicate; 10,000 cells | life cycle, `stationary_phase_viability` |
| `CLS in crSDC (day 7, 21, 35)` | 3 | as above | as above | life cycle, `stationary_phase_viability` |
| `<env>_doublingtime_hours` | 43 | h | 4 replicate growth curves, median | doubling time, `EnvironmentResponsePhenotype`, `growth_rate` |
| `<env>_yield_cellscount` | 43 | cells | 4 replicate growth curves, median | cell yield, `EnvironmentResponsePhenotype`, `cell_yield` (new) |
| `<env>_doublingtime`, `<env>_yield` | 86 | log2 relative to plate controls | as above | REFUSED (misaligned, see the inventory section) |
| `Ploidy`, `N of aneuploidies`, `EUP_ANEU`, `HOM_HET`, `Clade`, `Clade_class`, `Isolate_eco_origin` | 10 | isolate annotations | - | not served; karyotype has no genotype slot (#1001) |

The paper says "86 environments" for the growth assays; Supplementary Table 3 has 43 environments with two traits each (`Methods`, `Supplementary Table 4`), which is the 86.

New schema (additive): `LifeCyclePhenotype` (`life_cycle` keyed `<measurement type>@<time><h|d>`, typed `LifeCycleMeasurementType` and `LifeCycleAssay`, a part over its whole bounded by 100), `LifeCycleExperiment` / `...Reference` declaring a `CultureEnvironment`, `MeasurementType.cell_yield` (absolute), `ConcentrationUnit.mg_nitrogen_per_l` (the 30 mg N/L nitrogen dose), `PreCultureSource.timed_culture`; library media `SPORULATION_KAC` and `PBS`. A sporulation efficiency is not stuffed into `FitnessPhenotype` (it clamps non-positive values) nor into `EnvironmentResponsePhenotype` (one scalar, no time axis).

### Record design

- **Life cycle**: one `LifeCycleExperiment` per (isolate, protocol); a protocol is one culture followed over time by one assay, so its time points stay in one record. Five protocols: KAc sporulation (microscopy, 23 C, 250 ml flask at 5:1, 250 rpm, OD600 0.6, after a 48 h YPA presporulation culture); PBS sporulation (flow-cytometry scatter, day 8; temperature a typed gap); PBS viability (PI / YO-PRO-1 stain, 30 C, days 8 and 20); SDC and calorie-restricted SDC viability (30 C, days 7, 21, 35 counted from saturation at 72 h). A dyad value exists only where the asci value is above 0 (measured on the release, asserted at build time), so the dyad key is absent, not 0, when a culture formed no ascus.
- **Growth**: one `EnvironmentResponseExperiment` per (isolate, environment). A carbon environment is the background plate without glucose (base `SC`) plus an `EnvironmentPhysicalPerturbation(carbon_source)` at the stated percent (w/v); a nitrogen environment is the plate without ammonium sulfate or CSM (base `YNB`) plus `EnvironmentPhysicalPerturbation(nitrogen_source)` at 30 mg N/L; `YPD` is the library `YPD_AGAR` restated with the paper's recipe. Malic acid has no released concentration, so its magnitude is a typed gap. Compound labels are the paper's own words; no stereochemistry is added (`isoleucine`, not `L-isoleucine`), so an unresolved label carries the resolver's typed gap.
- **Genotype**: Caudal 2024's construction (sequence variants vs the R64 slice, accessory presences, core absences) from Peter 2018's gene-keyed stores, imported from `caudal2024.py`. All 1,011 Supplementary Table 3 names have a Peter genome, so 0 isolates are refused for a missing genome. Each isolate's genotype is interned once (`processed/interned`), which keeps the doubling-time store at 131 MB of records plus 578 MB of interned constants instead of ~22 GB.
- **Reference**: no control strain was released, so each reference phenotype is the population mean over the built isolates for that environment or protocol (the Caudal 2024 convention), labeled as such in `units`.

### Counts (measured on the dev stores)

| store | records | isolates | conditions | released cells kept / missing / refused |
|---|---|---|---|---|
| `life_cycle_dechiara2022` | 4,039 (KAc 850, PBS sporulation 826, PBS viability 749, SDC 807, CR 807) | 905 | 5 protocols, 13 time points | 9,980 / 3,163 / 0 |
| `doubling_time_dechiara2022` | 36,295 | 906 | 43 environments | 36,295 / 7,178 / 0 |
| `cell_yield_dechiara2022` | 36,286 | 906 | 43 environments | 36,286 / 7,187 / 0 |

Refused: the 36,295 + 36,286 control-normalized growth cells (issue #1006); 93 `SACE_` isolates carry no growth value in the release. Not loaded, counted in #1007: the engineered validation strains (IME1, WHI2 rescue, HPF1-like deletion, 757 source-data rows) and the two-strain spore-viability time course. Not loaded because they are other studies' data or statistics: the Peter 2018 stress yields restated in Extended Data Fig. 1 (row 2 of the schedule), the S. paradoxus water sporulation of Extended Data Fig. 2c-d (another species), the GWAS matrices and hits (Supplementary Tables 5-9, `si5.gz`), the flow-cytometry events of Fig. 3c, and the clade medians of Supplementary Table 1.

### Sourcing (verbatim, sha256-pinned)

From `paper.md` (sha256 `ca68d65691eaff4d6c1b6bf474a4ecef57a5bc41467b51ac993d816cfa033ea1`), the Reporting Summary text `si/si2.pdftotext.txt` (sha256 `ce323f8d2b0c2e59f4fef4551794599b350eb372ef247652aea814fef5075757`, rendered by [[experiments.036-dataset-fixes-before-kg-build.scripts.dechiara2022_reporting_summary_text]]) and the Peer Review File `si/si3.md` (sha256 `6ffb9b0b807b3b04958eee76041148db1281521355b7899aefeb0f50d8ed33a4`). All 29 module quotes are re-read against the bytes by `verify_build` (L3 `sourced_quotes_are_verbatim`); the key ones:

- Growth replicates (Reporting Summary): "Automatic growth curve were conducted in 4 replicates, median values were taken for each isolates."
- Absolute values (Peer Review File): "We now also show and report in table S3 the absolute cell doubling time and cell yield, in hours and numbers of cells"
- Carbon environments: "For alternative carbon sources or concentrations, $2 \%$ glucose was replaced by the indicated carbon source at the indicated concentration."
- Nitrogen environments: "For alternative nitrogen sources, ammonium sulfate and CSM were replaced by the alternative nitrogen source at growth yield limiting concentrations $( 3 0 \mathrm { m g N l ^ { - 1 } } ,$ )."
- Growth temperature: "Cultivation using the Scan-o-matic platform82 was performed in thermostatic cabinets at $3 0 ^ { \circ } \mathrm { C }$ and constant, high humidity."
- KAc counting: "To estimate sporulation efficiency, ${ \scriptstyle > } 1 0 0$ cells per sample were counted at 24 and $7 2 \mathrm { h }$ post-transfer to sporulation medium using an optical microscope (Zeiss Axio Lab.A1)."
- Sporulation efficiency: "Sporulation efficiency was estimated as the number of asci divided by the total cell count."
- Dyads (Extended Data Fig. 4a caption): "the percentage of produced asci that are dyads"
- KAc temperature: "Flasks were kept at $2 3 ^ { \circ } \mathrm { C }$ and shaken at $2 5 0 \mathrm { r . p . m }$ ."
- Water sporulation time (Fig. 3d caption): "FACS-based analysis of the sporulation of domesticated and wild yeast after 8 d in water." (the OCR keeps a thin space between 8 and d)
- CLS replicates: "Experiments were run in a single replicate and data were normalized to four internal controls distributed in each plate. Ten-thousand cells per sample were analysed."
- CLS time origin: "Aging was considered to start at saturation of the cultures, $7 2 \mathrm { h }$ post-inoculation84."
- Haploid exclusion (Reporting Summary): "Haploid isolates were not considered for sporulation statistics given they cannot sporulate." Measured: 0 haploid isolates carry a KAc value; 45 carry a PBS sporulation value, loaded as released.

The sporulation replicate count is a typed gap on `n_samples` and `sample_unit`: the Reporting Summary says only "Other phenotypes were replicated (at least four replicas) to verify reproducibility before being applied to the whole strain collection", which describes protocol validation, not the cultures behind a released value.

### Verification (dev stores, `verify_build`)

| level | rule | life cycle | doubling time | cell yield |
|---|---|---|---|---|
| L0 | structural (each shared genotype validated once) | 4,039 ok | 36,295 ok | 36,286 ok |
| L1 | count = released non-missing conditions | 4,039 / 4,039 | 36,295 / 36,295 | 36,286 / 36,286 |
| L1 | isolate x condition uniqueness | pass | pass | pass |
| L1 | every genotype names one Peter-panel isolate | 905 isolates | 906 | 906 |
| L2 | released value fidelity (every stored value equals its cell) | pass | pass | pass |
| L2 | viabilities above 100 = declared 37 / values positive | 37 = 37 | 36,295 positive | 36,286 positive |
| L3 | one assay per protocol / one measurement type | 5 protocols | `growth_rate` | `cell_yield` |
| L3 | reference is the population mean | pass | pass | pass |
| L3 | 29 quotes verbatim in their pinned files | pass | pass | pass |
| L3 | media membership, compound identity, gap census | pass | pass | pass |
| L4 | sequence-variant genes are SGD R64-4-1 genes | 5,977 / 5,977 | 5,977 / 5,977 | 5,977 / 5,977 |

### Owner decisions taken by recommendation

1. Load the ABSOLUTE growth block and refuse the normalized one (measured misalignment, #1006). Reason: the absolute block is correctly named and carries everything the normalized one does.
2. Growth on the existing `EnvironmentResponsePhenotype`, two stores (the verifier requires one measurement type per store); a new `MeasurementType.cell_yield` rather than reusing `colony_size`, since the source does not say whether its yield is an end-point size or a gain.
3. A new `LifeCyclePhenotype` leaf keyed by measurement type and time, one record per culture time course.
4. Nitrogen dose on its own unit (`mg_N/L`) rather than converted to compound mass with a molecular formula the source never uses.
5. Interning the isolate genotype in the dev LMDB; the two raw-cursor gene readers now resolve it (commit `5af868d60`).
6. Population-mean references (no control released), as for Caudal 2024.

Schema impact (`scripts/schema_impact_check.py --base origin/main`): 96 datasets impacted, 0 breaking; additive. The new `ConcentrationUnit` and `MeasurementType` members move every closure that holds them, so the stores read stale until the KG 4.0 full rebuild that follows this wave.

Candidate gate: admissible, G1-G5 pass (`database/candidates/dechiaraDomesticationReprogrammedBudding2022.json`). Issues: #1006, #1007, #1008 filed; De Chiara evidence added to #1001 (karyotype) and #1002 (shared isolate-genotype module).
