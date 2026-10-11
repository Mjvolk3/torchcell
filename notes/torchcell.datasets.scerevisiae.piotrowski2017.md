---
id: xkzbq19iin7qnkc14wa4ahf
title: Piotrowski2017
desc: ''
updated: 1791737263676
created: 1791737263676
---

## 2026.10.11 - Piotrowski 2017 MOSAIC chemical genetics: 289 strains x 11,975 conditions, loaded

Loader: `torchcell/datasets/scerevisiae/piotrowski2017.py` (`EnvChemgenPiotrowski2017Dataset`, dev store `data/torchcell/env_chemgen_piotrowski2017`). Source: Piotrowski JS, Li SC, Deshpande R, et al., Boone C. Nat Chem Biol 2017;13:982-993, doi:10.1038/nchembio.2436, group-library key `piotrowskiFunctionalAnnotationChemical2017` (Zotero `K8PYLHCJ`). Yeast expansion schedule row 73; audited in [[database.yeast-zotero-collection-audit.2026.10.11]]. Verifier: the environment-response family ([[torchcell.verification.runners]], `ENVIRONMENT_RESPONSE_DATASETS["env_chemgen_piotrowski2017"]`, streamed). Adapter: [[torchcell.adapters.piotrowski2017_adapter]]. Release-level counts come from `experiments/036-dataset-fixes-before-kg-build/scripts/piotrowski2017_release_inventory.py` (`results/piotrowski2017_release_inventory.json`); build counts from the store's `preprocess/dropped_records.json` and `preprocess/conditions.json`.

### What the release is

Three files of the PMC Article Datasets bucket (`PMC6056180.1`, NIHMS884855), copied from the literature mirror into `$DATA_ROOT/torchcell-raw/piotrowskiFunctionalAnnotationChemical2017/si/` with their `pmc_cloud` retrieval records:

| file | sha256 | what the loader reads |
|---|---|---|
| `NIHMS884855-supplement-Suppl_Dataset_5.xlsx` (Data Set 5, mirror `si/si13.xlsx`) | `96e07f956f7edeb07359fc8b1fa5c72a167bf9bb18d1e8a680c76362e97aaefd` | the z-score matrix |
| `NIHMS884855-supplement-Suppl_Dataset_4.xlsx` (Data Set 4, `si/si11.xlsx`) | `eba460fe354bafd4e57c2a8438201cdc9399cf97582e3d86adf3e9362f821f87` | per condition: name, dose, library, `In_final_dataset`, `In_high_confidence_set`, percent growth, SMILES, InChI |
| `NIHMS884855-supplement-Suppl_Dataset_2.xlsx` (Data Set 2, `si/si6.xlsx`) | `394505047acaea5e7e7d1afa2753e76be5356bb43abd4e01ec8eb6e1acd145a2` | pool version 2.0: ORF, common name, 20-mer UPTAG |

Data Set 5 holds 291 rows x 11,984 columns = 3,487,344 cells, 3,438,421 numeric (the audit's count, reproduced) and 48,923 `NA`. Its title reads "Average chemical genomic profile of 11983 compounds"; the 11,984th column, `Cond-008510` (NPE1123), is the one Data Set 4 marks `In_final_dataset = False`. Data Set 6 (1,522 high-confidence profiles) equals Data Set 5 on all 442,902 shared cells, so it is not consumed.

The scoring method is the Supplementary Note (`si/si1.docx`, sha256 `38cddea8...`), not the Online Methods: a strain's LOWESS-normalized log read count in a condition minus its log mean over the screen's DMSO control profiles, divided by the larger of the condition's middle-75% SD and a sign-specific reference SD, then multiplex-tag LDA components and the first SVD component removed and technical replicates collapsed. Sign, verbatim: "Positive z-scores suggest resistance to the condition, and negative z-scores suggest sensitivity to the condition relative to the fitness of each mutant in the control condition." Negative is the sick strain, as in Vanacloig 2022, Hoepfner 2014 and Wildenhain 2015, and opposite to both Hillenmeyer 2008 arms.

### What the audit missed or got wrong

- **291 rows are 289 strains.** TCD1 (YHR003C) and TCD2 (YKL027W) each appear twice: under the ORF in the 8,419 RIKEN columns and under the standard name in the 3,565 other columns, each row `NA` wherever its twin is scored. The loader resolves both rows to the same pool strain and refuses any pair that scores one condition twice (`strains_labelled_twice`). The other 7 strains `NA` in exactly the 3,565 non-RIKEN columns (CTF3, LSM6, MMS22, RAD55, RPL43A, SAC3, UBP8) are the NCI/NIH/GSK screen's strain-count filter, which the RIKEN screen did not run.
- **Pool v2.0 is 310 strains; 21 are absent from the matrix**, two of them (GTR1, AVT5) removed by the Note's stated filter and 19 with no stated reason (listed in `conditions.json`).
- **The release is two screens normalized separately** ("RIKEN" and "NCI/NIH/GSK", each with its own filters, LDA components and replicate-correlation threshold), so `screen_id` is the screen and the reference is per screen.
- **Replicates**: "three technical replicates" is the design; each released profile averages only the replicates the batch-effect filter kept and the correlation graph joined (thresholds 0.7 and 0.5), between 1 and 3 per profile, not released. The audit verdict's "3 technical PCR replicates averaged into each DS5 value" is corrected in the verdict JSON.
- **Doses** are per condition in Data Set 4, in five units plus two percent-formatted numbers (the MMS cells `4e-05`, shown `0.000%`), and 8 cells are ranges (`10-100-ug/ml`, ...).
- **Compound identifiers**: SMILES for all but 26 matrix columns and InChI for all but 33; every present InChI reads back in RDKit.
- **The untreated reference is not released.** The z-score's reference is the screen's DMSO control profiles (5,724 over both screens, per the Note), and no DMSO profile is a Data Set 5 column; the record's reference states that control (YPGal + 1% DMSO, z = 0).
- **Not loaded, worth a follow-up**: Data Set 2's second sheet, "Fitness of WT vs sensitized mut", holds SGA colony-size single-mutant fitness of 3,032 deletion strains in the original and in the pdr1 pdr3 snq2 background (a `FitnessPhenotype` dataset, not an environment response); Data Set 4's `Percent Growth` is a pool-level bioactivity per condition with no strain-level record to sit on.

### How a record is built

One `StrainEnvironmentResponseExperiment` per numeric cell of a kept strain and a kept condition. Genotype: the one `BarcodedKanMxDeletionPerturbation` (systematic name from the pool ORF through the shared resolver, the genome's own spelling as `perturbed_gene_name`, the Data Set 2 barcode, `collection` = "Diagnostic pool version 2.0 (310 strains, used for main screen)", cassette `kanMX`). YAL058C-A is an R64 alias of YAL056C-A and is kept under YAL056C-A with a `ConstructedOrf` whose relation and span are pending the SGD locus history. Background: the SGA MATa progeny of Y13206 x the kanMX array, imported from `vanacloig2022.library_background` (its construction quotes are this paper's Online Methods), so both datasets share one background entity. Environment: loader-local `YP_GALACTOSE_LIQUID` (base `YP`), 30 C, 48 h, `CultureFormat` (96-well plates, 200 uL, 250 cells per strain, fixed endpoint, shaking a typed gap), `PreCulture(frozen_stock)`, one `SmallMoleculePerturbation` (the compound at its Data Set 4 dose, `basis=fixed`, in DMSO at 1%). Phenotype: `measurement_type=z_score`, `assay_type=pooled_competitive_growth_barcode`, `n_samples=1`, `sample_unit=technical_replicate`, the three uncertainty fields typed gaps, `screen_id`, `replicate_id` = the condition id.

Compound identity: a curated `compound_identity_table` row wins (185 kept conditions); every other label takes the InChIKey hashed from the released standard InChI (11,757, the new `resolved_compound(derive_from_inchi=True)` route, [[torchcell.datamodels.compound_identity]]); 33 conditions that release no InChI are stored with a `deferred_pending_source_review` gap on `inchikey` (26 with no structure at all, 7 with a SMILES and no InChI), never dropped (#726: an uncurated label is a gap, not a drop). Over the 11,911 kept conditions with both strings, the InChI hash and RDKit's key from the SMILES differ in 194, which is why the InChI route was added rather than reusing the SMILES route.

### Sourced values

| value | quote | where |
|---|---|---|
| screen defers to optimization | "We performed our pooled growth assay with the diagnostic mutant collection under optimized conditions as described above." | paper.md line 226 |
| 200 uL, 48 h, 250 cells/strain | "Ultimately, the screening conditions we selected were $2 0 0 \mathrm { - } \mu \mathrm { L }$ microcultures grown for $4 8 \mathrm { { h } }$ at an inoculum of 250 cells/strain and 30 PCR cycles for barcode amplification." | paper.md line 33 |
| YP + 2% galactose, frozen-pool start | "To assay the mutant pool for drug hypersensitivity, cells were thawed, counted using a hemocytometer, and diluted to seven different final inoculum densities (3,727–58 cells/strain) in $\mathrm { Y P } + 2 \%$ galactose in a 96-well flat-bottom plate." | paper.md line 218 |
| YPGal recipe | "Cultures of the control strain (y13206) were grown to early log phase (0.4 OD) in YPGal $1 \%$ yeast extract, $2 \%$ peptone, $2 \%$ galactose)." | paper.md line 280 |
| 30 C | "After growing for 18, 24, or $4 8 \mathrm { { h } }$ at $3 0 ~ ^ { \circ } \mathrm { C } ,$ , cells from each well were harvested by centrifugation." | paper.md line 218 |
| 1% DMSO control | "Cultures were then spiked with either $3 4 . 4 \mu \mathrm { M }$ benomyl, $2 5 ~ \mathrm { n M }$ micafungin, or a $1 \%$ DMSO control." | paper.md line 218 |
| solvent DMSO | "The 5724 control profiles were generated by growing the yeast diagnostic mutant pool in the absence of a compound but in the presence of the compound’s solvent, DMSO (a negative control)" | Supplementary Note |
| doses | "The NPDepo is maintained as $1 ~ \mathrm { m g / m L }$ stocks, and we screened it at a final concentration of $1 0 ~ \mu \mathrm { g / m L }$ , with the exception of a number of compounds that received additional lower dosing in a pilot experiment (Supplementary Data Set 4)." | paper.md line 226 |
| 3 technical replicates (design) | "The genomic extractions for each plate were amplified in triplicate using three unique multiplex primer plates (three technical replicates)." | paper.md line 228 |
| replicate collapsing | "In the “replicate collapsing” step, profiles from technical replicate conditions were collapsed into single, mean profiles." | Supplementary Note |
| z-score | "Finally, chemical-genetic interactions were computed as z-scores representing the standardized deviation of each strain in the condition profile with respect to its counterpart strain in the reference profile." | Supplementary Note |
| two screens | "Data in this study were collected from two large screening efforts. The “RIKEN” screen encompasses all compounds originating from the RIKEN Natural Product Depository, and the “NCI/NIH/GSK” screen encompasses all remaining compounds (the NCI collections, NIH Clinical Collection, and GSK Kinase Inhibitor collection)." | Supplementary Note |
| UPTAG barcode | "Primers containing the Illumina sequencing adaptor, common priming site for the UPTAG barcode, and 1,000 selected 10-bp multiplex tags were synthesized (Sigma), arrayed in 96-well plates." | paper.md line 222 |

Every module-level `SourcedValue` (30) is written to `preprocess/sourced_values.json`; `test_every_sourced_value_is_a_verbatim_quote_in_the_mirror` re-reads each against the sha256-pinned paper OCR, the Note's `word/document.xml` runs, or the workbook cell.

### Counts (build ledgers)

| | records |
|---|---|
| released numeric z-scores (source) | 3,438,421 |
| dropped: `condition_not_in_the_final_dataset` (Cond-008510) | 289 |
| dropped: `dose_is_a_released_range` (8 conditions) | 2,312 |
| dropped: strain rules (background locus, retired ORF, legacy duplicate) | 0 |
| **kept** | **3,435,820** |

Kept: 11,975 conditions (8,410 RIKEN, 3,565 NCI/NIH/GSK), 289 strains (gene set 289). The 8 range conditions: (-)-ouabain, cisplatin, FK228, fluconazole, mitomycin C, nocodazole, puromycin, resveratrol.

### Owner decisions taken by recommendation

- **Unidentified compounds are stored with a typed gap**, as the wave brief and #726 direct, not dropped as `compound_without_a_structure_identifier` in Vanacloig, Hoepfner and Wildenhain (whose labels had been through the curator). 33 conditions, 9,474 records.
- **n_samples = 1**: the conservative end of the 1 to 3 replicates a profile averages (`preprocess/n_samples_derivation.json`, a `StatDerivation`); the alternative is the design's 3, which the Fig. 2 caption states.
- **Range doses are dropped** (8 conditions, 2,312 records): `Concentration` holds one value and the schema has no interval dose. Admitting them needs a lower/upper pair on `Concentration`, a full-rebuild schema change.
- **Cond-008510 is dropped**: the release's own metadata excludes it twice (Data Set 4's flag and Data Set 5's title count).
- **The curated table row wins** even where its key disagrees with the released InChI at the first block, as every loader does; 10 conditions (amitriptyline hydrochloride, aristolochic acid, berberine chloride, carboplatin, harmol, hematein, monensin, sodium butyrate, streptozocin, tetracycline), listed in `conditions.json` for the #726 curator.
- **Bortezomib is stored at 26 mM** (Cond-000100 and Cond-013963) exactly as Data Set 4 prints it, although 26 mM in a 200 uL culture is implausible; no other number is sourced.
- **The vehicle fraction is 1% DMSO** by the screen's deferral to the optimization section; the screen restates no fraction.
- **Every dose has `basis=fixed`**; ng/mL cells are stored in ug/mL (x 1e-3, exact) and the two MMS cells as 0.004 basis-free percent (#764).
- **The medium is loader-local** (`YP_GALACTOSE_LIQUID`, base `YP`): the library's `YP_GALACTOSE` is Bloom 2019's solid plate with Bloom's quote.

### Schema impact

`scripts/schema_impact_check.py --base origin/main` (main `628a0412a`): "No schema contract changes". The only shared-code changes are additive: the InChI route in `compound_identity.py` and `runners.verify_environment_response_entry`, the family loop's body, which the loader's `verify_build` calls so `build_dataset_lmdb --verify` covers this store.

### Issues and the candidate verdict

- #1025 (`dataset`): Data Set 2's sensitized single-mutant fitness sheet, 3,032 genes, not loaded.
- #726 (comment): the 10 conditions whose curated table key disagrees with the released InChI at the first block.
- `database/candidates/piotrowskiFunctionalAnnotationChemical2017.json` re-gated on this branch: admissible, G1 to G5 pass, G3 now 54 files over library + raw. The audit's G3 replicate evidence is corrected (3 by design, 1 to 3 per released profile) and the replicate-collapsing, matrix-title and sign quotes are added. `candidate-gate audit`: ok 79, binary 3 (the docx quotes are sha256-only). `candidate-gate ledger` refused ("already has '## 2026.10.11 - Candidate gate ledger'; one ledger per day"), so the verdict reaches the ledger with the next day's run.
