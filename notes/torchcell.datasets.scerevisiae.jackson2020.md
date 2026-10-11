---
id: 20ufu16owzt9wt854f61xb5
title: Jackson2020
desc: ''
updated: 1791732525678
created: 1791732525678
---

## 2026.10.11 - Jackson 2020 TF-deletion single-cell atlas: 784 pseudobulk records, loaded

Loader: `torchcell/datasets/scerevisiae/jackson2020.py` (`TfDeletionPseudobulkJackson2020Dataset`, dev store `data/torchcell/tf_deletion_pseudobulk_jackson2020`). Source: Jackson CA, Castro DM, Saldi GA, Bonneau R, Gresham D. eLife 2020;9:e51254, doi:10.7554/eLife.51254, citation key `jacksonGeneRegulatoryNetwork2020`, GEO GSE125162. Yeast expansion wave 2, schedule row 6. Verifier: [[torchcell.verification.pseudobulk_umi_count]]. Adapter: [[torchcell.adapters.jackson2020_adapter]]. Every count below that is not a verifier row comes from `experiments/036-dataset-fixes-before-kg-build/scripts/jackson2020_store_inventory.py` (`results/jackson2020_store_inventory.json`), which reads the built store and its `preprocess/release_ledger.json`.

### What the release is

eLife Source code 2, `elife-51254-code2.tsv.gz`: "The raw count matrix as a gzipped TSV file. This file contains 38,225 observations (cells). Doublets and low-count cells have already been removed; gene expression values are unmodified transcript counts after deartifacting using UMIs" (paper.md line 422, sha256 `453924f3...`). Measured: 38,225 rows x 6,829 count columns plus five metadata columns (`Genotype` = the strain label, `Genotype_Group`, `Replicate` = the array well, `Condition`, `tenXBarcode`). Every count is a non-negative integer; they sum to 83,703,440, the paper's "we recovered 83,703,440 transcript counts from a total of 38,225 individual cells." (line 75). The copy of the matrix inside Source code 1 (`data/103118_SS_Data.tsv.gz`) is byte-identical (same sha256 `4bb210b7...`).

The count header is R's `make.names` form of the Ensembl R64-1-1.93 gene ids (`YDL247W.A`, `X15S_rRNA`, `snR7.L`). Undoing it (drop the `X` before a leading digit, `.` to `-`) gives exactly the header of `TRIZOL_BULK.tsv` from the same release, all 6,829 names in the same order, which the loader requires and the verifier re-checks. Two of the columns are the drug cassettes ("The antibiotic resistance marker cassettes was added to both the FASTA and GTF files using command line tools.", line 265): `KANMX` and `NATMX`.

### The record, and why a new phenotype leaf

One `PseudobulkUmiCountExperiment` per (strain, condition): the summed raw UMI counts of that strain's cells in that flask, exactly the paper's own pseudobulk: "The raw, unmodified UMI counts of all cells from each biological replicate (with the same strain barcode) within a specific environmental growth condition were summed, resulting in 72 samples per condition (six biological replicates for each of the 12 transcription factor deletions)." (line 351). The six constructions of a genotype are six records.

No served expression leaf holds raw UMI counts: `PseudobulkExpressionPhenotype` is a log2 fold change against wild type (Nadal-Ribelles), which this release does not contain and which would need a normalization and a DE model the paper ran (DESeq2) but did not release; `RNASeqExpressionPhenotype` requires TPM, which a 3' UMI assay does not produce; `MrnaNumberFractionPhenotype` is for a count-less release. `PseudobulkUmiCountPhenotype` (additive, delimited block in `schema.py`) holds:

- `umi_count`: every gene feature with a nonzero sum, keyed by R64 id; `n_features` and `feature_universe_sha256` pin the 6,827-feature universe, so an absent key is a measured zero.
- `marker_umi_count`: `KANMX` and `NATMX`, zeros kept, never mixed into the gene dict.
- `total_umis`, `n_cells`, and the cells' `cell_barcodes` and `cell_total_umis` (aligned, release row order), so every per-cell library size survives the sum. The validator requires the per-cell totals and the two dicts to each sum to `total_umis`.

The reference of each condition is the six HO-deletion control strains of that flask with their cells pooled ("In addition, we constructed six biological replicates of the wild-type control in which we deleted the neutral HO locus.", line 60). The six control strains are also records of their own. The reference genome is a `StrainReferenceGenome` (`FY4/FY5`, diploid, MATa/MATalpha), the environment a `CultureEnvironment`.

### Counts (built store, 2026-10-11)

| quantity | value |
|---|---|
| records | 784 = 72 strains x 11 conditions - 8 pairs with no cell |
| references | 11 (one per condition) |
| strains / deleted genes | 72 / 12 (11 TFs + HO), 6 strains each |
| cells | 38,225 (all served; 3,306 are HO-control cells) |
| UMIs | 83,703,440 (all served); KANMX 199,778, NATMX 617,833 (0.98% of UMIs) |
| cells per record | min 1, median 23, max 266 |
| UMIs per record | min 594, median 43,850.5, max 749,705 |
| nonzero genes per record | min 298, median 4,028.5, max 5,561 |
| gene features | 6,827 in the universe, 6,518 nonzero in at least one record |

The eight (strain, condition) pairs with no cell, although the Results say "Each of the 72 strains is found in each of the 11 conditions" (line 85): `stp2_E12` and `stp2_G12` in YPD, RAPA and CSTARVE, `gcn4_G10` in NLIM-GLN, `gcn4_A10` in NLIM-NH4. They are absent from the store, listed in the ledger, and the verifier requires exactly these to be absent.

| condition | release label | medium (library key) | h to harvest | records | cells | reference cells | cells/mL at harvest |
|---|---|---|---|---|---|---|---|
| YPD | `YPD` | `YPD_LIQUID` restated | 4 | 70 | 11,037 | 976 | 1.4e7 |
| DIAUXY | `YPDDiauxic` | `YPD_LIQUID` restated | 10 | 72 | 3,332 | 314 | not stated |
| RAPA | `YPDRapa` | `YPD_LIQUID` restated + 200 ng/mL rapamycin, last 30 min | 4 | 70 | 11,805 | 932 | 1.2e7 |
| YPEtOH | `YPEtOH` | `YP_ETHANOL_LIQUID_JACKSON2020` | 4 | 72 | 1,427 | 134 | 1.0e7 |
| MMD | `MinimalGlucose` | `MMD_JACKSON2020` | 4 | 72 | 1,894 | 204 | 1.1e7 |
| MMEtOH | `MinimalEtOH` | `MMETOH_JACKSON2020` | 4 | 72 | 581 | 26 | 0.7e7 |
| NLIM-GLN | `Glutamine` | `NLIM_GLN_JACKSON2020` | 4 | 71 | 1,644 | 148 | 0.5e7 |
| NLIM-PRO | `Proline` | `NLIM_PRO_JACKSON2020` | 4 | 72 | 853 | 65 | 0.4e7 (OCR garbled, see below) |
| NLIM-NH4 | `AmmoniumSulfate` | `NLIM_NH4_JACKSON2020` | 4 | 71 | 1,111 | 121 | 0.8e7 |
| NLIM-UREA | `Urea` | `NLIM_UREA_JACKSON2020` | 4 | 72 | 992 | 147 | 0.5e7 |
| CSTARVE | `CStarve` | `CSTARVE_JACKSON2020` | 4 | 70 | 3,549 | 239 | 0.1e7 |

The release label to Table 1 abbreviation map is the authors', from `figure_common_settings.R` in Source code 1: `CONDITION.LEVELS <<- c("CStarve", "Urea", "AmmoniumSulfate", "Proline", "Glutamine", "MinimalEtOH", "MinimalGlucose", "YPEtOH", "YPDRapa", "YPDDiauxic", "YPD")` and `CONDITION.LABELS <<- c("CSTARVE", "NLIM-UREA", "NLIM-NH4", "NLIM-PRO", "NLIM-GLN", "MMEtOH", "MMD", "YPEtOH", "RAPA", "DIAUXY", "YPD")`. Without it, `AmmoniumSulfate` could be read as MMD (20 mM) or NLIM-NH4 (0.8 mM). The harvest densities are kept in the ledger, not on records (no schema field holds a culture's harvest density); NLIM-PRO's value is `$\phantom { - } 0 . 4 \phantom { - } \Theta ^ { 7 }$` in the OCR and reads 0.4e7 by the sentence's pattern.

### Sourced values

| value | quote | where |
|---|---|---|
| background | "All yeast strains were generated from the prototrophic FY4 (MATa) or FY5 (MATa) background strains." (the OCR drops the alpha; Table S2 rows 2 and 3 give `FY4 \| MATa` and `FY5 \| MATα`) | paper.md line 211 |
| whole-ORF deletion | "We engineered TF gene deletions by precisely excising the entire TF open reading frame and introducing a unique transcriptional barcode that enables multiplexed analysis of genotypes using scRNAseq." | line 54 |
| six constructions | "We engineered eleven different TF knockout genotypes, using six independently constructed biological replicates for each genotype." | line 60 |
| homozygous diploid | "Homozygous diploids were constructed by mating to a strain containing the same TF knockout marked with a nourseothricin drug resistance cassette." | line 60 |
| pre-culture | "The second recovery plate was pinned to a 96 well roundbottom plate containing $2 0 0 ~ \mu \ L$ YPD in each well and grown overnight at $3 0 ^ { \circ } \mathsf { C }$ ." | line 233 |
| 50 mL shake flask | "of the washed cells were used to inoculate $5 0 ~ \mathsf { m L }$ of the relevant media for the specific experimental condition in a shake flask." | line 233 |
| harvest times | "All cultures were harvested $4 \ : \mathrm { h r }$ after inoculation, except for the culture harvested after the diauxic shift [DIAUXY], which was harvested $1 0 \ : \mathrm { h r }$ after inoculation." | line 79 (Table 1 legend) |
| rapamycin | "The experiment treated with rapamycin was grown for 3 hr and $3 0 ~ \mathsf { m i n }$ in YPD, and then $1 0 ~ \mu \ L$ of rapamycin stock $1 \ m g / \mathsf { m L }$ Millipore #553210 in ethanol) was added to a final concentration of $2 0 0 ~ \mathrm { { n g / m L } }$ ." | line 233 |
| batches | "This final data matrix is assembled from 11 independent single-cell sequencing batches, each corresponding to a single shake flask with a different growth condition." | line 267 |
| chemistry | "Single cell library preparation was done using the $ 1 0 \times$ Genomics Chromium ${ \boldsymbol { 3 ^ { \prime } } }$ v2 Single Cell Gene Expression Kit" | line 247 |
| reference annotation | "The reference genome was obtained from Ensembl (Version R64-1-1) as a FASTA file, and the reference annotations were obtained from Ensembl (Version R64-1-1.93) in GTF format." | line 265 |
| strains, barcodes | Supplementary file 1, sheet `Supplemental Table 2 - Strains` (row renderings, e.g. `N/A \| gat1_H5 \| MATa/MATα Δgat1::KanMX/Δgat1::NatMX \| TGGGAGTGCGCCGTCGGCG \| BC_DEL_1_H5 \| This work`) | `si/si6.xlsx`, sha256 `6c96fdf4...` |
| media | Supplementary file 1, sheet `Supplemental Table 4 - Media`, every row quoted on its component in `torchcell/datamodels/media.py` | same workbook |

### Genotype

Each record's genotype is one `BarcodedKanMxDeletionPerturbation`: the systematic name (the eleven TF ORFs from `figure_common_settings.R`'s `nitrogen.tfs`, each checked to be a current R64 gene whose standard name is the label's gene; `YDL227C` for HO through the genome), the 19-nt barcode of the KanMX allele (Table S2, equal to Source code 4's `bcdel_1_barcodes.fasta` for all 72), `cassette="KanMX/NatMX"` (Table S2's allele notation), `collection` naming the BC_DEL_1 array, and `construction` = the strain label, plate `BC_DEL_1` and well.

Seven diploid rows of Table S2 name the other STP gene than their strain label: `stp1_H11` reads `Δstp2::KanMX/Δstp2::NatMX` and the six `stp2_*` rows read `Δstp1`. For each of the seven the MATa haploid parent row with the same barcode (`MATa Δstp1::KanMX` for `stp1_H11`, `MATa Δstp2::KanMX` for the `stp2_*`), the strain label and the release's `Genotype_Group` all name the label's gene, so the label is followed. The seven are pinned in the loader (`TABLE_S2_GENOTYPE_DISAGREEMENTS`) and in the ledger; a workbook with any other disagreement set is refused.

### Features

| status (`resolve_gene_name` against R64-4-1) | count |
|---|---|
| current gene | 6,599 |
| valid non-gene locus (TE genes, ncRNAs, pseudogenes, blocked reading frames) | 227 |
| retired | 1 (`YIL080W`) |

114 ids are remapped to their current R64 id (the ncRNA common names `snR*`, `RDN*`, `15S_rRNA`, `RPM1` and the like to their `YNC*` ids). Four pairs of R64-1-1 ids were merged into one current feature (`YIL168W` into `YIL167W`, `YIR044C` into `YIR043C`, `YFL057C` into `YFL056C`, `YAR062W` into `YAR061W`); the shared retain-all policy (`gene_name_reconcile.reconcile_systematic_names`) keeps both members of each pair under their source ids so two released columns never become one key. So five keys are not R64-4-1 features: `YIL080W` and the four merged-away ids. L4 containment 6,822 / 6,827 = 0.99927 (floor 0.999).

### Owner decisions taken by recommendation

- Raw summed UMI counts, not a log2 fold change: the release is counts, and the paper's pseudobulk is a plain sum; a fold change would add a normalization and a pseudocount nobody released.
- A new leaf, not a field on `PseudobulkExpressionPhenotype`: making its log2 ratio optional would weaken a served class and move Nadal-Ribelles 2025's closure.
- Overlap with Boocock 2025: that branch (`feat/scerevisiae-boocock2025`, unmerged) adds `SingleCellExpressionPhenotype`, also summed UMI counts with a pinned universe, but it requires per-cell cell-cycle stage counts this release does not carry. The two leaves should become one before KG 4.0 (an owner call; field names were kept compatible: `umi_count`, `total_umis`, `n_features`, `feature_universe_sha256`, `n_cells`).
- Temperature is a typed gap (`not_reported_by_primary`), not 30 C: "These flasks were grown for 4 hr." names none. Every other growth step (recovery plates, the 96-well pre-culture, the array steps, the bulk-RNA flasks) is stated at 30 C, so 30 C is likely and unconfirmed.
- The HO deletion is the wild type: the paper calls the Δho strains "the wild-type control", so the reference is their pooled cells and they are also records with a `YDL227C` deletion.
- FY4/FY5 carry no background allele: the paper says "prototrophic" and names none; the S288C ancestry of the FY strains (Winston et al. 1995) is not mirrored, and the note says so on the background's provenance.
- Genotype leaf: `BarcodedKanMxDeletionPerturbation` with `cassette="KanMX/NatMX"`, one leaf per strain. Two leaves (a KanMX and a NatMX deletion of one gene) would read as a double mutant to every perturbation counter.
- Rapamycin: `SmallMoleculePerturbation`, 0.2 ug/mL, solvent ethanol at 0.02% v/v (10 uL into the 50 mL flask, the two stated volumes); its 30-min exposure is in the perturbation's description because no field holds a perturbation's own duration. `duration_hours` is the time from inoculation to harvest.
- NLIM-UREA's Table 4 block is headed "Nitrogen Limiting Minimal Media (Ammonium Sulfate)" (a copy of the block above); its nitrogen row is "800 µM Urea" and Table 1 names the condition NLIM-UREA, so the urea row is the medium.
- Compound table: sodium molybdate dihydrate and potassium iodide had no row; both were curated from PubChem (input list `compound_identity_inputs/jackson2020.txt`) and spliced into the table, so the trace metals are twenty typed components rather than a quoted sub-mix. A full curator pass on 2026-10-11 does not reproduce the committed table (PubChem drift renames 9 rows), so the full pass was not committed.
- Boric acid has no yeast-GEM 9.0.2 exchange; `tests/torchcell/metabolism/test_media.py` lists it as a documented exception rather than mapping it to a species the model lacks.

### Refused, and released but not loaded

Nothing released was refused: 38,225 of 38,225 cells and 83,703,440 of 83,703,440 UMIs are in a record. Not loaded: `TRIZOL_BULK.tsv` (bulk 3' RNA counts of the six HO-deletion strains in YPD, Source code 1; used here only to corroborate the feature ids), the simulated negative-control matrix `110518_SS_NEG_Data.tsv.gz`, the per-condition inferred networks (derived analyses), and the GEO raw reads.

### Deposits

Raw mirror `$DATA_ROOT/torchcell-raw/jacksonGeneRegulatoryNetwork2020/` (`manifest.json`, retrieved 2026-10-11 from the PMC Article Datasets bucket `PMC7004572.1/`, bytes equal to the library mirror's `si/` copies):

| file | sha256 | retrieval |
|---|---|---|
| `data/elife-51254-code2.tsv.gz` | `4bb210b7a1e2e9724e475ab516de646fbc36f2638ac6ade3bf15bf66ffb8c011` | `pmc_cloud_object` |
| `si/elife-51254-supp1.xlsx` | `6c96fdf425f9d07c92d6bad8f44e54e1bebca6b83d06a7b7b3fbb06ea1923737` | `pmc_cloud_object` |
| `code/figure_common_settings.R` | `bf6b384c4f341d1a2c4a13024829967593b697bdfa109fa728f686d2c78acf9c` | tar member of `elife-51254-code1.tar.gz` (`ee6d8c92...`) |
| `data/TRIZOL_BULK.tsv` | `95b5823d5da02d8c762399f5f537d8a92a16a8f8267bcad291a4c47d29944b1b` | tar member of the same |
| `data/bcdel_1_barcodes.fasta` | `f16e245dde905c79f487b32c56a8eaeb07a5b530b71df8033afafa7c24904359` | tar member of `elife-51254-code4.tar.gz` (`371d9e79...`) |

The tar-member records name `torchcell.literature.retrieve.tar_member`, which lands with the Boocock 2025 branch. No genomes-tier deposit: the strains are S288C-derived deletions on the R64 coordinate system, and Source code 4's marker-augmented R64-1-1 FASTA is not consumed.

### Schema impact

`scripts/schema_impact_check.py --base origin/main`: 8 changed symbols, all additive (`PseudobulkUmiCountPhenotype`, `...Experiment`, `...ExperimentReference`, the three type unions, the two type maps); 10 dataset groups stale through the unions, 0 breaking.

### L0-L4 verification (dev store rebuilt with `--retire-existing --verify`, 2026-10-11)

| level | rule | result |
|---|---|---|
| L0 | structural | 784 records and 11 references validated |
| L1 | count | 784 of 784 |
| L1 | strain_condition_uniqueness | 784 unique pairs, exactly the pairs holding a cell |
| L1 | strain_identity | 72 strains, one barcode and one gene each, barcodes = Table S2 = Source code 4, 12 genes x 6 |
| L1 | provenance_gaps | 784 documented gaps (the temperature, one per record) |
| L1 | canonical_gene_names | 12 systematic names, each current |
| L2 | release_fidelity | every record equals a fresh re-sum of the matrix (genes, markers, totals, barcodes, per-cell totals) |
| L2 | reference_fidelity | 11 references, each the pooled HO-control cells |
| L2 | release_totals | 38,225 cells and 83,703,440 UMIs, the paper's numbers |
| L2 | uncertainty_sanity | pass (no uncertainty field) |
| L3 | measurement_type_consistent | `scrnaseq_pseudobulk_umi_count` |
| L3 | feature_universe | one 6,827-feature universe, every key inside it |
| L3 | conditions_documented | 11 environments, each a Table 1 row and an authors' label |
| L3 | sourced_values | 134 quotes audited (22 paper.md, 110 workbook rows, 2 settings lines) |
| L3 | compound_identity, media_compound_identity | 140 and 11,214 references, all identified |
| L3 | media_membership | 572 records on a library medium, 212 on the restated YPD |
| L4 | feature_containment | 0.99927 (floor 0.999) |
| L4 | feature_ids_corroborated | un-mangled header = `TRIZOL_BULK.tsv` header, 6,829 in order |

`check_store`: fresh. Tests: `tests/torchcell/datasets/scerevisiae/test_jackson2020.py` (miniature release in `_jackson2020_release.py`; two `--data` tests on the real mirror and store), `tests/torchcell/verification/test_pseudobulk_umi_count.py`, `tests/torchcell/datamodels/test_pseudobulk_umi_count_schema.py`, `tests/torchcell/adapters/test_jackson2020_adapter.py`, `tests/torchcell/adapters/test_pseudobulk_umi_count_node.py`.

### Candidate gate

`python -m torchcell.candidates gate --row "Jackson 2020" --table yeast --phenotype-class PseudobulkUmiCountPhenotype --write`: G1 to G5 pass, `admissible` (`database/candidates/jacksonGeneRegulatoryNetwork2020.json`). The ledger command refused a second 2026.10.11 section ("one ledger per day"), so no ledger section was written.
