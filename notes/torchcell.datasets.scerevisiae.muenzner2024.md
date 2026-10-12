---
id: vvr0q4eyqsxbsc7japirrwq
title: Muenzner2024
desc: ''
updated: 1791727384123
created: 1791727384123
---

## 2026.10.11 - Initial loaders, raw deposit, dev stores and verification

Yeast expansion wave 1, row 1 (`notes-tex/database/database-expansion-100/`). Muenzner et al. 2024, "Natural proteome diversity links aneuploidy tolerance to protein turnover", Nature 630:149-157, doi 10.1038/s41586-024-07442-9, PMID 38778096 (PMC11153158), citation key `muenznerNaturalProteomeDiversity2024`, Zotero group item BC5L4MZF.

### What is built

| store (`$DATA_ROOT/data/torchcell/...`) | class | records | per record |
|---|---|---|---|
| `proteome_muenzner2024` | `ProteomeMuenzner2024Dataset` -> `ProteinAbundanceExperiment` | 794 | `protein_abundance` over up to 1,576 proteins (systematic ORF), 1,214,201 values in total |
| `protein_turnover_muenzner2024` | `ProteinTurnoverMuenzner2024Dataset` -> `YeastProteinTurnoverExperiment` | 55 | `degradation_rate` = released `kdp` (1/h) and `half_life` = released `HL` (h), 58,061 (isolate, protein) pairs over 1,836 proteins |

Genotype: each isolate is a perturbation set off S288C R64 built by the Caudal 2024 construction (sequence variants, natural presences and natural absences from the Peter 2018 tier), through `caudal2024`'s own helpers (`PeterIsolateGenotypes`); `caudal2024.py` is not edited. Gene sets: 7,782 (proteome store), 7,045 (turnover store).

`YeastProteinTurnoverExperiment` / `...Reference` are new (schema commit 4913b258e): the existing turnover pair narrows `genome_reference` to the bacterial `AssemblyReferenceGenome`, whose assembly set is a closed bacterial vocabulary, so an S288C-referenced record cannot be written with it, and `test_schema` pins that pair as assembly-pinned. Schema impact vs origin/main: additive, 0 breaking; the union growth (`ExperimentType`, `ExperimentReferenceType`, both type maps) marks the stores that close over `ExperimentType` stale (Caglar 2017 x3, Gupta 2024, Ishii 2007, Li 2014, Schastnaya 2021, De Siqueira 2025 x4, Lim 2022, Volk 2021 private, Hoepfner 2014), which the KG 4.0 full rebuild remakes anyway.

### Sources and deposits

- Raw mirror: `$DATA_ROOT/torchcell-raw/muenznerNaturalProteomeDiversity2024/si/41586_2024_7442_MOESM3_ESM.xlsx`, sha256 `1e78de77ffeb5ffb4f64f3999765db62ec9f6b6729a71f75feb1b955f5d6e005`, 24,524,259 bytes, the Supplementary Tables workbook (SI Tables 1-20). Retrieval: `pmc_cloud`, retriever `torchcell.literature.retrieve.pmc_cloud_object`, key `PMC11153158.1/41586_2024_7442_MOESM3_ESM.xlsx`, retrieved 2026-10-11T10:43:58Z by the library SI capture; copied by `python -m torchcell.datasets.scerevisiae.muenzner2024 --deposit-from <library si/si3.xlsx>` (`deposit_raw_mirror`), which verifies the pin before writing. Manifest `manifest.json` beside it.
- Genomes tier (no new deposit): `peter2018_1011_assemblies` `allReferenceGenesWithSNPsAndIndelsInferred.tar.gz`, `genesMatrix_PresenceAbsence.tab.gz`, `genesMatrix_CopyNumber.tab.gz`, pinned by that tier's manifest; SGD R64-4-1 chromosome FASTA for the reference slices.
- Not deposited: MassIVE MSV000090435 (natural collection), PRIDE PXD044526 (disomes) and PXD048219 (turnover) hold raw spectra and search output; the released matrices are SI Tables 3, 4 and 17.
- Text sources quoted below: library `paper.md` (sha256 `1534d1e3d1218b3a5e34c3b57c22cf038919cfa318e6049bd67401587aa86cef`) and the reporting summary `si/si2.md` (sha256 `4b2c31e1595f606bde9052f549129952ba03933d521b0042ee48b24c9a357e06`). The worksheet captions are checked against the workbook bytes on every build (`SheetSpec`), so a layout change raises.

### Sourced values (verbatim; `SOURCED_VALUES`, audited by `test_every_sourced_value_audits_against_the_mirror`)

- Proteome medium (paper.md, Methods, High-throughput cultivation): "the collection was grown on agar plates containing synthetic minimal (SM) medium ( $\cdot 6 . 8 \mathsf { g l } ^ { - 1 }$ yeast nitrogen base, $2 \%$ glucose, without amino acids)". The PDF text layer reads "6.8 g l−1" too. The same Methods print 6.7 g/L for the SILAC and ubiquitinome SM; recorded as read (`MUENZNER_SM`, YNB line 6.8 g/L), not corrected.
- Proteome culture: 30 C, 8 h, 1,000 rpm in deep-well plates "to exponential phase (Supplementary Table 2)".
- Turnover medium: "were grown on synthetic medium containing $6 . 7 \mathrm { g } 1 ^ { - 1 }$ yeast nitrogen base, $2 \%$ glucose and $8 0 \mathrm { m g l ^ { - 1 } L }$ -lysine", switched to "$8 0 \mathrm { m g l ^ { - 1 } }$ heavy $[ ^ { 1 3 } { \bf C } _ { 6 } / ^ { 1 5 } { \bf N } _ { 2 } ]$ lysine" before sampling at 90, 135 and 180 min at 30 C (`MUENZNER_SM_LYS8` = library SM + 80 mg/L L-lysine-13C6,15N2; the isotopologue has no identity-table row, so its InChIKey is a typed gap).
- Rate definition: "we therefore calculated turnover rates (kdp) and the corresponding half-lives without doubling time or dilution rate correction (Supplementary Table 16)" and "Half-lives were calculated from the resulting kdp as log(2)/kdp (Supplementary Table 17)". The SI Table 17 caption: "kdp denotes the turnover rate in (1/h), HL the protein half-life (in h)". So `measurement_type = dynamic_silac_lys8_kdp_per_hour_no_dilution_correction`: total replacement, degradation plus dilution by growth.
- Replication (reporting summary): "the large dataset (> 1000 isolates) was acquired once", so `n_replicates = 1` per protein and no SE.
- Missing values: SI Table 3's caption says "(non-imputed)"; the release writes a missing quantity as the text `NA` (37,667 cells), which becomes a key-absent protein. The paper's own analyses imputed: "Missing values (less than $4 \%$ of all values) were imputed using $k$ -nearest neighbours (KNN) imputation65."
- Quantity: DIA-NN MaxLFQ after "batch correction was performed at the precursor level by bringing median precursor quantities of each batch to the same value" (`measurement_type = swath_ms_maxlfq_precursor_batch_corrected_quantity`, linear). Distinct from Messner 2023's type on purpose: different instrument, library and normalization.

### Reference (the paper's own choice)

"Relative abundances for transcriptomic and proteomic data were calculated gene-wise as the $\mathbf { l o g } _ { 2 }$ ratio between a gene’s mRNA or protein abundance in a given strain and the median mRNA or protein expression value of the respective gene across all euploid strains (‘all-euploid strain’ method)." The proteome `phenotype_reference` is therefore the per-protein median over the 518 euploid isolates of SI Table 7 (the paper's integrated set, after its SI Table 5 and 6 exclusions), on the released non-imputed values, with `n_replicates` = how many of the 518 measured that protein (minimum 401). One shared reference; it covers all 1,576 proteins. Differs from the paper's number only by the imputation the release omits.

Turnover: "We calculated the median quantile-normalized turnover rate for each protein when expressed from aneuploid chromosomes, from euploid chromosomes of aneuploid isolates or from euploid chromosomes of euploid isolates." The turnover reference is the per-protein median of the released `kdp` over the 10 euploid isolates of the 55 (not quantile-normalized: the stored values are the released ones). 312 of the 1,836 proteins (939 cells) were measured in no euploid isolate and have no reference key, never an imputed one.

### Isolates: matching, refusals, overlap with Caudal 2024

- The release spells the `SACE_` codes with a dot (`SACE.YAB`); `peter_code` maps `.` to `_`. Measured: all 794 matched isolates carry identical Ploidy and Aneuploidies in SI Table 1 and Peter 2018 Table S1, and 755 identical alternative strain names; the 39 that differ are Excel date mangling (`09-juil` for `9-7`) or longer forms of the same name (`CLI_1` vs `malade Hc6553`), none among the 67 `SACE_` isolates.
- Refused (no Peter genome, so no genotype): `Reference` (alternative name FY4, the S288C-derived control) and `XTRA.DGU` (KS11). Both in the ledger with the reason.
- **Overlap with the built Caudal store: 759 isolates** (`L4 caudal_isolate_overlap`, measured against `caudal_pantranscriptome2024/preprocess/data.csv`). The paper's "For 761 isolates, both proteomes (this study) and transcriptomes6 were available" counts the two refused isolates, whose transcriptomes exist in SI Table 18's Final_set (`XTRA_DGU`, `FY4-6`). The 35 proteome isolates without a Caudal record (ADA, ADH, ..., SACE_YDN) have no Final_set transcriptome. All 55 turnover isolates are in the Caudal store.
- `L4 caudal_genotype_agreement`: four sampled shared isolates (first, middle, last, first `SACE_`) carry the identical perturbation list in both stores.

### Karyotype: the schema gap (#1001)

Basal ploidy (1n to 5n) and whole-chromosome aneuploidy are genomic content with no genotype slot: every perturbation leaf is gene-keyed, and `ReferenceGenome.ploidy` is haploid/diploid. Not forced into per-gene `CopyNumberVariantPerturbation` leaves. Measured in the proteome store: ploidy 1n 90, 2n 636, 3n 36, 4n 29, 5n 3; 168 aneuploid (53 complex), 231 chromosome gains and 48 losses; 45 of the 55 turnover isolates are aneuploid. Each record's karyotype is in `preprocess/muenzner2024_ledger.json` (`karyotypes`, with the SI Table 7 membership and the SI Table 5 or 6 flag the paper excluded 80 + 66 isolates on).

### Ledger (`preprocess/muenzner2024_ledger.json`)

- Proteome: 796 release isolates, 1,576 proteins, 2 refused, 794 built; 1,251,344 cells of which 1,214,201 measured and 37,143 missing (key-absent).
- Turnover: 58,061 rows, 55 isolates, 0 refused, 55 built; HL equals ln 2 / kdp to a maximum relative error of 5.7e-15; SI Table 16 doubling times kept per isolate (83.8 to 191.2 min, median 126.2) because kdp is not corrected for them and no record field carries a growth rate.

### Not loaded (#1003)

SI Table 4 (38 proteomes of W303 lab disomes: no W303 assembly, and the aneuploidy gap), SI Table 13 (two isolates over five growth stages, 30 samples, a different instrument and pipeline), SI Table 15 (ubiquitinome, 8 isolates x 3: no phenotype class), SI Table 2 (harvest OD). SI Tables 8-12 and 14 are derived analyses.

### Verification (L0-L4, `verify_build`, written to each store's `preprocess/verification_report.json`)

Proteome (`proteome_muenzner2024`), PASS:

| level | check | result |
|---|---|---|
| L0 | structural | 794 records validate against `EXPERIMENT_TYPE_MAP[experiment_type]` |
| L1 | count | 794 = 794 |
| L1 | strain_uniqueness | 794 isolates, one record each |
| L1 | genotype_strain_consistency | every genotype names one isolate |
| L2 | value_fidelity | 1,214,201 values finite and > 0 |
| L2 | single_acquisition | `n_replicates` = 1 throughout |
| L3 | shared_reference | one reference over 1,576 proteins, finite and > 0 |
| L3 | reference_covers_every_key | 0 cells without a reference value |
| L3 | measurement_type_consistent | one type |
| L4 | gene_containment_sgd | 1,576 of 1,576 are S288C R64 systematic names |
| L4 | caudal_isolate_overlap | 759 (expected 759) |
| L4 | caudal_genotype_agreement | 4 of 4 sampled genotypes identical to Caudal |

Turnover (`protein_turnover_muenzner2024`), PASS:

| level | check | result |
|---|---|---|
| L0 | structural | 55 records validate |
| L1 | count / strain_uniqueness / genotype_strain_consistency | 55, one record per isolate |
| L2 | value_fidelity | 58,061 rates finite and > 0 |
| L3 | half_life_is_ln2_over_rate | max relative error 5.68e-15 |
| L3 | shared_reference | one reference over 1,524 proteins |
| L3 | uncovered_cells_match_ledger | 939 = 939 |
| L3 | measurement_type_consistent | one type |
| L4 | gene_containment_sgd | 0.9989 of 1,836; outside: `R0040C` (2-micron plasmid ORF) and `YFL057C` |

L0 validates each record against the class its own `experiment_type` names rather than the `ExperimentType` union: measured 21.8 s for one 4,768-perturbation proteome record through the union against 0.67 s through `ProteinAbundanceExperiment`.

### Owner decisions taken by recommendation

1. The proteome reference is the paper's euploid median over SI Table 7, not a population mean over the built isolates (Caudal's choice): the paper states it, and it is the baseline every relative number in the paper uses.
2. The turnover reference is a median of the released, not quantile-normalized, rates; proteins without a euploid measurement stay key-absent.
3. A new yeast turnover experiment pair instead of widening the served bacterial pair, so no served closure changes.
4. 6.8 g/L YNB recorded as printed for the proteome medium, although the same paper prints 6.7 g/L for its other SM cultures.
5. Aneuploidy and ploidy filed as a schema gap (#1001) rather than written as per-gene CNV leaves; the karyotype travels in the ledger.
6. FY4 (`Reference`) refused although it is S288C-derived: it has no Peter genome, and the brief refuses any isolate without one.
7. The candidate-table row now says "796 natural isolates (794 of the 1,011-isolate panel)", so G2 can name the host set; tables regenerated by the script.

### Tests

`tests/torchcell/datasets/scerevisiae/test_muenzner2024.py` (synthetic workbook + the Caudal synthetic Peter tier, end-to-end builds, ledgers, parsing refusals, media, deposit, download, both gates, `verify_build` against a synthetic Caudal store; `--data`: quote audits and the measured workbook counts), `tests/torchcell/adapters/test_muenzner2024_adapter.py`, plus pins in `test_runners.py`, `test_raw_pins.py`, `test_build_time_projection.py`, `test_adapter_schema_consistency.py` and `test_schema.py`.

## 2026.10.12 - The shared isolate genotype, stored as released (#987, #988, PR #1058)

The genotype of every isolate in this store is now built by [[torchcell.sequence.natural_isolate]], the one module every natural-isolate loader of the Peter 2018 panel shares, and audited by [[torchcell.verification.natural_isolate_genotype]]. Beyond the sequence variants, natural presences and natural absences it already carried (unchanged, the release's own representation off the 1,011-genome set), each genotype now stores what Peter 2018 releases about the isolate, as released and keyed as the release keys it: every call of the two loss-of-function matrices (`NaturalLossOfFunctionPerturbation`, keyed by the loss-of-function matrix's `NAME`, by the code's pangenome ORF name in the frameshift matrix, or by `Gene` verbatim on a frameshift row released without a code), the HO deletion Supplementary Table 1 marks (`DeletionPerturbation` of YDL227C, kept beside whatever the presence matrix states about HO), and Table S1's ploidy, zygosity and aneuploidy on `Genotype.natural_isolate` (each the table's value or a typed gap). No call is dropped, re-keyed or annotated by a lookup in another file or in the S288C annotation (owner rule, 2026-10-12: the data is the data).

Rebuilt with the code of `073d4b74a` (build manifests: `torchcell_commit` `6ffa14d10`, a notes-only commit on top, `torchcell_dirty: false`) with `build_dataset_lmdb --retire-existing --verify` under slurm (array 4292: task 1 the proteome store, 18 min; task 2 the turnover store, 2 min); the predecessor is kept as `processed.superseded.*`. Record counts unchanged: 794 proteome and 55 turnover records, one per isolate.

**These two generations are no longer the dev stores' `processed/`.** The #1045 audit branch's slurm array 4348 (tasks 2 and 3, started 11:41 and 11:45, from that branch's own source snapshot) retired them and rebuilt both stores with the pre-#1058 genotype code (the proteome task ended FAILED, the turnover task COMPLETED), minutes after the builds below passed `--verify` (the dev stores are shared across worktrees). Both #1058 generations are kept beside them, under the names given in the headings; the identity measurement reads them there. The next build from main after #1058 lands rebuilds both stores.

### `proteome_muenzner2024` (kept as `processed.superseded.20261012-114150`)

Genotype ledger: 794 isolates, 3,988,281 sequence variants, 66,957 natural presences, 105,431 natural absences, 85 HO deletions; loss-of-function calls: nonsense_or_sift 185,451 on 3,809 genes, homozygous_frameshift 114,589 on 2,309 gene keys (7,053 on rows released without a code).

All 16 rows pass (`preprocess.superseded.20261012-114150/verification_report.json`):

| level | rule | result |
|---|---|---|
| L0 | structural | 794 records validated |
| L1 | count | observed 794, expected 794 |
| L1 | strain_uniqueness | 794 unique (strain, condition) records over 794 strains |
| L1 | genotype_strain_consistency | all 794 genotypes name one isolate each |
| L2 | value_fidelity | 1214201 values checked |
| L2 | single_acquisition | n_replicates values [1] (the release: acquired once) |
| L3 | shared_reference | one shared reference over 1576 proteins, all finite and > 0 |
| L3 | reference_covers_every_key | 0 measured cells have no euploid reference value |
| L3 | measurement_type_consistent | single measurement_type: 'swath_ms_maxlfq_precursor_batch_corrected_quantity' |
| L4 | gene_containment_sgd | 1.0000 of 1576 measured proteins are S288C systematic names (>= 0.99); outside: [] |
| L4 | caudal_isolate_overlap | 759 isolates also in the built Caudal 2024 store (expected 759) |
| L4 | caudal_genotype_agreement | 4 isolates compared with the Caudal 2024 store: 4 identical genotypes |
| L1 | isolate_genotype_identity | 794 isolates, one genotype each, every strain-carrying perturbation names its isolate |
| L2 | isolate_genotype_round_trip | 794 stored genotypes equal their re-derivation from the pinned Peter 2018 releases |
| L4 | isolate_descriptor_matches_table_s1 | 794 descriptors and 85 HO deletions equal Supplementary Table 1 re-read from si3.xls |
| L4 | loss_of_function_calls_match_release | stored loss-of-function calls are the release's 1s, row for row by the matrix's own key, over 794 isolates: {'nonsense_or_sift': 185451, 'homozygous_frameshift': 114589} |

### `protein_turnover_muenzner2024` (kept as `processed.superseded.20261012-114548`)

Genotype ledger: 55 isolates, 278,223 sequence variants, 4,940 natural presences, 7,366 natural absences, 4 HO deletions; loss-of-function calls: nonsense_or_sift 12,695 on 1,440 genes, homozygous_frameshift 7,830 on 849 gene keys (474 on rows released without a code).

All 14 rows pass (`preprocess.superseded.20261012-114548/verification_report.json`):

| level | rule | result |
|---|---|---|
| L0 | structural | 55 records validated |
| L1 | count | observed 55, expected 55 |
| L1 | strain_uniqueness | 55 unique (strain, condition) records over 55 strains |
| L1 | genotype_strain_consistency | all 55 genotypes name one isolate each |
| L2 | value_fidelity | 58061 values checked |
| L3 | half_life_is_ln2_over_rate | max relative error 5.68e-15 over 58061 (rate, HL) pairs |
| L3 | shared_reference | one shared reference over 1524 proteins, all finite and > 0 |
| L3 | uncovered_cells_match_ledger | 939 measured cells without a euploid reference value (ledger: 939) |
| L3 | measurement_type_consistent | single measurement_type: 'dynamic_silac_lys8_kdp_per_hour_no_dilution_correction' |
| L4 | gene_containment_sgd | 0.9989 of 1836 measured proteins are S288C systematic names (>= 0.99); outside: ['R0040C', 'YFL057C'] |
| L1 | isolate_genotype_identity | 55 isolates, one genotype each, every strain-carrying perturbation names its isolate |
| L2 | isolate_genotype_round_trip | 55 stored genotypes equal their re-derivation from the pinned Peter 2018 releases |
| L4 | isolate_descriptor_matches_table_s1 | 55 descriptors and 4 HO deletions equal Supplementary Table 1 re-read from si3.xls |
| L4 | loss_of_function_calls_match_release | stored loss-of-function calls are the release's 1s, row for row by the matrix's own key, over 55 isolates: {'nonsense_or_sift': 12695, 'homozygous_frameshift': 7830} |

`caudal_genotype_agreement` compares four sampled isolates with the Caudal 2024 store as #1058 rebuilt it (10:30), so both stores carry the shared genotype.
