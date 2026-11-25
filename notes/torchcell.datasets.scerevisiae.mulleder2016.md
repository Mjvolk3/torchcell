---
id: wls9f96e2x0wkq7ojlvm5ch
title: Mulleder2016
desc: ''
updated: 1783309845159
created: 1783309845159
---

## 2026.07.05 - Recon + Build Plan (WS9 amino-acid metabolome)

Genome-wide amino-acid metabolome of the yeast deletion collection. Rounds out the
metabolomics/omics-for-pretraining set alongside Kemmeren (expression, already landed).
Maps to `MetabolitePhenotype` (WS4). Citation key `mullederFunctionalMetabolomicsDescribes2016`.

### Provenance (verified this session)

- **Paper:** Mulleder et al. 2016, *Cell* 165:1282, "Functional Metabolomics Describes
  the Yeast Biosynthetic Regulome." DOI `10.1016/j.cell.2016.09.007`, PMID **27693354**,
  open access PMC5055083.
- **Raw data (scriptable, hash-pinned):** Mendeley Data DOI `10.17632/bnzdhd6ck8.1`
  ("The Saccharomyces cerevisiae amino acid metabolome", CC BY 4.0). File
  `Table_S3_Complete_Dataset.xls`, **8,502,272 bytes**, sha256
  `a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e`. Direct download:
  `https://data.mendeley.com/public-files/datasets/bnzdhd6ck8/files/621f3646-9e51-488a-b6b8-f6427b40fc87/file_downloaded`
  (listed via `https://data.mendeley.com/public-api/datasets/bnzdhd6ck8/files?folder_id=root&version=1`).
  `retrieval_method = direct_url`. Downloaded + sha256-verified 2026-07-05.
- No OCR needed for the DATA (structured .xls). OCR the paper PDF only for Methods
  provenance (replicate structure -- see open question).

### Table_S3 structure (inspected)

8 sheets. The loader consumes **`intracellular_concentration_mM`**: one row per ORF
(4678 strains) x **19 amino acids** in mM (batch-normalised, adjusted for dilution,
extraction volume, cell number + volume). The 19 = 20 standard AAs minus **cysteine**
(excluded, oxidation) -- note the paper text says "18", the DATA has 19; source the
count from the data. Other sheets: `data_raw` (raw uM, incl. 237 QC-sample rows +
slow-growers, 4831 unique ORF), `robust_summary_statistics` (POPULATION mean/SD/RSD per
AA, Minimum Covariance Determinant), `Z-score`, `p_value_Z-test`, `Mahalanobis_distance`,
`p_value_X2-test`.

### OPEN PROVENANCE QUESTION -- n_replicates / SE (do NOT guess)

The per-strain `intracellular_concentration_mM` sheet has **one value per (strain, AA)**;
`data_raw` shows **4637/4678 strains with a single measurement** (170 with 2, 20 with 3,
4 with 4). So the genome-wide screen is effectively **n=1 per strain**; the paper's
"mean +/- SD, n=3" refers to targeted VALIDATION, not the screen. **No per-strain SE
column is released** (only population-level robust SD per AA). Per CLAUDE.md "value not a
released per-record column": must READ THE METHODS (OCR paper) to confirm the screen
replicate structure before setting `n_replicates`; likely `n_replicates={aa:1}` and
`metabolite_level_se=None` (honest), with the population robust SD recorded as context
only, NOT as per-record error. RESOLVE via OCR before finalizing.

### MetabolitePhenotype mapping (plan)

- `metabolite_level = {amino_acid -> mM}` (19 AAs) from intracellular_concentration_mM.
- `measurement_type = "intracellular_concentration_mM"`.
- `n_replicates` / `metabolite_level_se`: per open question above.
- `target_metabolite_ids`: **populatable** -- amino acids ARE native Yeast9 metabolites
  (map each AA -> `s_NNNN`). First dataset where CBM linkage is real, not deferred.
- Genotype: single-gene KO per strain (ORF = systematic name; ~1029 unannotated gene
  names, key on ORF). **Background = PROTOTROPHIC deletion collection** (auxotrophy
  restored episomally, Mulleder et al. 2012) -- NOT standard BY4741; reference/genotype
  must reflect prototrophy restoration. Grown exponentially in MINIMAL medium.
- `Publication(pubmed_id="27693354", doi="10.1016/j.cell.2016.09.007")`.

### Build checklist

1. [ ] Deposit raw `Table_S3_Complete_Dataset.xls` + provenance record to the raw-data
   mirror; mirror the paper PDF (Zotero + torchcell-library) + MinerU OCR -> paper.md.
2. [ ] Resolve n_replicates/SE from Methods OCR (open question).
3. [ ] Decide amino-acid -> Yeast9 `s_NNNN` map (target_metabolite_ids).
4. [ ] Model prototrophic background on genotype/reference.
5. [ ] Loader `torchcell/datasets/scerevisiae/mulleder2016.py` (mirror cachera2023.py);
   register in datasets `__init__`.
6. [ ] Add to `torchcell.verification` metabolite runner; build LMDB; verify L0-L4.
7. [ ] Land via rebase + ff (own worktree `feat/ws9-mulleder-metabolome`).

## 2026.07.06 - Built + Verified (L0-L4 PASS)

Loader `torchcell/datasets/scerevisiae/mulleder2016.py`
(`AminoAcidMulleder2016Dataset`) built and verified. Decisions resolved:

- **n_replicates = 1, metabolite_level_se = None** -- sourced from Methods: the
  genome-wide screen is a SINGLE culture per strain ("Yeast" section); the 237x QC
  sample is for analytical performance only ("Quantification of Amino Acids in High
  Throughput"), NOT per-strain averaging. The paper's "n=3" is targeted validation.
- **Reference = population robust mean per amino acid** (MCD estimate,
  `robust_summary_statistics` sheet) -- a WT-equivalent baseline, since the released
  per-strain data covers deletions only. This is ABSOLUTE mM, not centered, so the
  metabolite verifier gained `reference_centered=False` -> a `reference_finite` L3
  check (finite + key-matched) instead of the Cachera `reference_zero` (==0) check.
- **19 amino acids** (20 proteinogenic minus cysteine); all 4678 strains have all 19,
  no NaN, all ORFs valid systematic ids (no genome needed).
- **target_metabolite_ids = None (DEFERRED)** -- amino acids ARE native Yeast9
  metabolites so this is populatable (first real CBM linkage); deferred to a follow-up
  sourcing the `s_NNNN` ids from YeastGEM (not guessed).

Verification (`run_metabolite`): L0 4678 validated; L1 count 4678==4678 + 4678 unique
ORFs; L2 88882 values finite; L3 reference_finite + measurement_type_consistent; L4
gene_containment 0.950 of deletion genes in Ohya. ALL PASS. Cachera still PASS
(centered reference_zero path unaffected). Registered in `scerevisiae/__init__` +
`verification.runners.METABOLITE_DATASETS`.

REVIEW FLAGS: (1) reference = population mean (WT-proxy, sourced) vs a measured WT row
-- confirm modeling choice; (2) prototrophic background (pHLUM-like episomal marker
restoration) not yet modeled as a GeneAddition -- ReferenceGenome strain="BY4741";
(3) target_metabolite_ids CBM mapping deferred.

- [x] loader + build + L0-L4 verify + registration
- [ ] target_metabolite_ids (amino acid -> Yeast9 s_NNNN) follow-up
- [ ] prototrophy-restoring marker as GeneAddition (ties to plasmid-sequence store)

## 2026.09.15 - Cross-platform partner: Cooper 2010

Cooper 2010 ([[torchcell.datasets.scerevisiae.cooper2010]]) measures amino-acid pools of the
same deletion collection by CE-LIF (4,051 ORFs shared with this dataset on the deposit) as a
linear ratio to the plate mean, against this dataset's absolute mM. The join is exact on seven
keys (`arginine`, `alanine`, `threonine`, `serine`, `glycine`, `glutamate`, `aspartate`); Cooper's
composite peaks (`leucine+isoleucine+citrulline`, `glutamine+valine`, `methionine+proline`,
`asparagine+tyrosine`) need an aggregation rule on this side (sum of member mM is the obvious
candidate, the author decides), its two lysine peaks need a peak rule, and its
`lysine_related_peak1`, `gshbiotin`, `n_acetylornithine`, `ornithine` have no partner here.
Compare by rank or by log ratio to each dataset's own center, never by value.

## 2026.09.23 - SM media components (issue 143)

- Loader now emits `SM_AGAR` (6.7 g/L YNB without amino acids, 2% glucose, 2% agar, sourced
  from this paper's STAR Methods with a verbatim quote + `paper.md` sha256) instead of the
  bare `Media(name="SM", state="solid", is_synthetic=True)` stub. Formulation table, the
  ammonium sulfate gap, and the open question about whether this record's state should be
  liquid (the amino acids are extracted from the liquid SM subculture the agar spots
  inoculate) are in [[torchcell.datamodels.media-components]]. LMDB rebuilt + L0-L4 re-run.

## 2026.09.29 - Independent re-verification

A read-only Fable 5.1 agent graded eight recorded claims against the mirror, the Europe PMC SI, Mendeley, MetaboLights MTBLS434 and the dev LMDB: CONFIRMED 4, REFUTED 1, PARTLY 3, UNVERIFIABLE 0. Consolidated in [[datasets.showcase-verification.2026.09.29]]; raw report `notes/assets/verification/2026.09.29/mulleder2016.md`.

- Refuted: `download()` verifying the sha256 of a present file; it returns before hashing. #490. Graded partly, with its first half refuted: "the paper says 18 amino acids" (2026.07.05 section); no sentence gives 18, and Table S5 lists 19.
- The Cell SI `mmc3.xls` is byte-identical to the Mendeley pin (sha256 `a7fcb4bc...`), and MetaboLights MTBLS434 holds the raw-level uM data (equal to `data_raw` to 3 decimals) and 6,475 mzML files; no raw-mirror provenance record exists for Table S3. #487.
- "4637/4678 strains with a single measurement" mixes denominators: among the 4,678 released strains 4,487 have one `data_raw` row and 191 have two to four (167 / 20 / 4). Their released mM fits the mean of the normalized rows (diagnostic, not a published statement). `n_replicates = 1` is understated for those 191. #488 (before the next build).
- The reference `n_replicates = {aa: 1}` misdescribes the MCD robust mean, a statistic over 4,678 strains. #489 (before the next build).
- The complemented WT rows (YOR202W batch 04, YLR303W batch 02) exist in `data_raw` but not in the mM sheet, which bears on review flag 1.
- The citation is Cell 167(2):553-565.e12 (PMC5055083), not "Cell 165:1282".

## 2026.09.29 - Showcase page (group 3)

Shown on `docs/source/datasets/scerevisiae/amino-acid-betaxanthin.md` ([[experiments.034-showcase-datasets.scripts.amino_acid_betaxanthin]]). The dev store's content sha256 differs from release `2026.09.21-ab6d8c5d`: every record differs only in `environment.media` (dev `SM_AGAR` from the 2026.09.23 rebuild, served `SM` stub); no phenotype field differs. The page's caveats cite the 2026.09.29 re-verification (#487, #488, #489, claim M3) and the open medium-state question (#143).

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Ledger counts dropped rows, concentration cells refused by name (issue #528)

- Before: the ledger reported the number of ORFs that collided ("1 ORF collisions deduped" for three dropped rows); a blank concentration cell was served as NaN and a text cell stopped the build with Python's `float()` message.
- Now: the ledger reads "<n> repeated-ORF rows dropped (<m> ORFs kept at their first row)"; `_concentration` raises `InvalidConcentrationError` ("Mulleder Table S3: <ORF> <amino acid> concentration <cell> is not a finite number") for a blank or non-numeric cell.
- Record-neutral on the pinned Table S3: 4,678 concentration rows, 0 non-systematic ORFs, 0 repeated ORFs, 0 blank and 0 text cells (all 19 columns float64).
- Left open: `n_replicates` stays 1 although `data_raw` has two to four rows for 191 of the 4,678 released ORFs (167 with 2, 20 with 3, 4 with 4; issue #488), the reference count (#489) and `SM_AGAR` (#143).
- Behavior note: a numeric-looking TEXT cell such as `"1.5"` is now refused, where `float()` used to accept it; the pinned sheet has 0 text cells.
- Tests: `test_ledger_counts_dropped_rows_and_their_orfs`, `test_blank_concentration_refuses_the_build_by_name`, `test_text_concentration_refuses_the_build_by_name`.

## 2026.10.01 - Correction to the replicate count above

The first count of multi-row ORFs (194) was taken over all 4,831 ORFs in `data_raw`. Three of them (YOR221C, YOL086C, YOL067C, two rows each) are not in the released `intracellular_concentration_mM` sheet, so the affected stored records are 191 (167 with 2 rows, 20 with 3, 4 with 4), matching issue #488. Re-measured with `verify_review.py` in the fix scratchpad.

## 2026.10.02 - Per-strain n_replicates from data_raw, reference n = strains (issues #488, #489)

- Wrong before: every record carried `n_replicates = 1` (understated for the 191 released strains with two to four `data_raw` rows), and the reference, the MCD robust mean over the collection, also carried 1, which describes one measurement.
- #488 now: per record and amino acid, `n_replicates` is the number of `data_raw` rows for that ORF with a value in that column; QC rows (237, no ORF) count for nobody. A released ORF with no raw row, or an amino acid blank in all its rows, raises `MissingRawMeasurementError` (0 of each on the pinned workbook). `preprocess/data.csv` gains `n_raw_rows`. The loader docstring now states that the released value of a multi-row strain being the mean of its normalized rows is a diagnostic inference from the 2026.09.29 report (`mulleder_multi_average_test.py`), not a published statement; the only published statement is the batch 12 repeat of the 479 lowest-concentration strains (paper.md line 341, now `SOURCED_VALUES["repeat_screen"]`).
- #489 now: the reference `n_replicates` is the number of released strains counted from the concentration sheet at build time (4,678) on every key, with the line 387 MCD quote as `SOURCED_VALUES["reference_estimator"]` and the line 395 count as `SOURCED_VALUES["n_profiled_strains"]`. MCD down-weights outlying strains, so this is the size of the population a robust estimate summarizes, not 4,678 replicates of one sample. The schema field says only "metabolite_id -> number of independent replicates" and does not forbid a population count, so no gap was recorded. This is a decision the user may revisit (alternatives: a population-estimate representation in the schema).
- Measured (`experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_replicates.py`, scratch build vs the dev store, results in `experiments/036-dataset-fixes-before-kg-build/results/mulleder2016_replicates.json` and `mulleder2016_replicates_multirow.csv`): 4,678 records in both; strains by raw row count 1: 4,487, 2: 167, 3: 20, 4: 4; experiment `n_replicates` changed on 191 records; 7 ORF x amino-acid counts on 5 strains sit below the strain's row count because a raw cell is blank (YGL252C, YBR196C-A, YCR037C, YJR060W, YNL096C); reference `n_replicates` 1 to 4,678 on all 4,678 records; 0 records differ in any other field. Diagnostic: scikit-learn `MinCovDet(random_state=0)` on the 4,678 x 19 released mM matrix gives a location within 1.2% (max) and 0.13% (median) of the released `mean (mM)`, against 7.9% / 1.6% for the plain mean, which supports but does not prove that the reference was fit on these 4,678 strains.
- Open: the user may prefer a different representation of a population reference (#489). Hypothesis (untested against a published statement): the multi-row released values are means. #143 (medium state) and #487 (raw-mirror record) are unchanged.
- Tests: `test_n_replicates_counts_the_data_raw_rows_per_amino_acid_issue_488`, `test_a_released_orf_without_a_data_raw_row_is_refused`, `test_an_amino_acid_blank_in_every_raw_row_is_refused`, `test_reference_n_is_the_number_of_strains_the_robust_mean_summarizes_issue_489`, `test_every_sourced_value_quote_is_verbatim_in_the_mirrored_paper` (skips without `DATA_ROOT`; passed with it set).

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: a row whose ORF was not ORF-shaped was skipped, and a repeated ORF kept its first row (both 0 on the pinned workbook).

Now: a row with no ORF form is counted in the build log as deferred to the released-gene-name leaf (never dropped for good; 0 of 4,678), and a repeated ORF raises rather than keeping the first (0). 4,678 records, unchanged.

## 2026.10.12 - The deferred name is a record on the released-gene-name leaf (#1045, PR #1085)

A released identifier with no ORF form goes on `ReleasedGeneNamePerturbation` (0 of 4,678 rows). The two concentration-sheet rows that name no deletion strain, the blank ORF cell and the `WT` background row, are listed by name in the log rather than counted as deferred. The count stays 4,678.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**amino_acid_mulleder2016**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/amino_acid_mulleder2016/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 4678 records validated |
| L1 | `count` | gate | ok | observed 4678, expected 4678 |
| L1 | `genotype_uniqueness` | gate | ok | 4678 unique strains (deletion sets), one record each |
| L2 | `value_fidelity` | gate | ok | 88882 values checked |
| L2 | `se_nonnegative` | gate | ok | 0 values checked |
| L3 | `reference_finite` | gate | ok | reference level finite + key-subset for all 88882 values |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'intracellular_concentration_mM' |
| L4 | `gene_containment_scmd_ohya2005` | report | ok | 0.950 of scmd_ohya2005's 4678 deletion genes are in Ohya (>= 0.9) |

## 2026.11.24 - One record per data_raw culture (#1178, #1176, #528, #1172)

Policy: one stored record per biological replicate the source releases (tracking issue #1178, owner decisions 2026-11-17 and 2026-11-24).

### Before and after

- Before: 4,678 records, one per row of `intracellular_concentration_mM`. The `data_raw` sheet was read only to count `n_replicates` (2026.10.02 section); its 5,053 strain rows were never stored.
- Now: 9,731 records. The 4,678 mM records are unchanged and come first, in the same order. Then one record per `data_raw` strain row, 5,053, in sheet order: `measurement_type = "cell_extract_concentration_uM"` (`RAW_MEASUREMENT_TYPE`), `metabolite_level` = that row's uM values verbatim with a blank cell left out of the dict, `n_replicates = 1` per amino acid, `metabolite_level_se = None`, `replicate_id` = the row's `identifier` (the well's plate_row_column, `41_G_10`; batch 12 prints `01_D12`), unique over the sheet.
- The 237 `QC_*` rows are not records: they are repeat injections of one pooled extract of 351 strains, which has no statable genotype.
- No schema change: `MetabolitePhenotype.measurement_type` is a free string and `replicate_id` exists since PR #1181.

### What a data_raw row is (sources, all in `paper.md` sha256 `20412bec5b930d1fa43d326d8f9baca267130ddef5fdaa804f19e1209f925e6e`, and `Table_S3_Complete_Dataset.xls` sha256 `a7fcb4bc8aa5e394e7f6e2b99e327eaa88fa04111ab5602fc7cb3445f653802e`)

- Overview sheet, row `data_raw`: "Raw concentration data of the cell extract in [µM]; includes QC sample  and slow growing strains. " (verbatim, with the double space).
- Overview sheet, row `intracellular_concentration_mM`: "Intracellular concentration in [mM]. Batch-normalised data, adjusted for dilution and corrected for extraction volume, cell number and volume." So uM and mM are two measurements, never compared directly.
- paper.md L333 (Yeast): each strain was "finally cultured in 96-deep well plates with ... SM medium"; L341: "Strains from the same batch were cultivated and their metabolites extracted on the same day and analyzed together by LC-MS/MS." One row is one well, one culture, one extract (`SOURCED_VALUES["batch_cultivation"]`).
- L341: "In a second screening round, the 479 deletion strains that showed lowest metabolite concentrations were arranged on 6 additional plates and the analysis was repeated (batch 12)."
- L341: "The QC was prepared by joining the extracts of the first batch including metabolites of 351 deletion strains." and "a quality control sample (QC) was assessed every 24 samples and in total 237 times" (`SOURCED_VALUES["qc_sample"]`).
- Measured on the pinned workbook: all 410 rows of the 191 multi-row released ORFs sit at distinct identifiers (wells), and 161 of the 191 span two or more batch labels (160 if `03-2` is read as `03`). Only 14 of those 410 rows are batch 12 (7 ORFs), so most repeated ORFs are repeats across batches 01 to 11, not the batch 12 re-screen. Batch 12 holds 468 rows of 461 ORFs, against the paper's 479 strains.
- Withdrawn samples: none in `data_raw`. The three cross-contaminated strains of L379 ("aco1D, lip5D and mtc1D", YLR304C, YOR196C, YJL123C) have no row, so there is nothing to drop. The other strains the authors left out of the analysis (slow growers, outdated annotations, the two complemented controls) are released in `data_raw` and stored.
- How two to four rows become one released mM value is still unpublished (#1172); the mM rows stay as released, with `replicate_id = None`.

### Reference of the raw records

The paper names the complemented deletion strains as the isogenic wild type: L487 (Figure S5B) "contain a complemented his3 deletion strain representing the isogenic reference wt strain"; L459 (Figure S1F) shows the WT profile "for the complemented deletion strains his3D::kanMX4 and met17D::kanMX4". `data_raw` has one row each: `20_D_10` YOR202W (HIS3, batch 04) and `23_A_8` YLR303W (MET17, batch 02). Each raw record's reference is their mean, `SD / sqrt(n)` and n = 2 per amino acid, `replicate_id = None`, restricted to the amino acids the record measured (the verifier's key-subset rule). The two controls are records themselves under their released deletions. Open for review: the two rows come from two batches of un-normalized uM, so this reference carries batch effects; no released uM reference exists.

### Counts (`experiments/036-dataset-fixes-before-kg-build/scripts/mulleder2016_per_replicate.py`, results `experiments/036-dataset-fixes-before-kg-build/results/mulleder2016_per_replicate.json`)

Scratch build: `/scratch/projects/torchcell-scratch/tmp/scratch/per-replicate-mulleder2016/amino_acid_mulleder2016`.

| quantity | value |
|---|---|
| records, mM / raw uM / total | 4,678 / 5,053 / 9,731 |
| `data_raw` rows / QC rows not stored | 5,290 / 237 |
| raw records of released ORFs / of 153 ORFs absent from the mM sheet | 4,897 / 156 |
| released ORFs with 1 / 2 / 3 / 4 cultures | 4,487 / 167 / 20 / 4 |
| raw records by position within an ORF, 1st / 2nd / 3rd / 4th | 4,831 / 194 / 24 / 4 |
| blank `data_raw` cells left out | 265 |
| max abs difference, raw record vs `data_raw` cell | 0.0 |
| mM records vs the previous dev store, key paths that differ | only `replicate_id` (new key, None) on experiment and reference, 4,678 each |
| max abs difference, mM value vs the previous dev store | 0.0 |

Mean reproduction: grouping the records by genotype and environment on the mM measurement type returns the previous store's values exactly (max abs difference 0.0), because those records are the released rows themselves. The raw uM records are a different measurement before batch normalization and the dilution corrections, so their per-strain mean does not, and is not meant to, reproduce the mM value.

### L0-L4 on the scratch build

**amino_acid_mulleder2016**: verdict PASS (`preprocess/verification_report.json` in the scratch root). `runners.py`: `expected_count` 9,731 and `protocol_measurement_types` = {mM, uM}, so L1 keys on (strain, measurement type, replicate) and L3 requires each record's reference on its own type.

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 9731 records validated |
| L1 | `count` | gate | ok | observed 9731, expected 9731 |
| L1 | `genotype_uniqueness` | gate | ok | 9731 unique (strain, protocol) pairs per released replicate, one record each |
| L2 | `value_fidelity` | gate | ok | 184624 values checked |
| L2 | `se_nonnegative` | gate | ok | 0 values checked |
| L3 | `reference_finite` | gate | ok | reference level finite + key-subset for all 184624 values |
| L3 | `measurement_type_consistent` | gate | ok | 2 declared protocol measurement_types, each record's reference on its own protocol |
| L4 | `gene_containment_scmd_ohya2005` | report | ok | 0.949 of scmd_ohya2005's 4831 deletion genes are in Ohya (>= 0.9) |

### Also changed

- `data_raw` is read with `batch` as text, so `01` stays `01` (it only fed `n_replicates` before, which did not use it).
- New refusals, `InvalidRawRowError`: a repeated identifier, an ORF-less row that is not `QC_*`, a text cell, a row blank in all 19 amino acids, a missing complemented control (0 of each on the pinned workbook).
- New side file `preprocess/data_raw.csv`, one row per raw record.
- `build_time_projection.py`: 4,678 to 9,731.

### Not done

- The batch label of a culture is not stored: `MetabolitePhenotype` has no batch field, and the raw uM values are batch-dependent. It needs a schema field (owner decision).
- The raw mirror `$DATA_ROOT/torchcell-raw/mullederFunctionalMetabolomicsDescribes2016/` does not exist; the workbook lives only in the dev store's `raw/` with its sha256 pin (#487, unchanged).
