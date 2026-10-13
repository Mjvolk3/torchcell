---
id: zyn8xo41fsy0q63qkrvackr
title: Messner2023
desc: ''
updated: 1783832523640
created: 1783832523640
---

## 2026.07.12 - Messner 2023 genome-wide KO proteome build

`ProteomeMessner2023Dataset` (`torchcell/datasets/scerevisiae/messner2023.py`).
Messner et al. 2023, *Cell* 186:2018, doi:10.1016/j.cell.2023.03.026, PMID 37080200.
Proteomes of the whole *S. cerevisiae* (S288C) haploid MATa gene-deletion collection
(restored prototrophy) by microflow-SWATH-MS (DIA). Maps each KO strain to
`ProteinAbundancePhenotype` (WS9). **L0-L4 verified, 4,699 records.**

### Data source -- "mirror once + hash-pin" (NOT a live Mendeley dependency)

The curated protein matrix is deposited **only on Mendeley** (doi:10.17632/w8jtmnszd9.1).
We do NOT use Mendeley as a live source ([[no-mendeley-data-source]] preference). Checked
all alternatives:

- **PRIDE/ProteomeXchange PXD036062** -> actually **MassIVE MSV000090136**; only
  processed object is a **74.8 GB** raw DIA-NN precursor report (not the protein matrix).
- **Cell SI (Tables S1-S7)** -> none is the abundance matrix; SI defers to Mendeley.
- **y5k web app** (<https://y5k.bio.ed.ac.uk/>) -> R Shiny, no scriptable download.

So the matrix is Mendeley-only. User approved the **mirror-once + hash-pin** path: fetch
from Mendeley ONCE into the library mirror
(`$DATA_ROOT/torchcell-library/messnerProteomicLandscapeGenomewide2023/data/`), pin by
sha256, and have the loader read from the mirror (`download()` copies + verifies sha256;
no live Mendeley call). Files + provenance recorded in the mirror `manifest.json`:

- `yeast5k_noimpute_wide.csv` -- sha256 `69a9df05...aa1df9`, 167,754,298 B. Proteins
  (rows, UniProt `Protein.Group`) x samples (cols). Batch-corrected MaxLFQ, **no
  imputation** (measured values only; NaN dropped per strain).
- `yeast5k_metadata.csv` -- sha256 `48864282...878b`, 377,047 B. Per-sample: `Filename`
  (matrix col id), `sampletype` (`ko`|`HIS3`|`qc`), `ORF` (deleted gene), plate nr.

### Design decisions (all sourced, not guessed)

- **Value = linear MaxLFQ** (`measurement_type = "swath_ms_maxlfq_batch_corrected_quantity"`).
  Values ~340-430 confirm linear; log2 is applied only downstream in the paper's
  differential analysis.
- **noimpute over impute** -- store only actually-measured quantities; never persist an
  imputed value as if measured (honest-to-source). First strain stored 1,830/1,850.
- **Single-replicate KOs.** "Strains were not measured in replicates" (STAR Methods) ->
  per-strain `n_replicates = 1`, `protein_abundance_se = None`. (L2 `se_nonnegative`
  checks 0 values -- the correct signature.)
- **388-replicate WT reference.** Control = his3D::kanMX complemented by HIS3
  (`sampletype == HIS3`, ORF YOR202W), 388 reps across 57 batches -> per-protein
  mean + SE + n. Reference is **restricted per record to the strain's measured proteins**
  (as in the Zelezniak metabolite loader), so reference keys == experiment keys (L3
  `reference_finite` passes) and every reference value is a real WT measurement.
- **UniProt -> systematic ORF, 100%.** Matrix ids are UniProt accessions; mapped to ORFs
  via the SGD S288C GFF `protein_id=UniProtKB:` cross-refs (`build_uniprot_to_orf_map`),
  keeping the proteome joinable with the ORF-keyed Zelezniak proteome. 1,850/1,850 map
  (incl. mito `P00410` -> `Q0250`/COX2 via the `Q\d{4}` pattern); an unmapped id RAISES.
- **Duplicate strains kept per-instance.** One experiment per KO SAMPLE (4,699), not per
  ORF. The verifier's L1 `orf_uniqueness` was parameterized (`allow_duplicate_orfs=True`)
  for this dataset.
- **Background** = BY4741 MATa deletion collection, restored prototrophy (Mulleder); the
  prototrophy marker is not yet modeled as a GeneAddition (as in the Mulleder loader).
  Environment = synthetic minimal (SM) liquid, 30 C.

### Two data quirks handled (more rigorous than the source)

1. **Inconsistent ORF casing.** Metadata has `YML009c` and `YAL043C-a`; a case-sensitive
   ORF regex would have silently dropped 2 real strains. Fixed by uppercasing the deletion
   ORF (systematic names are uppercase) -> full 4,699. Gene-name parse from `Filename` is
   case-insensitive so `MRPL39` is still recovered.
2. **146 duplicated ORFs vs the paper's 145.** The raw metadata splits gene MRPL39 across
   `YML009C` and `YML009c`; case-sensitive grouping counts them as two singletons (the
   paper's 145). Normalizing case recognizes MRPL39 has **2 strains** -> 146 duplicated
   ORFs. Total strains unchanged (4,699); we are simply more rigorous than the source.

### L0-L4 verification (`torchcell/verification/runners.py` `proteome_messner2023`)

All pass: L0 structural (4,699), L1 count (4,699), L1 orf_uniqueness (4,549 ORFs, 146
multi-strain, expected), L2 value_fidelity (8,466,210 finite values), L2 se_nonnegative
(0 values), L3 reference_finite (key-matched, all 8,466,210), L3 measurement_type_consistent.

### Follow-ups

- Growth rates (`yeast5k_growthrates_byORF.csv`) + differential tables (`yeast5k_stat_DE*`)
  are separate phenotypes for later.
- Model the prototrophy-restoring marker as a `GeneAddition` (shared with Mulleder/Zelezniak).
- Consider an `impute` variant if a dense matrix is needed downstream.

## 2026.09.23 - SM media components (issue 143)

- Loader now emits the shared `SM` constant (6.7 g/L YNB without amino acids + 2% glucose,
  liquid). This paper both defers its growth protocol to Mulleder 2016 (its ref 15, which is
  mirrored) and restates the identical recipe, so one sourced object carries both papers.
  Table and gaps in [[torchcell.datamodels.media-components]]. LMDB rebuilt + L0-L4 re-run.

## 2026.09.29 - Independent re-verification

A read-only Fable 5.1 agent graded twenty recorded claims against the mirror, Mendeley, ProteomeXchange/PRIDE/MassIVE, Europe PMC and the dev LMDB: CONFIRMED 13, REFUTED 1, PARTLY 5, UNVERIFIABLE 1. Consolidated in [[datasets.showcase-verification.2026.09.29]]; raw report `notes/assets/verification/2026.09.29/messner2023.md`.

- Refuted: the test note's "only bites on a malformed filename". 156 records carry a numeric `perturbed_gene_name` from real filenames (SLA1 is served as "2824") and 1,182 a lowercase ORF; gene names should come from the SGD GFF by systematic name. #485 (before the next build).
- "Values ~340-430 confirm linear" (2026.07.12 section) is wrong: the matrix runs 0.0313 to 393,221, median 235.1. Linear scale is confirmed instead by reproducing the paper's median CVs (WT 11.0% vs 11.3%, QC 7.9% vs 8.1%, KO 15.5% vs 16.2%). Batch correction is applied to precursors before MaxLFQ. #486.
- The paper's 8,693,150 protein quantities is 4,699 x 1,850, the imputed grid; the store holds 8,466,210 measured values (97.4%).
- The 8 h post-dilution culture duration is stated in the Methods but `duration_hours` is None. #486.
- PXD036062 is not in PRIDE; MassIVE MSV000090136 holds raw `.wiff`, the 74,825,469,404 B DIA-NN report, the spectral library and a plate-layout file, no protein matrix. The Cell SI was not inspected (no `si/` in the mirror) and the y5k app has a "Download" tab whose content was not retrievable, so the mirror-once argument rests on the Key Resources Table and MassIVE.
- BY4741 and kanMX for the KO strains come from the cited collections (refs 5 and 94), not from a Messner sentence about the KOs.
- The paper table's "vector (1830)" is the first record's protein count; per-record counts run 1,441 to 1,849 and the union is 1,850. The served protein count (1,667 vs 1,850 in two memories) is unverified.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #528; the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes (#528, present file skipped); PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `MATRIX_SHA256`, `METADATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - KO protein without a WT measurement refused by name (issue #528)

- Before: a protein a KO sample measured but no HIS3 sample did was absent from the reference and failed `create_experiment` with a bare `KeyError('<ORF>')`.
- Now: `process` raises `MissingWildTypeReferenceError` naming the KO sample, its ORF, the count and the first proteins, before any LMDB is written.
- Record-neutral: 0 of 1,850 proteins in the pinned matrix are measured in a KO sample (4,699 columns) and in no WT sample (388 columns).
- Test: `test_a_ko_protein_no_wt_sample_measured_is_refused_by_name`.

## 2026.10.02 - Gene names from the SGD GFF, not the filename (issue #485)

- Wrong before: `perturbed_gene_name` was the Filename token after the ORF (`_gene_from_filename`). On real filenames that token is sometimes a plate-position number (`10_9_hpr57_ko_YBL007C_2824_0.49` served SLA1 as "2824"), a lowercase ORF (`YBR174c`), "Unknown", or an outdated name.
- Now: `build_orf_to_gene_name_map` reads the same SGD GFF the UniProt map uses (R64-4-1 on GilaHyper) and maps every feature whose `ID` is a nuclear ORF to its percent-decoded `gene=` (so `MF%28ALPHA%291` is `MF(ALPHA)1`), else to the ORF. `perturbed_gene_name` is that name, or the uppercase ORF when the ORF is no feature `ID` in the GFF (logged: "64 KO strains (62 ORFs) whose ORF is no SGD GFF feature ID"). `systematic_gene_name` (metadata ORF, uppercased) is unchanged and stays the join key. `_gene_from_filename` is deleted.
- Measured (`experiments/036-dataset-fixes-before-kg-build/scripts/messner2023_gene_names.py`, scratch build vs the dev store, results in `experiments/036-dataset-fixes-before-kg-build/results/messner2023_gene_names.json` and `messner2023_gene_names_changed.csv`): 4,699 records in both; `perturbed_gene_name` changed on 1,524 records: 156 numeric tokens (154 ORFs), 1,182 lowercase ORFs (346 of them now carry an SGD standard name, 836 the uppercase ORF), and 186 other (109 case only, such as `RPL37a` to `RPL37A`; 29 "Unknown" to the ORF; the rest renamed genes such as `AIM5` to `MIC12`; 22 of the 186 on ORFs that are no GFF feature ID). 0 records differ in any other field. After the fix 0 names are all digits and 0 contain a lowercase letter.
- Open: the 62 KO ORFs that are no feature `ID` in R64-4-1 are retired or merged ORFs (several appear as an `Alias` of a current feature, e.g. YAR044W under YAR042W/SWH1). Their `systematic_gene_name` is the outdated ORF and their name is that ORF; resolving them to current loci is a separate decision. The `tests.torchcell.datasets.scerevisiae.test_messner2023` note still quotes the refuted "only bites on a malformed filename" line in its older sections.
- Tests: `test_numeric_filename_token_is_not_the_gene_name_issue_485` (SLA1), `test_lowercase_orf_filename_token_is_not_the_gene_name_issue_485` (`YBR174C`), `test_build_orf_to_gene_name_map_reads_gene_attribute_by_feature_id`, and the updated fixture expectations and build log.

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before: the 1,850 `Protein.Group` UniProt accessions were lifted to systematic ORFs through the SGD GFF `protein_id=UniProtKB:` cross-references and stored as the `protein_abundance` keys; `perturbed_gene_name` was the GFF `gene=` standard name of the metadata ORF (3,780 of 4,699 records), else the ORF.

Now: the protein keys are the released accessions (`P00410`, the mitochondrial COX2, included); `systematic_gene_name` and `perturbed_gene_name` are both the metadata `ORF`, upper-cased. No annotation is consulted; `build_uniprot_to_orf_map` and `build_orf_to_gene_name_map` stay in the module as downstream utilities because the Messner 2021 loader calls the first. A metadata ORF with no ORF form is counted as deferred (0 of 4,699). 4,699 records; joining an accession to a gene is a modeling-time step.

## 2026.10.12 - The deferred name is a record on the released-gene-name leaf (#1045, PR #1085)

A metadata ORF with no ORF form goes on `ReleasedGeneNamePerturbation` (0 of 4,699 strains). The count stays 4,699.

### L0-L4 from the rebuilt stores (slurm arrays 4457 and 4604, family verifiers job 4658)

**proteome_messner2023**: verdict PASS (/scratch/projects/torchcell-scratch/data/torchcell/proteome_messner2023/preprocess/verification_report.json)

| level | rule | class | outcome | message |
|---|---|---|---|---|
| L0 | `structural` | gate | ok | 4699 records validated |
| L1 | `count` | gate | ok | observed 4699, expected 4699 |
| L1 | `orf_uniqueness` | gate | ok | 4549 ORFs, 146 with multiple strains (expected) |
| L2 | `value_fidelity` | gate | ok | 8466210 values checked |
| L2 | `se_nonnegative` | gate | ok | 0 values checked |
| L3 | `reference_finite` | gate | ok | reference abundance finite + key-matched for all 8466210 values |
| L3 | `measurement_type_consistent` | gate | ok | single measurement_type: 'swath_ms_maxlfq_batch_corrected_quantity' |
| L4 | `gene_containment_scmd_ohya2005` | report | ok | 0.948 of scmd_ohya2005's 4549 deletion genes are in Ohya (>= 0.9) |
