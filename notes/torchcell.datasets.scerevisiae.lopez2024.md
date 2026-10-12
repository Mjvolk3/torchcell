---
id: 5q5cofs8uz78xralky1tnv0
title: Lopez2024
desc: ''
updated: 1783923943375
created: 1783923943375
---

## 2026.07.13 - Lopez/Montaño-López 2024 isobutanol biosensor YKO screen (in-house)

Two `MetabolitePhenotype` datasets from the in-house isobutanol GFP-biosensor screen of the
BY4741 gene-knockout collection (José de Jesús Montaño López 2024 Princeton dissertation).

- **`IsobutanolScreenLopez2024Dataset`** (`isobutanol_screen_lopez2024`) -- first genome-wide
  screen, **4554 records** (one per current-R64 ORF; 186 ORFs aggregated from >=2 plate rows,
  mean FC, n=row count, SE=SD/sqrt-n when n>=2).
- **`IsobutanolValidatedLopez2024Dataset`** (`isobutanol_validated_lopez2024`) -- validated
  re-screen (FC>=2 or FC<=0.5) in **triplicate**, **224 records** (n=3, SE=STD/sqrt3).

**Phenotype**: `metabolite_level={"isobutanol": FC}`, `measurement_type=
"biosensor_gfp_fluorescence_fold_change"`. **FC (verbatim)** = median GFP fluor(deletion) /
median fluor(WT control, same plate); reference = WT = 1.0. Biosensor = GFP alpha-
ketoisovalerate/isobutanol-pathway sensor (Leu1 promoter), integrated in every strain
(constant reporter background -- documented, not a per-record perturbation). **Genotype** =
one `KanMxDeletionPerturbation` per ORF; **environment** = SC liquid, 30 C (standard), aerobic,
no inducer (biosensor reports endogenous pathway flux). L0-L4 all pass.

**Provenance**: sha256-pinned dissertation supplementary tables (`supplementary_tables.xlsx`
f97cf13c...) + `thesis.pdf` (525e03b4...) in the mirror; thesis retrieved from Zotero item
6DQAEFWZ. Methods sourced born-digital (pdftotext Ch 3.3).

**FLAGGED for review**: (1) the dissertation has NO DOI/PMID (only Princeton DataSpace handle
88435/dsp019s161956t); `Publication` requires doi|pubmed_id so it cites the same-lab biosensor
methods paper (Montaño-López et al. Nat Commun 2022, PMID 35022416) -- the Provenance record
points to the true dissertation source. Consider a schema affordance for DOI-less dissertation
sources. (2) `YBL071W-A` dropped from S3 (listed in both up FC=3.469 and down FC=0.0757 blocks
-- contradictory). (3) temperature 30 C = standard, not explicitly stated. See memory
`[[remaining-datasets-blocked-status]]`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_XLSX_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.12 - Every released record stored as released; no genome at load time (#1045)

Owner rule, 2026-10-12: the data is the data; the genome and the experiment are loaded separately and meet through their gene sets at modeling time, so a loader never consults the annotation to decide whether a record exists or how it is keyed. Sweep row: [[database.genome-gating-sweep.2026.10.12]]. PR #1079 (group 1). The store is rebuilt from the branch with `--retire-existing --verify` (slurm array 4457); the L0-L4 table is appended when the task finishes.

Before (Table S2): rows were resolved through the genome's ID and alias tables (57 rows dropped: 25 with no alias, 31 whose alias target is a directly present ORF, 1 not ORF-shaped), 25 rows renamed to the current ORF, and the rows of one ORF averaged into one record (194 rows into 186 ORFs, SE from the sample SD). Before (Table S3): both `YBL071W-A` rows were dropped as contradictory, `YIR043C` as unresolved, 3 rows renamed.

Now: one record per released row under the released ORF (upper-cased), `n_replicates = 1` and no SE for S2, the released triplicate SD for S3. S2: 4,805 rows, 4,804 records (was 4,554); the one row with no ORF form (`YLR287-A`, the release dropped the strand letter) is counted under `deferred_to_released_gene_name_leaf` in `preprocess/dropped_records.json` and is stored by the released-gene-name leaf PR. S3: all 227 rows, both `YBL071W-A` rows included (FC 3.469 and 0.0757; the contradiction is a finding for the paper note, not a reason to drop either). The metabolite verifier runs both stores with `allow_duplicate_orfs`. The reference is one shared WT control with n 1 (S2) or 3 (S3).
