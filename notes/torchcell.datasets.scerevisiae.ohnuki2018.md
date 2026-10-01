---
id: afpnvd6q0b1naaim7sibhqa
title: Ohnuki2018
desc: ''
updated: 1783884471903
created: 1783884471903
---

## 2026.07.12 - Ohnuki 2018 essential-gene het morphology build

`ScmdOhnuki2018Dataset` (`torchcell/datasets/scerevisiae/ohnuki2018.py`). Ohnuki & Ohya
2018, *PLoS Biol* 16(5):e2005130, doi:10.1371/journal.pbio.2005130, PMID 29768403.
CalMorph morphology of the essential-gene HETEROZYGOUS diploid deletion collection →
`CalMorphPhenotype`. **L0-L4 verified, 1,112 records.**

- **Source (SCMD2, mirror + hash-pin):** `ess1112data.tsv` (1,112 het-diploid mutant
  averages, sha256 `2d168bd1…`) + `wt114data.tsv` (114 BY4743 WT reps, sha256 `f48d42da…`),
  fetched from SCMD2 per the paper's Data Availability. 501 CalMorph traits (281 base +
  220 CV) — byte-identical vocabulary to Ohya 2005 (CALMORPH_LABELS/STATISTICS, zero mismatch).
- **Het deletion = 50% dosage, NOT a KO:** genotype uses
  `EngineeredCopyNumberPerturbation(copy_number=1, reference_copy_number=2, marker="KanMX")`
  (the schema names HIP het deletion as its motivating case; SO:0001019). ReferenceGenome
  BY4743 **diploid**. Env YPD **25 C** liquid (log-phase; sourced verbatim).
- Raw per-strain CalMorph averages (not z-scores); zero NaN; 0 ORFs dropped. Only the
  YPD/25 C arm built (SD/37 C poor-medium arm has no released matrix).

## 2026.07.15 - Migrate ORF reconciliation onto the shared genome resolver

Replaced the local `_load_sgd_genes` FASTA R64 drop with the shared
`reconcile_systematic_names` helper (genome resolver; see
[[torchcell.sequence.genome.scerevisiae.s288c]] and
[[torchcell.datasets.scerevisiae.gene_name_reconcile]]). Retain-all: **0 dropped for
naming** (was already empirically zero). Build: 1112 strains — 1111 CURRENT + 1
NON_GENE_FEATURE (a valid non-`"gene"` essential locus, now retained rather than at risk of a
FASTA-vs-gene_set mismatch). Added an optional injectable `genome` (defaults to `DATA_ROOT`).

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_RAW_FILES`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Blank and duplicate ORFs, refusals before the store opens (issue #537)

Previous behavior: blank or whitespace-only ORF rows were dropped with no log line while the count read "0 dropped for naming"; two spellings of one ORF gave two records; `create_experiment` returned None; records were built inside the open write transaction, so a refused reference left an empty store a retry served as 0 records.

Fix: blank ORF rows are dropped with a counted warning, duplicate spellings are refused, `create_experiment` raises `NotImplementedError`, every record is built before the store opens, and `TCV` is removed from `_CV_PREFIXES` (as in Ohya 2005, #494). Measured on the pinned matrices (1112 x 501): 0 blank ORFs, 0 duplicate spellings, 0 TCV columns; the built records do not change.

Still pinned, not an item of #537: a blank CalMorph cell is stored as 0.0 (`create_calmorph_experiment`); the pinned matrix has 0 blank cells.
