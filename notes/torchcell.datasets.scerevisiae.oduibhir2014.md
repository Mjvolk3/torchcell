---
id: pa621rbrdwtsfs5m902pkga
title: Oduibhir2014
desc: ''
updated: 1783885375641
created: 1783885375642
---

## 2026.07.12 - O'Duibhir 2014 growth-rate (fitness) build

`SmfODuibhir2014Dataset` (`torchcell/datasets/scerevisiae/oduibhir2014.py`, REPLACING a
broken Mac-path stub). O'Duibhir et al. 2014 (Holstege), *Mol Syst Biol* 10:732,
doi:10.15252/msb.20145172, PMID 24952590. Per-deletion relative growth rates →
`FitnessPhenotype`. **L0-L4 verified, 1,312 records.**

- **Classified as FITNESS, not expression** (user hypothesis confirmed): the only new
  per-strain data is Dataset S2 = relative doubling times; the paper's "expression"
  datasets are the Kemmeren 2014 compendium + PCA transforms (100% redundant with the
  existing Kemmeren microarray dataset).
- **Source (open-access SI, mirror + hash-pin):** Dataset S2 `data set 2.txt` (sha256
  `37ef19ee…`) from the EMBO/EuropePMC SI (PMC4265054 — the task's PMCID was wrong).
  Cols: ORF, commonName, `log2relT`, similarity. n=1,312.
- **Fitness direction (empirically reconciled, NOT from the docstring):** the schema field
  says `wt/ko`, but the BUILT Costanzo SMF stores sick mutants BELOW 1 (e.g. ded1-f144c
  = 0.114) → real convention is ko/wt. So `fitness = 2^(-log2relT)` (slow grower <1). Verified:
  PAF1Δ=0.573, min 0.255, max 1.183, WT ref=1.0. (Schema docstring is misleading — flag for fix.)
- n_samples=2 (biological duplicates, sourced); no per-strain SD released → uncertainty None.
  Genotype KanMxDeletionPerturbation; ReferenceGenome BY4741; SC liquid 30 C (Kemmeren setup).
- Adds a new fitness verifier `torchcell/verification/fitness.py` (L3 `reference_one`) +
  `FITNESS_DATASETS` / `run_fitness` in runners.
- FLAG: mating type (BY4741 vs BY4742) not resolvable per strain from S2 → BY4741 representative.

## 2026.09.14 - Medium moved to the shared library SC

The pre-rebuild sweep verified the rebuilt store and L3 `media_membership` failed: the loader
built `Media(name="SC", state="liquid", is_synthetic=True)` inline, a free-text medium that is
its own node in the graph and joins nothing. It now uses `torchcell.datamodels.media.SC` (the
defined synthetic complete recipe: YNB, 20 amino acids, uracil, adenine, 20 g/L glucose,
liquid), the same object Mormino 2022 and Wildenhain 2015 resolve to. The paper states SC
without a recipe (the Kemmeren 2014 setup), so the library's sourced recipe stands in for it.
Store rebuilt under the current schema before the 2026.09.14 full KG rebuild.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #537; the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes (#537); PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes (#537); refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_DATASET_S2_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Refusals before the store opens (issue #537)

Previous behavior: `process` opened `processed/lmdb` first and built each record inside the write transaction. A duplicated or lowercase-duplicated ORF gave one record per row, a blank `commonName` was stored as the string "nan", and a blank `log2relT` reached the phenotype validator as NaN fitness, which aborted the build after the store existed; a retry then found the empty store, skipped `process` and served 0 records.

Fix: every record is built first, then the store is opened and written. Two rows resolving to one ORF, a blank `commonName` and a blank `log2relT` are each refused with a `RuntimeError` naming the row. Measured on the pinned Dataset S2 (sha256 `37ef19ee...`): 0 duplicates, 0 lowercase ORFs, 0 blank common names and 0 blank `log2relT` of 1312 rows, so the built records do not change.

Tests: `test_an_orf_listed_twice_is_refused_before_the_store_opens`, `test_a_blank_common_name_is_refused_before_the_store_opens`, `test_a_blank_log2relt_is_refused_before_the_store_opens`.
