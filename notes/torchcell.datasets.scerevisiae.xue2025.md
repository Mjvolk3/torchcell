---
id: 4vp0a9tm13r73od3gy8bpqw
title: Xue2025
desc: ''
updated: 1783966590843
created: 1783966590843
---

## 2026.07.13 - Xue 2025 in-house combinatorial FFA-titer dataset

`FattyAcidXue2025Dataset` -- FIRST COMBINATORIAL-genotype metabolite dataset (up to 6
deletions/strain). 176 records + 1 measured WT reference, from the in-house Xue 2025 FFA
overproduction screen.

- **Genotype**: every non-WT strain = the `POX1-FAA1-FAA4` FFA-overproduction baseline
  (3 deletions, implicit in ALL strains) + 0-3 TF deletions decoded from the genotype string
  (`<letters> <N>Δ`, letters via the Abbreviations sheet, `#TF + 3 == N`). `+ve Ctrl` =
  baseline only. One `KanMxDeletionPerturbation` per gene (**markers unknown/mixed for
  in-house combos -> KanMX is a documented REPRESENTATIVE**; state=absent is the asserted
  fact -- can't stack 6 KanMX). 13 genes all resolve to R64. Decode source =
  `experiments/008-xue-ffa` parser.
- **Phenotype**: `MetabolitePhenotype`, `metabolite_level={C14:0,C16:0,C18:0,C16:1,C18:1: mg/L}`,
  `measurement_type="titer_mg_per_l"`, `metabolite_level_se` = sample SD/sqrt(n) PER FFA (n
  computed from actual non-blank reps: mostly 3; 10 strains n=2, `G-O-T 6Δ` n=1 -> SE NaN,
  which the verifier skips). `reference_centered=False`. `target_metabolite_ids=None` (defer
  ChEBI/Yeast9). Env = SC/30C aerobic (in-house-assumed, flagged).
- **Reference**: measured wt BY4741 titers.

**Verifier change**: metabolite L1 `orf_uniqueness` -> `genotype_uniqueness` (keys on the full
deletion-set signature, not per-ORF) so combinatorial strains where an ORF recurs across many
strains pass; backward-compatible (single-deletion datasets reduce to per-gene; confirmed on
isobutanol). L0-L4 all pass (176 records).

**FLAGGED**: (1) in-house/unpublished -> `Publication` anchors to the FFA-chassis paper
Runguphan & Keasling 2014 (PMID 23899824); `Provenance` points to the in-house xlsx (sha256
023de80e). (2) KanMX marker is a representative (real markers mixed/unknown). (3) SC + 30C
assumed. See memory `[[remaining-datasets-blocked-status]]`.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.
