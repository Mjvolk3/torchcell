---
id: xf5dfc7aasskwpqnx1b1lbi
title: Nadal_ribelles2025
desc: ''
updated: 1784006057031
created: 1784006057031
---

## 2026.07.14 - Nadal-Ribelles 2025 pseudobulk Perturb-seq build

Loader: `torchcell/datasets/scerevisiae/nadal_ribelles2025.py`
(`NadalRibellesPerturbSeq2025Dataset`). Source: Nadal-Ribelles et al. 2025,
*Nat. Commun.* 16, doi:10.1038/s41467-025-57600-4; data Zenodo
10.5281/zenodo.14062629.

### What it is

Genome-scale single-cell **Perturb-seq** of an RNA-barcoded YKO deletion library
(KANMX4 marker swapped for URA3 so the genotype barcode in the URA3 3'UTR is
polyA-readable), profiled under CONTROL and osmostress (0.4 M NaCl, 15 min).
**LOCKED representation: pseudobulk-per-genotype + dispersion, NOT per-cell** (the
43 GB single-cell object is not used).

### Representation

- New phenotype family `PseudobulkExpressionPhenotype` (schema.py), a sibling of the
  RNA-seq expression family: `expression_log2_ratio` (per-gene pseudobulk log2 FC vs
  same-condition WT; scanpy `logfoldchanges`, Wilcoxon DE) + two per-genotype
  single-cell scalars — `dispersion` (`sd_lvscore_scaledFU2`, SD of the scaled SVD
  leverage score; WT ~= 1) and `n_cells` (`cell_number`). Both scalars are the POINT
  of the pseudobulk+dispersion decision. `dispersion`/`n_cells` are optional
  (default None) → backward-compatible additive fields.
- Genotype: one `MarkerDeletionPerturbation(marker="URA3")`; the source genotype
  barcode label is carried on **`strain_id`** (new optional field on
  `MarkerDeletionPerturbation`, default None → backward-compatible). This preserves
  replacement strains (`bc_YBR020W-1`/`-2`) that delete the same ORF as DISTINCT
  records (they carry different dispersion/n_cells).
- Environment: 2 records per genotype — control (base YPD) and osmostress (YPD +
  0.4 M NaCl `SmallMoleculePerturbation`, `duration_hours=0.25`).
- Reference: per-condition WT — `expression_log2_ratio` 0 for every gene, carrying
  the WT dispersion (~1.001) and n_cells (control 500 / NaCl 458). Two references.

### Build result (6188 records)

- Control 3091 + NaCl 3097 = **6188** records (one per genotype×condition); 3150
  distinct strains; 0 ORF-labels unparsed.
- **Ragged** gene vectors (per-comparison 0-count genes dropped): min 4223, max
  6313, median ~5837; union 6796. Gene *common* names → current-R64 systematic ORFs
  via the genome alias table; a gene not tested for a genotype is KEY-ABSENT (never
  0). 35.2M gene-values kept; ~1.13M (3.1%) dropped as unresolvable (ncRNA/retired
  symbols e.g. `15S-RRNA`, `SNR*`, `RUF*`); 15.7k within-record alias collisions
  deduped (two source names → one ORF, keep first).
- Verification (`RNASEQ_DATASETS`, `run_rnaseq`) **L0–L4 PASS**: L4 SGD gene
  containment 1.000 (6352 measured genes). L1 uniqueness extended to key on
  (strain_id, environment) so a strain profiled in both conditions is two records
  (backward-compatible with Caudal's single-condition survey).

### Provenance flag for review — assay-phase TEMPERATURE

The paper does NOT explicitly state the temperature of the 6-h YPD growth +
15-min NaCl treatment of the profiled cells. Only the **48-h URA- recovery is
stated, at 25 °C** (Methods, "Yeast Growth and harvest for the Perturb-seq
experiments"). **30 °C** is used as the documented representative (standard
*S. cerevisiae* growth temperature, and the temperature this paper uses for its own
growth-curve validations) and is **FLAGGED for review**. It is a shared constant
across both conditions and every genotype, so it does not affect any mutant-vs-WT
log2 FC — only the environment metadata. `GROWTH_TEMP_C` in the loader.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no (symlink); refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `SHA256_EXPECTED`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Repeated ptbs labels and unknown conditions are refused (issue #541)

Previous behavior: a `ptbs` label on two rows of one condition kept the first row's `sd_lvscore_scaledFU2` and `cell_number` with no log and no batch field; any condition other than `nacl` was stored with the control environment.

Fix: `_load_ptbs` raises `RepeatedPtbsLabelError` naming the condition and the repeated (normalized) labels; `_environment` raises `UnknownConditionError` for any condition outside `CONDITIONS = ("control", "nacl")`, which happens while the references are built, before the store opens. `_ptb_scalars` no longer carries the `iloc[0]` branch.

Evidence on the pinned files: `ptb_summary.Rdata` has 3,207 control and 3,204 nacl rows, 0 repeated labels, no batch column; `FC_genotype.Rdata` has 6,188 keys, Control 3,091 and NaCl 3,097, no other condition. No stored record changes. The closest thing to a repeat is the same ORF under replacement-strain labels (`bc-YBR020W-1`, `-2`): 55 ORFs in control (111 rows) and 57 in nacl (115 rows), each its own record. The spread in `sd_lvscore_scaledFU2` between such strains has median 0.100 (max 0.713, YGR180C) in control and 0.074 (max 0.53) in nacl. Tests: `test_a_label_repeated_within_a_condition_is_refused`, `test_ptbs_lookup_is_by_label_not_row_position`, `test_an_unknown_condition_is_refused_before_any_store`.
