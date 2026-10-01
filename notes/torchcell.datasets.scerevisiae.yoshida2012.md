---
id: gm71e202w4ajfj5zfs1ox97
title: Yoshida2012
desc: ''
updated: 1783835401860
created: 1783835401860
---

## 2026.07.12 - Yoshida 2012 organic-acid titer build

`OrganicAcidYoshida2012Dataset` (`torchcell/datasets/scerevisiae/yoshida2012.py`). Yoshida
& Yokoyama 2012, *J Biosci Bioeng* 113(5):556, doi:10.1016/j.jbiosc.2011.12.017, PMID
22277779. HPLC organic-acid titers of 17 selected KO overproducers → `MetabolitePhenotype`.
**L0-L4 verified, 17 records.**

- **Born-digital source (OCR was garbled):** the data is Table 3 in the paper (no SI file).
  MinerU OCR scrambled Table 3's numeric cells; recovered cleanly via `pdftotext -layout`
  on the sha256-pinned mirror `paper.pdf` (WT row + 17 genes × {OD, Ace, Cit, Mal, Pho, Pyr,
  Suc}, mM, mean±SD, n=3). Values transcribed into a vetted module-level `TABLE_3` literal
  (deterministic; PDF not re-parsed at build). Illustrates the born-digital-first rule.
- **17 records** (one per gene). metabolite_level = {acid: mM} for acetate/citrate/malate/
  pyruvate/succinate + phosphate. **OD dropped** (biomass, not a metabolite).
  measurement_type=`hplc_organic_acid_titer_mM`, n_replicates=3, se=SD/√3 (sample SD).
- **target_metabolite_ids → Yeast9 s_NNNN** (via `build_metabolite_s_id_map`): acetate s_0362,
  citrate s_0522, malate s_0066, pyruvate s_1399, succinate s_1458. Phosphate unmapped (inorganic).
- **Measured WT reference** = the Table 3 WT (BY4742) row, restricted per record to measured
  analytes. Static YPD liquid, 25 C, 72 h. All 17 genes resolved to R64 ORFs.
- **DEFERRED layers:** the 36-gene BCP-halo categorical screen (Table 2 ordinal per-acid
  fold-change) — no clean existing schema fit; Table 4/5 out of scope (overexpression / chem
  sensitivity). Table 3 titers are the clean quantitative layer.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `PDF_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Genome-checked names and refusals before the store opens (issue #537)

Previous behavior: a systematic-shaped name skipped the genome (a nonexistent YAL999W was stored), an alias with two candidate ORFs took the first, a strain written by its common and its systematic name gave two records, and a NaN cell was stored as a NaN level and SE. `data.csv` and the store were written while records were still being built.

Fix: a systematic-shaped name must be in `genome.gene_set`; an alias must name exactly one ORF; two rows resolving to one ORF and a NaN mean or SD in any Table 3 cell are refused with a named `RuntimeError`. Every record is built before `data.csv` or the store is written. Measured on the released Table 3 (17 strains, 126 cells): `YDR379C-A` is a gene of R64, every common name has exactly one candidate, 0 duplicates, 0 NaN cells, so the built records do not change.

Test: `test_a_bad_table_row_is_refused_before_anything_is_written` (four cases).
