---
id: 2pb4smgtn8c90wqisw8vsar
title: ohya2005
desc: ''
updated: 1757482771391
created: 1755123685272
---

## Ohya2005 CalMorph Morphology Dataset

### Overview

The Ohya2005 dataset contains morphological measurements from the CalMorph software for yeast cell imaging analysis. The 501-trait CalMorph matrices are **Ohya et al. (2005, PNAS)'s own published data**, distributed via the SCMD (Saccharomyces cerevisiae Morphological Database) portal. (Correction 2026.07.15: an earlier version of this note stated the data was "published by Suzuki et al. 2018 after reanalyzing images." That is wrong — Suzuki et al. 2018, BMC Genomics 19:149, merely *reused* this same dataset as its reference [21]; it did not generate the values. The loader's Ohya-2005 citation is correct. See the 2026.07.15 provisioning section below.)

### Dataset Description

#### Non-essential gene deletion mutants

- **4718 gene deletion mutants**
  - Average data (27.7 MB)
  - Number of cells for ratio parameter (885.6 KB)
  - Number of cells in specimen for ratio parameter (1.14 MB)
  
#### Wild-type references

- **122 replicated wild-type (his3)**
  - Average data (1.06 MB)
  - Number of cells for ratio parameter (23.4 KB)
  - Number of cells in specimen for ratio parameter (29.9 KB)

### Data Source

#### Primary URLs (from SCMD - Saccharomyces cerevisiae Morphological Database)

- Mutant data: <http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=mt4718data.tsv>
- Wild-type data: <http://www.yeast.ib.k.u-tokyo.ac.jp/SCMD/download.php?path=wt122data.tsv>

Cell images are available at SSBD:ssbd-repos-000349

### Supplementary Information Files

The supplementary information from the original papers has been converted from PDF to markdown format using MathPix tools for easier processing.

#### SI_1.mmd - CalMorph Parameter Descriptions

**Path:** `/Users/michaelvolk/Library/CloudStorage/Box-Box/torchcell/data/host/SmfOhya2005/SI_1.mmd`

This file contains a comprehensive table of all CalMorph morphological parameters organized by nuclear stage:

- **Stage_A**: Unbudded cells (73 parameters + 73 CV parameters)
- **Stage_A1B**: Small/medium budded cells (226 parameters + CV)
- **Stage_C**: Large budded cells (461 parameters + CV)
- **Total_stage**: Aggregate measurements across all stages (501 parameters)

Parameter categories:

- **C-parameters**: Cell morphology (size, shape, wall)
- **A-parameters**: Actin organization and distribution
- **D-parameters**: Nuclear morphology and position
- **CV-parameters**: Coefficient of variation for each measurement

#### SI_2.mmd - Statistical Information

**Path:** `/Users/michaelvolk/Library/CloudStorage/Box-Box/torchcell/data/host/SmfOhya2005/SI_2.mmd`

This file contains statistical analysis of the parameters:

- Box-Cox power transformation parameters
- Shapiro-Wilk P-values for normality testing
- Number of disruptants at various significance thresholds (E-06, E-05, E-04, E-03)
- Used for data quality validation and transformation

**Note:** SI_2 has a complex multi-line header structure that requires careful parsing.

### Implementation Details

#### Data Processing Workflow

1. **Download**: Data is downloaded with Safari headers due to server restrictions
2. **Preprocessing**: Raw TSV data is parsed and gene names are cleaned
3. **Reference Calculation**: 122 wild-type replicates are averaged to create reference phenotype
4. **Storage**: Data is stored in LMDB format with serialized Pydantic models

#### CalMorph Phenotype Structure

The morphology measurements are stored as a dictionary in the `CalMorphPhenotype` class:

```python
morphology: Dict[str, float]  ## e.g., {"C11-1_A": 123.45, "A101_A": 0.67, ...}
```

#### Key Features

- Handles multiple wild-type references (122 replicates) unlike other datasets with single references
- Comprehensive morphological profiling with 501 parameters per strain
- Three cell cycle stages captured separately
- Coefficient of variation included for robustness analysis

### References

1. Ohya Y, et al. (2005) High-dimensional and large-scale phenotyping of yeast mutants. PNAS 102(52):19015-20. [PubMed: 16365294](https://pubmed.ncbi.nlm.nih.gov/16365294/) — **the data source** (SCMD-distributed).

2. Suzuki G, et al. (2018) Global study of holistic morphological effectors in the budding yeast Saccharomyces cerevisiae. BMC Genomics 19:149. [PubMed: 29458326](https://pubmed.ncbi.nlm.nih.gov/29458326/) — *reused* the Ohya-2005 dataset (its ref [21]); did NOT generate it.

### Integration Status

- [x] Dataset class implemented (`SmfOhya2005Dataset`)
- [x] Adapter created (`SmfOhya2005Adapter`)
- [x] Schema updated with `CalMorphPhenotype` and related classes
- [x] BioCypher configuration updated
- [x] Dataset registered in adapter map
- [ ] Fallback download URLs to be added when shared storage available
- [ ] Complete CALMORPH_LABELS dictionary population from SI_1.mmd

## 2026.07.15 - Rebuild-provisioning audit + corrections

Pre-graph-rebuild audit of the SCMD CalMorph morphology datasets (Ohya 2005 +
Ohnuki 2018/2022). Ohya 2005 was the least-provisioned of the three (the other two
already had sha256-pinned mirrors + verbatim-sourced environments). This section
records the remediation. Branch `fix/ohya2005-mirror-provenance`.

### Mirror + sha256 pin (was: live-URL download, no pin)

Deposited the two SCMD average-data matrices into the library mirror
`$DATA_ROOT/torchcell-library/ohyaHighdimensionalLargescalePhenotyping2005a/`
(`data/` + `manifest.json`), and switched `download()` to read the mirror and verify
both hashes (same pattern as the Ohnuki loaders) instead of hitting the live SCMD
portal. Pins:

- `mt4718data.tsv` (4718 mutants x 501 features; ID `ORF`) sha256
  `c4ba1e84b4ea6273f0162ef9230e15634933c8c0c4910dd7546a21c6293e0fc0`
- `wt122data.tsv` (122 his3 WT replicate averages; ID `NAME`) sha256
  `ab2c31b5150b2a33c15b5d22f1bef8687719975223559a740ea233c1f67b27c3`

The SCMD portal URL + Box fallback are retained as retrieval metadata in the manifest
(both were live at retrieval, each ~27.7 MB / 1.06 MB). Verified at retrieval: 501
feature columns == exactly the CalMorph vocabulary (281 `CALMORPH_LABELS` + 220
`CALMORPH_STATISTICS`), 0 out-of-vocab columns, 0 rows with missing values.

### Environment corrected: YEPD/solid/30 C -> YPD/liquid/25 C (now sourced)

The old environment was unsourced and disagreed with the two Ohnuki CalMorph loaders.
Sourced from Ohya 2005 Methods (verbatim, via PMC1316885): *"Each strain was grown in
yeast extract/peptone/dextrose medium, and logarithmic-phase cells were fixed."* ->
media YPD, state liquid (log-phase liquid culture). Temperature is NOT stated in Ohya
2005; resolved to 25 C via the Ohya-lab CalMorph standard, corroborated by Suzuki 2018
("grown at 25 C") and the same-lab Ohnuki 2018/2022 loaders (deferral convention).
FLAG: pin the verbatim Ohya-2005 temperature once that paper is mirrored.

### NaN policy: impute-to-0.0 -> drop-whole-row

The old loader converted any missing CalMorph value to `0.0` (silent imputation). Now
rows with any missing value are dropped whole (never imputed), matching Ohnuki 2022. On
the current pinned file this is a no-op (0 NaN rows), but it removes a latent
corruption path on any future re-retrieval.

### R64 gene-name reconciliation (record count 4718 -> 4695)

Added R64-4-1 validation (Ohnuki 2018 pattern). 27 of the 4718 legacy 2005-annotation
ORFs do not resolve to R64-4-1. Per user decision (2026.07.15) "map mappable, drop
retired":

- **4 remapped in place** (SGD alias-based renames, R64 GFF `Alias` field; target NOT
  otherwise measured in the screen): YGR272C->YGR271C-A (EFG1), YIL015C-A->YIL014C-A,
  YLR391W->YLR390W-A (CCW14), YMR158C-B->YMR158C-A. `_LEGACY_ORF_RENAMES` in the loader.
- **6 dropped as merge-collisions**: YDL038C->YDL039C, YDL134C-A->YDL133C-A,
  YER108C->YER109C, YIL168W->YIL167W, YIR044C->YIR043C, YML033W->YML034W. Each legacy
  ORF is an R64 alias of a gene that ALSO has its own strain record in the screen (an
  SGD merge of two 2005 ORFs); remapping would duplicate that gene's morphology, so the
  legacy strain is dropped and the canonical target strain retained.
- **17 dropped as retired** (removed from SGD since 2005, mostly dubious `-A`/`-B`
  ORFs): YAL043C-A, YAL058C-A, YAR037W, YAR040C, YAR043C, YGL154W, YIR020W-B,
  YML010C-B, YML010W-A, YML013C-A, YML035C-A, YML048W-A, YML058C-A, YML095C-A,
  YML102C-A, YML117W-A, YMR158W-A.

Net: 4695 built records (4 renamed, 23 dropped). Rebuild verified from the pinned
mirror: 4695 records, media YPD/liquid, temp 25, 281+220 vocabulary, Ohya-2005 citation.

### Downstream follow-ups (NOT done here)

- **Stale LMDBs must be cleared before the graph rebuild.** The existing
  `$DATA_ROOT/data/torchcell/scmd_ohya2005` and `$DATA_ROOT/database/data/torchcell/
  scmd_ohya2005` still hold the OLD 4718-record / YEPD-solid-30 build; the dataset base
  class skips processing when `processed/` exists, so these must be deleted so the
  corrected loader re-runs.
- **Supported-datasets table** (`notes/paper.supported-datasets-and-databases.md`, row
  "Ohya 2005"): 4718 -> 4695 on regeneration (regenerate via its script, do not hand-edit).
- **Abstract/paper morphology result** (r=0.619, single-KO) was computed on the 4718-record
  build; it should be re-run on the 4695-record dataset for consistency.
- **Verifier** (`torchcell/verification/morphology.py`): count oracle 4718 -> 4695.

## 2026.07.15 - Supersede the 4695/drop reconciliation with the shared genome resolver

The 2026.07.15 section above (4718 -> 4695, drop 23 via a hand-authored `_LEGACY_ORF_RENAMES`
table) is **superseded**. That per-loader table was error-prone: a crude GFF grep found only 4
renames and mis-bucketed the other 23. The loader now calls the shared
`SCerevisiaeGenome.resolve_gene_name` (see [[torchcell.sequence.genome.scerevisiae.s288c]]).

Retention policy (per user decision "track the perturbation as long as we know it"): **NO
record is dropped for a naming reason** -- every strain is a real measured deletion. A name
that resolves to a current R64 identifier (live gene, SGD rename, or valid non-`"gene"`
feature) is remapped to it; a name whose remap would collide with another strain's record (an
SGD merge of two distinct 2005 ORFs) or that SGD retired entirely is kept verbatim as its
legacy 2005 systematic name.

Build outcome (4718, 0 dropped): resolver statuses `{current: 4678, renamed: 20,
non_gene_feature: 16, retired: 4}`; **17 remapped** to current ids; **12** kept as legacy
names on 6 merge-collisions (YDL038C/YDL039C, YDL134C-A/YDL133C-A, YER108C/YER109C,
YIL167W/YIL168W, YIR043C/YIR044C, YML033W/YML034W); **4** retired legacy names (YAR037W,
YAR040C, YAR043C, YGL154W). `OHYA_EXPECTED_COUNT` back to 4718; the paper's r=0.619 dataset
(4718) is reproduced.

## 2026.07.22 - CalMorph normalization sourced from the SI (WS10b)

The Ohya-2005 SI prescribes the CalMorph normalization; used for the morphology training
target in experiment 019. Full analysis + variance study + implemented transform:
[[experiments.019-simb-multimodal.scripts.calmorph_variance_analysis]].

- **281 vs 501 resolved:** served `calmorph` = the 281 BASE params (`CALMORPH_LABELS`);
  `calmorph_coefficient_of_variation` = the 220 CV params (`CALMORPH_STATISTICS`); 501 total.
- **Paper's own per-parameter pipeline** (SI `si12.md`, Supporting Text): WT data *"divided
  by the mean"* -> Box-Cox `F_{p,a}(x)` (Anderson-Darling-minimizing params) -> standardized
  *"y = (F_{p,a}(x) - Mean)/SD"*, then a **Shapiro-Wilk** normality filter that **kept 254 of
  501 and discarded 247** as unreliable (P>=0.5 threshold; paper.md line 25/38).
- **Near-constant features** (the "values barely change across samples" set): on 4718
  mutants, 3 features have zero IQR - `A113_A1B`/`A113_C` (Actin_n_ratio = no-actin-patch
  ratio) and `C123_C` (Small_bud_ratio). 0 truly-constant. Per-feature scale spans ~8 orders
  of magnitude (|mean| 6.7e-5 .. 1.49e4), the root cause of the O(1e6) morphology loss.
- **torchcell decision:** KEEP all 281 with a per-feature z-score (train-split only,
  epsilon-floored) + FLAG the degenerate set for user review; do NOT import the paper's
  247-drop (a normality verdict over base+CV, not a variance floor on our 281 base target).

## 2026.09.29 - Independent re-verification

A read-only Fable 5.1 agent graded eight recorded claims against the mirror (paper, SI, raw TSVs), the SCMD and CalMorph portals, the CalMorph User Manual, Suzuki 2018, SSBD and the dev LMDB: CONFIRMED 4, REFUTED 0, PARTLY 4, UNVERIFIABLE 0. Consolidated in [[datasets.showcase-verification.2026.09.29]]; raw report `notes/assets/verification/2026.09.29/ohya2005.md`.

- The distributed matrices are the Suzuki 2018 CalMorph 1.2 re-analysis of the 2005 images (SCMD2 datasheet page: "The data sheets here have been published by Suzuki et al. (2018, BMC Genomics) by reanalysing the images first published in Ohya et al. (2005, PNAS) after a quality control."). The 2026.07.15 statement that they are "Ohya 2005's OWN published data" is wrong, and the manifest's `reused_by_doi` is the wrong relation. #491.
- SSBD ssbd-repos-000349 (DOI 10.24631/ssbd.repos.2024.05.349) publishes sha256 values equal to both loader pins and should be the cited source; the cell-count companion files (`mt4718nmrt.tsv`, `mt4718dmnt.tsv`) are not mirrored. #492, #493.
- 25 C: Suzuki 2018's 25 C sentences concern its 19-mutant validation set, not this dataset; the same-lab source is the CalMorph User Manual section 6.1 ("3. Culture the cells at 25°C on a rotator at 25-30 r/min."). The FLAG in the 2026.07.15 section closes as "the paper is mirrored and contains no temperature". #494.
- The loader comment's BY4741 genotype has `lys2D0` (BY4742's marker; BY4741 is `met15Δ0`), and the `TCV` CV prefix does not exist (CCV 60 + ACV 33 + DCV 127 = 220). #494.
- Three wild-type counts: the paper imaged 126, the average file has 122 rows, Suzuki 2018 selected 109. The mean-WT reference is correct as a torchcell choice but is not the paper's Box-Cox standardization. #494.
- The `2005a` mirror directory is a deliberate `dataset_raw_mirror` record with no `paper.md`, not a broken capture as #271 says; the lit sync should read `kind`. #495.
- Four strains the Supporting Text flags (rad18 an α cell, ctf8 an a/α mixture, scp160 2N, rho4 unlinked to the cassette) carry no qc flag. #496.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: yes; refused deposit leaving a directory: n/a.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `_RAW_FILES`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.10.01 - Interning, TCV prefix, duplicate and blank ORFs (issues #537, #546)

Previous behavior: `process` wrote every record inline (no `interned` env, unlike the thirteen interning loaders); `TCV` was in `_CV_PREFIXES` although no TCV parameter exists (#494); two spellings of one ORF gave two records; blank ORF rows were dropped silently; `create_experiment` was a bare `pass`; records were built inside the open write transaction, so a refusal left an empty store.

Fix: records are built first and written through `_open_write_lmdb` + `_intern_record`, so the constant reference (the 501-feature WT phenotype) is stored once in `processed/interned` and each record carries a `$ref` pointer; `_CV_PREFIXES` is `("CCV", "ACV", "DCV")`; duplicate spellings are refused; blank ORF rows are dropped with a counted warning; `create_experiment` raises `NotImplementedError`.

Record values are unchanged: `test_the_interned_store_resolves_to_exactly_the_inline_records` builds the synthetic matrices with the shipped writer and with the earlier inline writer and compares every resolved record exactly. The on-disk layout of a rebuilt store does change; the dev store at `$DATA_ROOT/data/torchcell/scmd_ohya2005` keeps its inline layout until its next rebuild. Measured on the pinned matrices: 4718 x 501, CV columns DCV 127 + CCV 60 + ACV 33 = 220, 0 TCV, 0 blank ORFs, 0 duplicate spellings (every ORF is lowercase, all distinct after uppercasing), 0 missing or non-numeric cells.

Left open: the publication is Ohya 2005 while the matrices are the Suzuki 2018 CalMorph 1.2 re-analysis (#491); that changes every stored record and belongs to a database-labelled PR.

## 2026.10.01 - Review follow-up on PR #591

- The duplicate-spelling refusal runs before the incomplete-row drop, so a strain listed twice with one incomplete row is now refused where the earlier loader kept one record. The pinned matrices have 0 such rows (0 duplicate spellings at all).
- `data.csv` is written after every record is built, so a refused matrix leaves neither `preprocess/data.csv` nor `processed/lmdb`. On success the file bytes are unchanged (`test_side_files` asserts them exactly).
