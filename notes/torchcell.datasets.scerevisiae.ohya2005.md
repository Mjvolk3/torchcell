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
