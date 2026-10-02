---
id: 7obg80ny7el85mk9r2eh4bf
title: Kemmeren2014
desc: ''
updated: 1757648227528
created: 1757467231171
---

## Kemmeren2014 Dataset Implementation

### Dataset Overview

- Paper: "Large-scale genetic perturbations reveal regulatory networks and an abundance of gene-specific repressors" (Kemmeren et al., 2014)
- PubMed ID: 24766815
- DOI: 10.1016/j.cell.2014.02.054

### Data Sources

The Kemmeren dataset is split into two GEO accessions:

1. **GSE42527**: Responsive mutants (deletion mutants with strong expression changes)
2. **GSE42526**: Non-responsive mutants (deletion mutants with minimal expression changes)
3. **Supplementary Table S1**: Excel file with strain information (BY4741 vs BY4742) for each deletion

Both GEO datasets need to be combined to get the full ~1484 deletion mutants dataset.

### Technical Design

- **Dye-swap design**: Each deletion has multiple measurements with dye swaps
  - Sample `-a`: typically deletion in ch2, reference in ch1
  - Sample `-b`: reference in ch2, deletion in ch1 (dye swap)
- Averaging dye-swaps provides one expression profile per deletion with technical std
- **Strain information**: Critical for each deletion (BY4741 MATa or BY4742 MATalpha)

### Major Implementation Challenges Solved

#### 1. Gene Name Resolution (Most Complex)

The dataset has a complex gene name resolution problem due to multiple naming systems:

**The Problem:**

- GEO samples use common gene names (e.g., CDK8, MED13, RIS1)
- Excel Table S1 uses systematic names in "orf name" column (e.g., YPL042C, YDR443C)
- Some genes have multiple aliases (CDK8 is also SSN3, CAC1 is also RLF2)
- Excel has duplicate orf names with different common names (YDR443C appears as both MED13 and SSN2)
- Non-standard gene names exist (TLC1, SNR10 with non-standard systematic names)

**The Solution - Multi-Pass Resolution Strategy:**

1. **Pass 1**: Direct Excel mapping (common_to_systematic from Table S1)
2. **Pass 2**: gene_attribute_table lookup (one-to-one SGD mappings)
3. **Pass 3**: alias_to_systematic lookup (handles one-to-many aliases)
4. **Pass 4**: Direct systematic name check in Excel
5. **Fallback**: Log as unresolvable

**Key Fixes:**

- Use "orf name" column from Excel as authoritative source
- Build both common_to_systematic and systematic_to_strain mappings
- Allow multiple aliases to map to same systematic name (removed already_assigned filter)
- Track and log resolution statistics

#### 2. Performance Optimization

- **Problem**: Processing hung at 0% due to slow gene name conversion in expression extraction
- **Solution**: Removed conversion from inner loop processing 15,000+ probes per sample

#### 3. Schema Validation

- **Problem**: Strict regex validation rejected non-standard gene names
- **Solution**: Relaxed validation to accept any non-empty string as systematic name

#### 4. Duplicate Logging

- **Problem**: Gene names appeared in both title and characteristics, causing duplicate resolution attempts
- **Solution**: Added `gene_resolved_from_title` flag to prevent redundant processing

#### 5. Special Case Gene Names

- **HSN1**: Historical SGD alias for YHR127W that was withdrawn but retained for compatibility
- **CycC/CYCC**: Case-sensitive aliasing issue resolved with case-insensitive matching
- These are handled via `SPECIAL_GENE_MAPPINGS` dictionary and case-insensitive fallback

### Current Implementation Status

#### ✅ Completed Features

1. **Data Loading**
   - Downloads and processes both GEO datasets (GSE42527, GSE42526)
   - Loads supplementary Table S1 with strain information
   - Extracts probe-to-gene mappings from GPL11232 platform
   - Downloads wildtype reference datasets (GSE42241, GSE42240, GSE42217, GSE42215)

2. **Expression Processing**
   - Groups samples by gene deletion for dye-swap averaging
   - Calculates mean expression and technical std across replicates
   - Handles genes with no expression data gracefully
   - **Multiprocessing support** with `process_workers` parameter for parallel batch processing

3. **Gene Name Resolution**
   - Comprehensive multi-pass resolution system
   - Handles complex aliasing (MED13/SSN2 → YDR443C)
   - Special case mappings (HSN1 → YHR127W, case-insensitive matching for CycC)
   - Preserves all common name mappings from Excel
   - Tracks resolution statistics (by Excel, gene_table, alias, unresolved)

4. **Quality Features**
   - Technical std as quality metric for each gene
   - CV-scaled standard deviations using refpool measurements
   - Comprehensive logging with resolution summary
   - Clean error handling for missing data
   - Validation of log2 ratios against original GEO data (correlation = 1.000)

### Output Statistics

When run successfully, the dataset produces:

- **1483 unique gene deletions** (from 2633 GEO samples)
- **6169 genes** with expression measurements per deletion
- **Resolution summary** showing how genes were mapped:
  - Resolved by Excel mapping: ~2489
  - Resolved by gene_attribute_table: ~81
  - Resolved by alias_to_systematic: ~63
  - Could not resolve: 0
- **Missing gene**: YCR087C-A (present in Excel but not in GEO samples)

### Wildtype Reference Implementation (Refpool)

#### Understanding the Refpool Design

The Kemmeren dataset uses a **refpool** (reference pool) design where:

- **Refpool** = Pooled RNA from many wildtype strains (HybSet)
- Used as common reference across ALL experiments
- Each deletion mutant is hybridized against this refpool

#### Wildtype Reference Datasets

Four GEO datasets provide wildtype measurements:

**MATa (BY4741)**:

- **GSE42241**: Tecan plate, 20 samples
- **GSE42240**: Erlenmeyer flask, 8 samples
- Total: 28 MATa samples

**MATα (BY4742)**:

- **GSE42217**: Tecan plate, 200 samples  
- **GSE42215**: Erlenmeyer flask, 200 samples
- Total: 400 MATα samples

#### Refpool Processing Strategy

1. **Extract refpool from WT samples**:
   - Each WT sample contains "wt vs. refpool" or "refpool vs. wt" hybridizations
   - Detect which channel (Cy3 or Cy5) contains refpool based on sample naming
   - Extract refpool values (same pooled RNA measured many times)

2. **Calculate Coefficient of Variation (CV)**:
   - For each gene: CV = std/mean across all refpool measurements
   - CV is scale-independent, allowing transfer across different normalizations
   - Typical results: Median CV ≈ 0.49, ~49% of genes have CV > 0.5

3. **Apply CV to deletion samples**:
   - Extract refpool from deletion samples' Cy3/Cy5 (depending on dye-swap)
   - Calculate scaled std = CV × refpool_value_in_deletion_sample
   - This provides proper noise estimates at experimental scale
   - Store as `expression_log2_ratio_std` in MicroarrayExpressionPhenotype

### Performance Optimization

#### Multiprocessing Implementation

**Parameters**:

- `io_workers`: Controls parallel data loading (inherited from ExperimentDataset)
- `process_workers`: Controls parallel experiment processing (new in Kemmeren2014)
- `batch_size`: Number of experiments to process per batch (default: 10)

**Performance Gains**:

- Sequential processing: ~17:24 minutes for 1483 experiments
- Parallel processing (10 workers): ~2:37 minutes (6.6x speedup)
- ProcessPoolExecutor used for CPU-bound DataFrame operations
- Static methods created for multiprocessing compatibility

**Key Implementation Details**:

- Batches gene deletions for parallel processing
- Uses static methods to avoid pickling issues
- Maintains data integrity - identical results between sequential and parallel
- Clean separation of IO-bound (LMDB writes) and CPU-bound (DataFrame processing) work

#### VALUE Column Convention

**IMPORTANT**: The VALUE column in GEO follows standard microarray convention:

- VALUE = log2(Cy3/Cy5) = log2(refpool/deletion)
- Negative VALUE = deletion has HIGHER expression than refpool
- Positive VALUE = deletion has LOWER expression than refpool
- Validation shows perfect correlation (1.000) between VALUE and calculated log2(Cy3/Cy5)

### Known Limitations

1. Missing gene YCR087C-A in GEO data (present in Excel Table S1 but not in GEO samples)
2. Duplicate orf names in Excel require careful handling (e.g., YDR443C appears as both MED13 and SSN2)
3. PyTorch Geometric library warnings about missing dependencies (does not affect functionality)

### Usage

```python
import os
import os.path as osp
from dotenv import load_dotenv
from torchcell.datasets.scerevisiae.kemmeren2014 import MicroarrayKemmeren2014Dataset
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

# Setup environment
load_dotenv()
DATA_ROOT = os.getenv("DATA_ROOT")

# Initialize genome for gene name mapping (optional dependency injection)
genome = SCerevisiaeGenome(
    genome_root=osp.join(DATA_ROOT, "data/sgd/genome"),
    go_root=osp.join(DATA_ROOT, "data/go"),
    overwrite=True,
)

# Sequential processing (original)
dataset = MicroarrayKemmeren2014Dataset(
    root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
    genome=genome,  # Inject genome dependency
    io_workers=0,
    process_workers=0  # Sequential processing
)

# Parallel processing (faster, recommended)
dataset = MicroarrayKemmeren2014Dataset(
    root=osp.join(DATA_ROOT, "data/torchcell/microarray_kemmeren2014"),
    genome=genome,       # Inject genome dependency
    io_workers=10,       # For parallel data loading
    process_workers=10,  # For parallel experiment processing
    batch_size=10        # Process 10 experiments per batch
)

# Access data
print(f"Dataset size: {len(dataset)}")
print(f"Gene set size: {len(dataset.gene_set)}")

# First item
data = dataset[0]
experiment = data['experiment']
reference = data['reference']
publication = data['publication']

# Access phenotype data
phenotype = experiment['phenotype']
expression = phenotype['expression']  # SortedDict (linear scale)
log2_ratios = phenotype['expression_log2_ratio']  # SortedDict
log2_se = phenotype['expression_log2_ratio_se']  # SortedDict
n_replicates = phenotype['n_replicates']  # Dict[str, int]
```

### Summary

The Kemmeren2014 dataset implementation is now fully functional with sophisticated gene name resolution, proper dye-swap averaging, multiprocessing support, and comprehensive error handling. Key achievements include:

1. **Complex gene name resolution** - Solved multi-aliasing problem with comprehensive fallback strategies
2. **Multiprocessing optimization** - Added parallel processing that's 6x faster than sequential
3. **Refpool-based error propagation** - Implemented CV-scaled standard deviations for proper noise estimation
4. **Perfect data validation** - Achieved correlation of 1.000 between calculated and original log2 ratios
5. **Production-ready** - Clean logging, robust error handling, and consistent output between processing modes

The dataset successfully processes 1483 deletion mutants with full expression profiles and quality metrics, ready for downstream machine learning applications.

## 2026.01.29 - Replicate Counting Philosophy

### Data-Driven n_replicates Computation

This dataset follows the principle of **computing `n_replicates` from raw data** rather than using paper-reported constants.

**Global constants in code** (lines 42-86 in `kemmeren2014.py`):

```python
N_EXPECTED_BIOLOGICAL_REPLICATES = 2
N_EXPECTED_DYE_SWAP_TECHNICAL_REPLICATES = 2
N_EXPECTED_MAX_REPLICATES_DELETION = 4
```

These constants are **documentation only** - they record what the paper reports about experimental design:

- Quote from paper: "Each mutant strain was grown twice, from two independently inoculated cultures. Each culture was expression-profiled in technical replicate to yield four measurements for each profiling mutant."
- Expected design: 2 biological replicates × 2 dye-swap measurements = 4 total

**Actual implementation**: Code counts actual measurements per gene per deletion:

```python
# Example from processing logic
for gene, values in grouped_values.items():
    n_replicates[gene] = len(values)  # Count actual data points
```

**Why compute rather than use constants?**

1. **QC filtering**: Some samples may fail quality control (actual < 4)
2. **Data is authoritative**: Direct counts from GEO samples are ground truth
3. **Gene-specific variation**: Different genes may have different replicate counts
4. **Reference replicates**: WT reference pool measured in 8-200+ samples (varies by strain/method)

**Reference pool replicates** (lines 82-84):

- `N_EXPECTED_REFPOOL_REPLICATES_BY4741 = None` (computed from data)
- `N_EXPECTED_REFPOOL_REPLICATES_BY4742 = None` (computed from data)
- Actual values: ~28 MATa samples, ~400 MATα samples from WT GEO datasets
- Each deletion's reference phenotype uses computed replicate count from WT sample measurements

This approach enables validation (compare computed vs expected to detect issues) while ensuring data integrity.

Related: [[torchcell.datamodels.schema#20260129---philosophy-around-n_replicates]]

## 2026.09.27 - The channel assignment contradicts GEO's labels (audit finding, not yet fixed)

The Phase 8 audit of [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]] read `label_ch1` and `source_name_ch1` from the dev-tree GEO pickles for GSE42527 and GSE42526. On "-a" arrays GEO puts the reference pool in Cy5 and the deletion in Cy3, the reverse on "-b" arrays; the loader assumes the opposite (kemmeren2014.py line 902, repeated at 951, 1342, 1412 and 1512). Of 2633 deletion arrays the loader reads the reference pool as the deletion on 2594 and the true deletion channel on 39 (mostly "-c" and "-d" titles that fall to the Cy5 default, plus GSM1107979, `yil014c-a-del-1-b`, whose gene name contains "-a"). From the code, not measured in the LMDB: line 2152 negates the ratio against its own comment, so the 2594 arrays end up with log2(deletion / reference pool) by double inversion, which would fit the +0.599 cross-study correlation recorded earlier; the linear `expression` then holds reference-pool intensities and the reference `expression` holds deletion intensities, and the 39 arrays carry the opposite log2 sign and are averaged with their siblings. This assumes GEO's own channel metadata is correct. The tests pin the current behavior; a fix waits for a decision (weekly note).

## 2026.09.28 - Channel assignment from GEO metadata, within-array ratios (fix on branch fix/kemmeren-sameith-channel-strain, not yet served)

The 2026.09.27 finding was re-checked before any change, because the dataset had trained many models and correlates with Sameith 2015 (+0.599 recorded earlier), which had given confidence in it. Everything below was measured on 2026.09.28 on the raw SOFT files under `microarray_kemmeren2014/raw`, the GEO pickles and the dev-tree LMDB (scripts `kemmeren_channel_check.py`, `kemmeren_affected_genes.py`, `lmdb_self_probe.py` and `verify_fixed_loader.py` in the session scratchpad).

### What the sources say

- GEO metadata, all six series (GSE42527, GSE42526, GSE42215, GSE42217, GSE42240, GSE42241; 3061 arrays): `label_ch1` is Cy5 and `label_ch2` is Cy3 on every array; `source_name_ch1` / `source_name_ch2` name the common reference in exactly one channel on 3061 of 3061 arrays ("refpool"; "ref1" on 193 GSE42217 arrays). On the "-a" arrays the reference is ch1 (Cy5) and the deletion ch2 (Cy3); on "-b" the swap. The same holds for the wildtype arrays (`wt-htp07-a` has ch1 refpool, ch2 wt).
- The paper's Extended Experimental Procedures, verbatim: "For the first hybridization the Cy5 (red) labeled cRNA from the deletion mutant is hybridized together with the Cy3 (green) labeled cRNA from the common reference. For the replicate hybridization from the independent cultures, the labels are swapped." By GEO's labels the "first hybridization" so described is the "-b" array, not the "-a" one the loader assumed.
- The deleted gene's own probes settle it without trusting either text. Matching the title's gene symbol to the platform's `GENE_SYMBOL` (705 arrays): median log2(Cy5/Cy3) at the deleted gene is -2.83 where GEO says the deletion is in Cy5 and +2.97 where it says Cy3; the self-probe agrees with GEO's dye on 702 of 705 arrays and with the loader's title rule on 10 of 705.
- The `#VALUE` column definitions are mixed within the series (466 arrays "log2 ratio (Cy3/Cy5)", 461 "(Cy5/Cy3)", 131 + 131 dye-bias-corrected each way, 114 "test/ref", 42 + 42 "-INV_VALUE"), so the old `_validate_log2_ratios`, which compared log2(Cy3/Cy5) with VALUE on the first 20 genes and logged "correlation 1.000", validated a column convention on a subset and never the channel assignment. The earlier sections of this note that cite that correlation as proof of correctness are superseded here.

### What the served records hold

- The loader read the reference pool as the deletion on 2594 of 2633 deletion arrays, and the negation at the old line 2152 turned that back into log2(deletion / refpool). In the dev LMDB (1484 records) the deleted gene's own stored `expression_log2_ratio` is negative on 97.0% of records, below -1 on 85.5%, median -2.48. So the sign the models trained on is right on 1448 records, which is what the cross-study correlation was showing.
- 35 records are wrong: the 39 arrays whose title ends in "-c", "-d", "1-d", "[hs199x] ..." or belongs to `yil014c-a` fell to the "-a" default and took the true deletion channel, so their sign was flipped once. Six genes have every array flipped (crf1, gal3, gat4, rps0b, rps21a, and rcm1 which is not in the LMDB) and 29 have one flipped array cancelling one correct array (abf2, apn2, asf1, cgi121, ctf4, ctf8, dal80, eaf5, kap114, kap123, lsm12, msh6, pfk1, rnh1, rpa14, rrd1, sam3, sfl1, snf2, spo22, swc7, swh1, tup1, uaf30, urn1, yap6, yil014c-a, yml082w, ynr004w, yrr1). Their stored self ratios: kap123 +1.53, sam3 +1.49, rps21a +1.34, pfk1 +1.12, rpa14 +1.00, tup1 +0.40, snf2 +0.21, asf1 not resolved to a probe; 35 of the 115 records with a self ratio above -0.5 are in this set.
- On every record the linear `expression` holds reference-pool intensities and the reference's `expression` holds deletion intensities. No consumer trains on the linear field (`label_name` is `expression_log2_ratio`; `mean_experiment_deduplicate` only averages it through).
- The SE was not a within-array statistic: each array's chosen channel was divided by the mean of the other channel over the gene's arrays, so the SE measured the scatter of one channel's absolute intensity. On GSE42527 (687 genes with two or more arrays) the within-array SE is smaller on 75.7% of gene-by-probe entries, median ratio 0.34 (10th percentile 0.04, 90th 2.5), while the mean log2 barely moves (median absolute difference 0.0045, 95th percentile 0.16). The two-color design cancels the spot only within the array, and [[experiments.019-simb-multimodal.scripts.expression_ceiling_replicate]] reads `expression_log2_ratio_se`, so this matters.

### The fix

- `_channel_columns(gsm)` returns the test and reference `Signal Norm_*` columns from `source_name_ch*` (a name starting with "ref" and not containing "-del") and `label_ch*`; it raises when the reference is not in exactly one channel or the labels are not one Cy5 and one Cy3. The title is not consulted. `_extract_channels_from_gsm_static` reads both channels of each row, `_collect_replicate_pairs_static` gives one (deletion, refpool) pair per array, and `create_expression_experiment(dataset_name, sample_info, replicate_pairs, refpool_n_replicates)` takes log2(deletion / refpool) within each array, then mean, sample SD, SE and variance over the arrays, dropping an array for a gene when either signal is not positive; the linear `expression` is the mean deletion signal and the reference `expression` the mean refpool signal over the same arrays. The negation is gone. The sample classification reads `genotype/variation` from both `characteristics_ch1` and `_ch2` (the deletion is in ch1 on "-b" arrays). The wildtype refpool extraction uses the same helper. The title-rule helpers, the VALUE fallback, the two dead `_calculate_wt_reference_*` methods and `_validate_log2_ratios` are removed.
- `_validate_channel_assignment` replaces the old check: for every deletion array whose gene has a probe, the median log2(deletion / refpool) at that gene's probes must be negative; it logs the fraction and warns below 0.9. On the real pickles with the fixed code (ORF-matched, 1445 genes): 2538 of 2560 arrays, 0.991. Building those 1445 records in memory with the fixed helpers (`verify_fixed_loader.py`, no LMDB written), the deleted gene's own stored log2 ratio is negative on 99.1% (median -2.51, below -1 on 87.3%), against 97.0% and -2.48 in the served build; the five largest are +0.06, +0.06, +0.20, +1.35 and +1.88. `n_replicates` at the deleted gene is 2 on 1115 genes and 1 on 330: two arrays per mutant, since the second probe of each gene is dropped (below).
- Effect on the served graph: every record's SE, variance, linear expression and reference expression change, the 35 log2 signs change, and the mean log2 moves by the Jensen-level amounts above, so all 1484 experiment ids (content hashes) change. That is not a superset and cannot go in as an increment; re-admission waits for the next build (issue `#459`), by decision of 2026.09.28 (fix on the branch, PR open). Not changed here, open for a follow-up: every one of the 6169 ORFs on GPL11232 has two probes ("Each gene is represented twice on the microarray"), and `_extract_probe_to_gene_mapping` plus the row loop keep only the last row per gene, so half of the paper's four measurements per mutant are discarded; and the reference `n_replicates` still counts wildtype arrays with a positive refpool value.
- Tests: [[tests.torchcell.datasets.scerevisiae.test_kemmeren2014_synthetic]] rewritten with GEO-style channel metadata on every fixture array (12 tests).

## 2026.09.29 - Independent re-verification

A read-only Fable 5.1 agent graded eight recorded claims against the mirror, Table S1, GEO (raw SOFT and live) and the dev LMDB: CONFIRMED 4, REFUTED 1, PARTLY 3, UNVERIFIABLE 0. Consolidated in [[datasets.showcase-verification.2026.09.29]]; raw report `notes/assets/verification/2026.09.29/kemmeren2014.md`.

- Refuted: the "VALUE Column Convention" above. GEO's `VALUE` is log2(deletion/refpool) on 2169 of 2633 deletion arrays and log2(refpool/deletion) on 464, and its `#VALUE` label is wrong on 465. No loader reads VALUE; #483.
- The channel metadata (reference in Cy5 on "-a", Cy3 on "-b") is confirmed on 3061 of 3061 arrays, and the deleted gene's own probes read negative under GEO's labels on 2458 of 2479 ORF-matched arrays (0.9915). The old title rule flips 39 arrays (29 "-d", pfk1-del-2-c1, ctf8-del-3-g, lsm12-del-3-e, the six rcm1/rps0b/rps21a arrays, yil014c-a-del-1-b); "-c" arrays are not flipped; 36 served records are affected, and rcm1 (YNL022C) is in the dev LMDB.
- The paper's "four measurements" are 2 arrays x 2 spots per gene; GEO holds at most 2 arrays per mutant. The "2 biological x 2 dye-swap" comment and `N_EXPECTED_MAX_REPLICATES_DELETION = 4` are wrong; #482.
- The reference `n_replicates` (400 for BY4742, 28 for BY4741) counts the WT arrays of two GEO series; the paper compares each mutant to one pool of 200, 200, 20 or 8 arrays. Still passed that way on the PR #460 branch; #484 (before the next build).
- The "Output Statistics" and "Known Limitations" above are stale for the raw file the loader consumes: Table S1 has 1484 unique ORFs, YDR443C appears once (SSN2), and YCR087C-A is in GEO (`lug1-del-3-a/-b`) and in the dev LMDB.
- GEO's `strain` field says BY4742 for the ten MATa deletions; strain must keep coming from Table S1; #483.
- Every record carries 6169 ORFs; 6127 is the graph-mapped subset, not a loader or paper count.

## 2026.10.02 - main() builds the genome with overwrite=False

`main()` constructed `SCerevisiaeGenome(..., overwrite=True)`, an explicit rebuild of the shared `data.db` on every run. It now passes `overwrite=False` (PR #605), which opens the verified shared database, or builds/migrates it once, under the root lock.
