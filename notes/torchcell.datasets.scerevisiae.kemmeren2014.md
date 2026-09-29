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

## 2026.09.29 - Independent re-verification

A read-only Fable 5.1 agent graded eight recorded claims against the mirror, Table S1, GEO (raw SOFT and live) and the dev LMDB: CONFIRMED 4, REFUTED 1, PARTLY 3, UNVERIFIABLE 0. Consolidated in [[datasets.showcase-verification.2026.09.29]]; raw report `notes/assets/verification/2026.09.29/kemmeren2014.md`.

- Refuted: the "VALUE Column Convention" above. GEO's `VALUE` is log2(deletion/refpool) on 2169 of 2633 deletion arrays and log2(refpool/deletion) on 464, and its `#VALUE` label is wrong on 465. No loader reads VALUE; #483.
- The channel metadata (reference in Cy5 on "-a", Cy3 on "-b") is confirmed on 3061 of 3061 arrays, and the deleted gene's own probes read negative under GEO's labels on 2458 of 2479 ORF-matched arrays (0.9915). The old title rule flips 39 arrays (29 "-d", pfk1-del-2-c1, ctf8-del-3-g, lsm12-del-3-e, the six rcm1/rps0b/rps21a arrays, yil014c-a-del-1-b); "-c" arrays are not flipped; 36 served records are affected, and rcm1 (YNL022C) is in the dev LMDB.
- The paper's "four measurements" are 2 arrays x 2 spots per gene; GEO holds at most 2 arrays per mutant. The "2 biological x 2 dye-swap" comment and `N_EXPECTED_MAX_REPLICATES_DELETION = 4` are wrong; #482.
- The reference `n_replicates` (400 for BY4742, 28 for BY4741) counts the WT arrays of two GEO series; the paper compares each mutant to one pool of 200, 200, 20 or 8 arrays. Still passed that way on the PR #460 branch; #484 (before the next build).
- The "Output Statistics" and "Known Limitations" above are stale for the raw file the loader consumes: Table S1 has 1484 unique ORFs, YDR443C appears once (SSN2), and YCR087C-A is in GEO (`lug1-del-3-a/-b`) and in the dev LMDB.
- GEO's `strain` field says BY4742 for the ten MATa deletions; strain must keep coming from Table S1; #483.
- Every record carries 6169 ORFs; 6127 is the graph-mapped subset, not a loader or paper count.
