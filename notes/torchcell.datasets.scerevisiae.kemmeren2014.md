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
# Sequential processing (original)
dataset = MicroarrayKemmeren2014Dataset(
    root="data/torchcell/microarray_kemmeren2014",
    io_workers=0,
    process_workers=0  # Sequential processing
)

# Parallel processing (faster, recommended)
dataset = MicroarrayKemmeren2014Dataset(
    root="data/torchcell/microarray_kemmeren2014",
    io_workers=10,       # For parallel data loading
    process_workers=10,   # For parallel experiment processing
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
```

### Summary

The Kemmeren2014 dataset implementation is now fully functional with sophisticated gene name resolution, proper dye-swap averaging, multiprocessing support, and comprehensive error handling. Key achievements include:

1. **Complex gene name resolution** - Solved multi-aliasing problem with comprehensive fallback strategies
2. **Multiprocessing optimization** - Added parallel processing that's 6x faster than sequential
3. **Refpool-based error propagation** - Implemented CV-scaled standard deviations for proper noise estimation
4. **Perfect data validation** - Achieved correlation of 1.000 between calculated and original log2 ratios
5. **Production-ready** - Clean logging, robust error handling, and consistent output between processing modes

The dataset successfully processes 1483 deletion mutants with full expression profiles and quality metrics, ready for downstream machine learning applications.
