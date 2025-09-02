---
id: hpef4a5ltjkhabm76bw5095
title: Ohya2005
desc: ''
updated: 1755124027050
created: 1755123685272
---

## Ohya2005 CalMorph Morphology Dataset

### Overview

The Ohya2005 dataset contains morphological measurements from the CalMorph software for yeast cell imaging analysis. This dataset was published by Suzuki et al. (2018, BMC Genomics) after reanalyzing images originally from Ohya et al. (2005, PNAS) following quality control.

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

1. Ohya Y, et al. (2005) High-dimensional and large-scale phenotyping of yeast mutants. PNAS 102(52):19015-20. [PubMed: 16496002](https://pubmed.ncbi.nlm.nih.gov/16496002/)

2. Suzuki G, et al. (2018) Global study of holistic morphological effectors in the budding yeast Saccharomyces cerevisiae. BMC Genomics 19:149.

### Integration Status

- [x] Dataset class implemented (`SmfOhya2005Dataset`)
- [x] Adapter created (`SmfOhya2005Adapter`)
- [x] Schema updated with `CalMorphPhenotype` and related classes
- [x] BioCypher configuration updated
- [x] Dataset registered in adapter map
- [ ] Fallback download URLs to be added when shared storage available
- [ ] Complete CALMORPH_LABELS dictionary population from SI_1.mmd
