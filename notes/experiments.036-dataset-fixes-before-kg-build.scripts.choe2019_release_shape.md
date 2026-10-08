---
id: lpbifg9v5lwgnqhf8tbof0o
title: Choe2019_release_shape
desc: ''
updated: 1791451598933
created: 1791451598933
---

## 2026.10.08 - Measuring the Choe 2019 Release

`experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py` settles
three questions about row 47 of the bacterial schedule by measurement rather than by
reading the abstract, and writes its results beside itself:

- `results/choe2019_release_shape.json` -- the whole measurement
- `results/choe2019_variant_rows.csv` -- all 117 Supplementary Data 2 calls with their
  blocking reasons
- `results/choe2019_expression_gene_resolution.csv` -- the 3,457 RNA-seq gene labels
  against the pinned MG1655 annotation

Every file it reads is sha256-verified against the library mirror's `manifest.json`
before it is opened, so a re-OCR or an upstream change stops the run instead of changing
a number silently.

### What it measures

1. **What the release gives per sample.** The shape of all seven Supplementary Data
   workbooks and the Source Data workbook, the column semantics of the three variant
   tables, how far a day-62 frequency is from a clone genotype, and a per-strain
   writability verdict for every row of Supplementary Table 3.
2. **Whether any existing perturbation leaf can carry a called variant.** Real pydantic
   construction attempts against every candidate leaf for one real variant row (the
   `ilvH` G to T SNV at 82,242, Gly21Cys), with the exact exception each one raises
   recorded verbatim. Also whether MS56 can be stated as a background at all, and whether
   its unenumerated content can be a typed `ProvenanceGap` (it cannot: `alleles` defaults
   to `[]` and a gapped field must be `None`).
3. **Which phenotypes are storable without an evolved genotype.** The Fig. 2c growth
   rates, the Fig. 2f reconstruction ratios, the two Supplementary Fig. 6 Keio
   percentages, and the RNA-seq RPKM against `RNASeqExpressionPhenotype`'s required
   integer `expression_count` (including the Caglar 2017 read-count back-solve, which
   fails here).

### Corrections this version carries over its first run

- Supplementary Data 2 has 117 variant rows, not 118: a stray single-cell row twelve rows
  past the last call was being counted, which also produced a phantom `None` mutation
  type. A row is a variant only when it has BOTH a gene and a position cell.
- Supplementary Data 3's intergenic calls are a bare `-` and Supplementary Data 4's the
  bare word, so the first pass reported 0 intergenic rows for Data 3 when it has 20.
- The Keio panel is Supplementary Fig. 6, not Fig. 5.
- Supplementary Data 3 and 4 each carry a second, header-less 44-row worksheet the first
  pass never opened; it is identical in both workbooks and is a working sheet, not a
  release.

Findings and their consequences for the loader:
[[torchcell.datasets.ecoli.choe2019_growth_rate]]

Usage, from the repo root:

```bash
python experiments/036-dataset-fixes-before-kg-build/scripts/choe2019_release_shape.py
```
