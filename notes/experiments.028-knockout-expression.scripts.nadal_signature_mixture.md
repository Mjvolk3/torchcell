---
id: 0ivmxs0t6f97wmj1jmdqtfu
title: Nadal_signature_mixture
desc: ''
updated: 1789186810279
created: 1789186810279
---

## 2026.09.11 - Kemmeren signatures scored in single cells

Script: `experiments/028-knockout-expression/scripts/nadal_signature_mixture.R` (`r-seurat`
env); input `recomputed/kemmeren_signatures.tsv` (built in the session from the Kemmeren
LMDB: for every shared deletion, its reporters with |log2| > 1, resolved to the Seurat row
names); output `recomputed/signature_mixture.tsv`; summary in
`results/nadal_readthrough_and_signature_tests.json`.

A purity test that never uses the deleted gene. For 240 shared deletions with >= 20 strong
Kemmeren reporters and >= 20 cells: per-cell score = mean log-normalized expression over
the up reporters minus over the down reporters; AUROC of the genotype's cells against the
500 WT cells.

- AUROC median 0.53 (IQR 0.50-0.61); 8 of 240 above 0.8, 44 above 0.65.
- Fraction of a genotype's cells above the WT 95th percentile: median 7.5% (chance 5%).
- The 8 that separate are strong regulators: `YLR180W` (sam1) 0.86, `YHR206W` (skn7) 0.85,
  `YLR176C` (rfx1) 0.83, `YHL027W` (rim101) 0.83, `YDR448W` (ada2) 0.83, `YBR289W` (snf5)
  0.82, `YGR122W` 0.81, `YJL176C` (swi3) 0.80. In those, 33-68% of the labeled cells sit
  above the WT 95th percentile and the rest inside the WT range: the response is present
  in roughly half the cells, which matches the excess-zero purity of 0.36-0.40.

For the other ~230 genotypes the Kemmeren response is not detectable in the cells at all.
Whether that is the same impurity, a response too small for ~1,000-UMI cells, or a response
absent under YPD and pooled growth is not separable here; the split-half test
([[experiments.028-knockout-expression.scripts.nadal_split_half_signal]]) shows it is not
a Kemmeren-specific failure.
