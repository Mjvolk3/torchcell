---
id: 5flxz9obibx2pcss57tnt7c
title: Expression_morphology_covariation
desc: ''
updated: 1791149333722
created: 1791149333723
---

## 2026.10.04 - Does the knockout transcriptome carry morphology, the third side of the triangle

`experiments/019-simb-multimodal/scripts/expression_morphology_covariation.py`, the same
estimator, folds, penalties, seed and reliability threshold as
[[experiments.019-simb-multimodal.scripts.proteome_morphology_covariation]], with Kemmeren
2014 expression in the proteome's place. 1,438 single deletions measured in both panels,
6,169 expression genes, 278 CalMorph features. Results in
`results/expression_morphology_covariation.json`. Run on the Mac from the local Kemmeren
LMDB and `scmd_ohya2005/raw` tables (`--morph-dir`); no figure yet.

| read | statistic | value |
|---|---|---|
| observed expression to each CalMorph feature | median held-out Pearson, 278 features | 0.215 (IQR 0.12 to 0.32, max 0.50) |
| the 116 features that move under deletion (reliability at least 0.5) | median | 0.324 (62 percent above 0.3) |
| the 162 quiet features | median | 0.155 |
| strain-permuted null | median | 0.008 (max 0.08) |
| first principal component projected out of both sides, all features | median | 0.127 |
| first principal component projected out, moving features | median | 0.166 (max 0.42) |
| observed morphology to each expression gene | median over 6,169 genes | 0.087 (max 0.38), null 0.007 |
| response magnitude, expression against morphology L2 norms | Spearman over 1,438 strains | 0.246, p 3e-21 |
| first PCs of the two panels | absolute Pearson | 0.21 |

The three sides are not on the same strains: this side and proteome against expression
(1,349) sit on the Kemmeren deletions, proteome against morphology on 4,314. Not measured
here: any of the three sides restricted to one common strain set, and whether any of it
reaches a genotype-only trunk.
