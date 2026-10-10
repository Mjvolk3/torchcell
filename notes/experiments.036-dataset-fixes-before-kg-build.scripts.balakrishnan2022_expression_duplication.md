---
id: 6wk4a059r3z8apgn5ignv79
title: Balakrishnan2022_expression_duplication
desc: ''
updated: 1791625327841
created: 1791625327841
---

## 2026.10.10 - Duplication by content against served E. coli RNA-seq stores

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/balakrishnan2022_expression_duplication.py`
Results: `results/balakrishnan2022_expression_duplication.json`

Every Balakrishnan 2022 record (28) against every record of the three served E. coli
RNA-seq dev stores, both sides renormalized to a fraction over the shared b-numbers. 0
exact profile matches in 13,468 pairs; the highest log10 Pearson r is 0.918 (PRECISE-1K)
and 0.901 (Public K-12), below the release's own within-condition replicate r of 0.989
to 0.993. Caglar 2017 shares no locus (REL606 namespace). Loader note:
[[torchcell.datasets.ecoli.balakrishnan2022]].
