---
id: rgowht2lb9swnv1glzglrbg
title: Build_essentiality_holdout_030
desc: ''
updated: 1790417059190
created: 1790417059190
---

## 2026.09.26 - Held-out singles for the essentiality readout

Two sets, all single-deletion records, disjoint from the pinned val/test: `released`, the 198 Merzbacher 2025 test genes that resolve to 030 singles (31 essential, 167 not; 25 released genes are absent from the build, 4 labels disagree with the build's SGD entry), and `matched`, 250 essential (with a measured entry) + 250 non-essential singles matched on closure-double and triple coverage (coverage AUROC 0.500 after matching, 0.374 before). PPI degree alone scores AUROC 0.591 released and 0.706 matched, the confound baseline. 698 records are excluded from training and served as the `val_ess` loader. Artifact `results/essentiality_holdout_030.json.gz` and its summary.
