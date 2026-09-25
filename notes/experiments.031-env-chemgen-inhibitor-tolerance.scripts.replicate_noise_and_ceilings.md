---
id: s4wp9ih125osmvkbe8t2b2k
title: Replicate_noise_and_ceilings
desc: ''
updated: 1790379796270
created: 1790379796270
---

## 2026.09.25 - Two views of noise per condition, joined on the served compound name

The served view: per condition, the replicate count, the served SE, and the reliability
index `1 - mean(SE^2) / var(response)` over genes, with `sqrt(rel)` as the ceiling on the
Pearson correlation between a measurement and the noise-free truth and `rel` as the
ceiling between two independent measurements. The empirical view: pairwise Spearman and
bottom-5% hit Jaccard between the raw replicates the loaders consumed (Vanacloig's three
batches as log2 CPM ratios against their own batch controls; Hillenmeyer's replicate
arrays of one condition label). Raw labels are mapped to served compound names with
`resolved_compound`, the loaders' own resolver, so the two views join.

Outputs under `results/`: `noise_summary.csv` (one row per dataset),
`condition_noise_<dataset>.csv` (one row per condition, both views),
`raw_replicate_agreement.csv` (one row per raw label). Figures `noise_distributions`,
`reliability_index_vs_replicates`, `vanacloig_ceilings` in
`notes/assets/images/031-env-chemgen-inhibitor-tolerance/` (`--stable` writes the
un-timestamped names the note and the notes-tex document reference).

Finding: the served index tracks Vanacloig's raw replicate agreement at rank 0.87, but
overstates Hillenmeyer's (index median 0.92 for HOM against raw replicate Spearman 0.39,
rank agreement 0.40). Discussed in [[experiments.031-env-chemgen-inhibitor-tolerance]].
