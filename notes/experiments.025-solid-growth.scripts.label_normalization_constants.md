---
id: ul3smtjkjw09dtx0f42a86x
title: Label_normalization_constants
desc: ''
updated: 1788998990652
created: 1788998990652
---

## 2026.09.09 - Which Records Set the Label Z-Score, and What That Can Move

The trainer standardizes `gene_interaction` once, before training, as `(y - mean) / sd`
with the two constants computed over the population `transforms.fit_on` names: `subset`
is every record of the arm (what 010 did over all 376,732 triples), `train` is the
arm's pinned training split. Pearson is invariant to an affine map of the labels, so
the choice cannot move the reported correlation; it rescales the point loss relative to
the Sinkhorn and graph terms by the ratio of the two sds squared.

Measured on the 025 build (`results/label_normalization_constants.json`):

| population | n | mean | sd |
|---|---|---|---|
| subset (all S0 triples) | 376,732 | -0.008024324 | 0.063263549 |
| train, R split | 301,386 | -0.007688739 | 0.063186255 |
| train, Q split | 301,236 | -0.007843739 | 0.063776418 |
| val, Q split | 37,705 | -0.009232021 | 0.062545118 |
| test, Q split | 37,791 | -0.008258833 | 0.059760599 |

The subset and R-train rows are the constants the 010 additive-baselines document
quotes from the 010 training logs, to nine digits. Point-loss scale ratio
`(sd_subset / sd_train)^2`: 1.00245 for R, 0.98398 for Q.

Arm `cgt_s0_q_kl_004` (job 1609) is the first run with `fit_on: train`; the
replication arm keeps `fit_on: subset` because reproducing 010 requires 010's
constants.

```bash
python experiments/025-solid-growth/scripts/label_normalization_constants.py
```
