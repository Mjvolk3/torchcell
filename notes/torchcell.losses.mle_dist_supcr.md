---
id: y83hnr2rd0l7lpqs6nn7c57
title: Mle_dist_supcr
desc: ''
updated: 1752609699927
created: 1752609699927
---

## 2026.09.27 - The dist and SupCR terms changed underneath this loss

`MleDistSupCR` builds its `dist` term on `WeightedDistLoss` and its SupCR term on `WeightedSupCRCell` from `multi_dim_nan_tolerant.py`. The fix PR after Phase 6 of [[plan.test-suite-buildout.2026.09.25]] changed both: the soft-sort gradient sign, the pairing of sorted predictions with the theoretical labels, the default dist weight normalization, and the SupCR tie rule (details in [[torchcell.losses.multi_dim_nan_tolerant]]). In this file the `use_buffer=False` branch now applies the scheduled temperature to the unbuffered cell, which it computed and logged but never used. A training run with `lambda_dist > 0` or `lambda_supcr > 0` before and after this change optimizes a different objective; the experiments whose configs actually selected those terms are listed in [[test-campaign.2026.09.25]].
