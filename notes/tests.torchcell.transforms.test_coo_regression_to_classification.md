---
id: 6a9s9i8xzkspdue7nuwftt7
title: Test_coo_regression_to_classification
desc: ''
updated: 1790759605212
created: 1790759605212
---

## 2026.09.30 - Phase 12: normalization, bin edges, labels, inverses in closed form

Fifteen to forty-eight tests, 75 to 99 percent. Normalization drops infinities and uses the population std (sqrt(2/3)); the robust strategy maps [0, 1, 4] to [-0.5, 0, 1.5] and back; `fit_indices` resolves through the `index` column. Equal width gives edges [0..4], equal frequency [0, 2, 4, 6, 8] with counts [2, 2, 2, 3], auto int(4/sqrt(2)) = 2 bins. One-hot is left-closed and clamped with the right edge folded into the last bin; ordinal uses strict crossings; soft labels match the Gaussian values 0.6224593/0.3775407 and clamped 0.7310586/0.2689414. The inverse draws from the seeded stream (seed 42: 0.8822692632675171, 0.9150039553642273; seed 0: 0.4962566), NaN decodes to NaN, `COOInverseCompose` runs last-first.

Findings: an unknown normalization strategy is stored unchecked (line 87) and fails on first use (112, 132); auto binning's metadata says `"equal_width"` (403); ordinal's strict `>` (291) puts exactly 1 in class 0 while one-hot puts it in bin 1; when the binned label is not first the new type indices point past the rewritten type list and unconfigured labels are dropped (560); an unknown `label_type` raises `UnboundLocalError` (539-553); ordinal forward emits 3 values but names 4 bin types (579 versus 556); soft decoding softmaxes values that are already probabilities (688; 4.480023 where a probability-weighted mean gives 4.357); a narrow sigma underflows the soft label to zeros (320).
