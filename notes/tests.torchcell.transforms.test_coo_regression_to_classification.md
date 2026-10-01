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

## 2026.09.30 - Issue #521 findings retired

All eight findings are fixed in the source and the tests now assert the corrected contracts, 48 to 49 tests. Ordinal and one-hot agree on every edge (1 -> [1, 0, 0], 3 -> [1, 1, 1], and the count of ones equals the one-hot argmax on [0, 0.999, 1, 1.5, 2, 2.999, 3, 4]). A two-label batch with the binned label second gives the exact COO values [0.9, 1, 0, 0, 0, 0, 0, 1, 0], type indices [0, 1, 2, 3, 4, 1, 2, 3, 4], sample indices [7, 7, 7, 7, 7, 9, 9, 9, 9] on types ["fitness", "gene_interaction_bin_0..3"]; the new `test_binning_inverse_passes_unconfigured_labels_through` decodes it back to [0.9, 0.8822693, 2.9150040] on ["fitness", "gene_interaction"]. An unknown `label_type` and an unknown normalization strategy raise named `ValueError`s at construction. Ordinal names 3 types on indices [0, 1, 2]. Soft decoding gives 3.05 / 0.7 = 4.357143 and 1.5. A sigma of 0.01 gives the exact rows [1, 0], [1, 0], [0.5, 0.5]. Auto binning records "auto". The configured label present with no entries now has its types rewritten to its bins. Run against the old source, ten tests fail, covering all eight items.

Review notes applied: the unknown-strategy test now uses a frame without `fitness`, so a label-first check order would fail the match; the inverse pass-through test adds the binned-label-first ordering (forward type indices [0, 1, 2, 3, 4, 0, 1, 2, 3], inverse values [0.8822693, 2.9150040, 0.9] on types [0, 0, 1], samples [7, 9, 7]); `test_binning_forward_leaves_data_without_the_label_unchanged` is renamed `test_binning_forward_rewrites_the_type_list_only_for_configured_labels`.
