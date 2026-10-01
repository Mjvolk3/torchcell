---
id: 7av4cczuditlcze5dsespc3
title: Coo_regression_to_classification
desc: ''
updated: 1790814963329
created: 1790814963329
---

## 2026.09.30 - Fix: one edge convention, type mapping, validation, soft decode (issue #521)

Previous behavior, pinned by Phase 12 of the test campaign: ordinal labels compared with a strict `>`, so a value exactly on interior edge 1 was class 0 while the one-hot path put it in bin 1; when the binned label was not the first type, the new type indices were `label_idx * num_bins + j` and pointed past the rewritten type list, and unconfigured labels were dropped from the COO; an unknown `label_type` raised `UnboundLocalError` on first use; ordinal emitted 3 values per sample but named 4 bin types; soft decoding applied a softmax to values that are already probabilities; a narrow sigma underflowed the soft label to an all-zero row; auto binning recorded `"equal_width"` in its metadata; an unknown normalization strategy was stored unchecked and failed only on first use.

Now:

- Both label paths use the left-closed convention of `torch.bucketize(..., right=True) - 1` and `np.digitize`, bin i = [edge_i, edge_i+1). Ordinal compares with `>=`, so its count of ones equals the one-hot bin index on and between every edge.
- `COOLabelBinningTransform.forward` rewrites the type list in its own order (a configured label becomes `<label>_bin_0 .. <label>_bin_{w-1}`, any other label passes through), builds an explicit old type index to new type indices map, and replaces each COO entry in place. `label_width` gives `w`: `num_bins` for categorical and soft, `num_bins - 1` thresholds for ordinal. The type list depends only on the input type list, so a configured label present with no entries is still renamed to its bins.
- `inverse` mirrors it: bins of a configured label collapse back to the label at the position of its first bin, other types pass through with their entries and a remapped index.
- `label_type` and the normalization `strategy` are validated at construction with a `ValueError` naming the bad value, the label and the valid names.
- Soft labels are a softmax of the log-weights `-0.5 * (d / sigma) ** 2`, so a narrow sigma still yields a distribution (the midpoint gives exactly [0.5, 0.5] at sigma 0.01).
- Soft decoding weights the bin centers by the probabilities directly over the 5-bin window (the pinned case now decodes to 3.05 / 0.7 = 4.357143). The short-window snap to the argmax center is unchanged.
- `AutoBinStrategy` records `strategy: "auto"`.

Consumers: every committed experiment config (005, 006, 009, 010, 011, 019, 025) uses only `COOLabelNormalizationTransform` with `strategy: standard`, plus `COOInverseCompose`; none configures a `binning` block, so no configured path changes. Evidence: `tests/torchcell/transforms/test_coo_regression_to_classification.py`, tests `test_ordinal_labels_count_left_closed_crossings_of_interior_edges`, `test_binning_forward_expands_each_value_into_its_bins`, `test_binning_inverse_passes_unconfigured_labels_through`, `test_unknown_label_type_is_refused_at_construction`, `test_ordinal_round_trip_counts_crossings_under_the_given_seed`, `test_soft_inverse_averages_a_five_bin_window_of_probabilities`, `test_soft_labels_stay_a_distribution_when_sigma_is_narrow`, `test_auto_bins_truncate_range_over_std`, `test_unknown_normalization_strategy_is_refused_at_construction`.

The near-duplicate `torchcell/transforms/regression_to_classification_coo.py` (no non-test consumer) still carries the same defects; it was not changed here.
