---
id: nae3vewr3hy4los94gtiapm
title: Regression_to_classification
desc: ''
updated: 1790872416287
created: 1790872416287
---

## 2026.10.01 - Validation, underflow-safe soft labels, and the soft inverse docstring (#529)

Four defects pinned by Phase 14 of the test campaign are fixed, following the rules PR #558 set for the COO sibling.

- `LabelNormalizationTransform` refuses a strategy outside `NORMALIZATION_STRATEGIES` (minmax, robust, standard) in its constructor: `Unknown normalization strategy 'zscore' for label 'fitness'; valid strategies: minmax, robust, standard`. Before, only `normalize`/`denormalize` refused it, at the first batch.
- `compute_soft_labels` is a softmax of the log-weights `-0.5 * (d / sigma) ** 2`. A row whose raw weights all underflowed used to stay `[0, 0]`; it now puts all of its mass on the nearest center. Rows that did not underflow are the same distribution as before, up to float32 rounding.
- `resolve_label_type` refuses a `label_type` outside `LABEL_TYPES` (categorical, ordinal, soft) by name: `Unknown label_type 'bogus' for label 'fitness'; valid label types: categorical, ordinal, soft`. The constructor calls it, and `forward` and `inverse` call it before writing anything, so an unknown type is no longer silently unbinned forward and NaN back.
- `inverse` converts a list prediction to a tensor before reading its device (it used to raise `AttributeError`).
- The `inverse` docstring now describes each type: categorical and ordinal draw uniformly in the selected bin, soft is a deterministic windowed mean that returns the argmax center within two bins of an edge. The callers pass logits and read a point estimate, so the code was kept.

Configs: every committed `label_type` is categorical, ordinal or soft, so no config changes behavior except a soft row that used to underflow to zeros. Tests: `test_an_unknown_strategy_is_refused_at_construction`, `test_soft_labels_that_would_underflow_put_all_mass_on_the_nearest_center`, `test_soft_inverse_is_a_windowed_expectation_only_away_from_the_edges`, `test_an_unknown_label_type_is_refused_by_name`, `test_inverse_converts_a_list_prediction_before_reading_its_device` in [[tests.torchcell.transforms.test_regression_to_classification]].
