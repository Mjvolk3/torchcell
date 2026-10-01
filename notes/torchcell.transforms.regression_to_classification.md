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

## 2026.10.01 - Real training targets changed by the soft-label fix; zero-width bins refused

The softmax change alters real training targets. Measured by `experiments/003-fit-int/scripts/soft_label_underflow_audit.py` ([[experiments.003-fit-int.scripts.soft_label_underflow_audit]]) on `003-fit-int/001-small-build/processed/label_df.parquet`. The settings match the soft configs: 32 bins, equal_frequency, sigma scale 3, with and without the minmax normalizer, and the counts are the same either way.

- `gene_interaction`: 9,422 of 1,023,196 labeled rows (0.92%) were all-zero under the old rule; 7,034 in bin 0 and 2,388 in bin 31, the two wide tail bins, so they are the strongest interactions. Under the new rule those rows sum to 1 (within 1e-7) with a peak probability of at least 0.5143.
- `fitness`: 0 of 1,340,841.
- Affected configs (soft, 32 bins, equal_frequency, binning on), each now carrying a YAML comment at its transform block: `experiments/003-fit-int/conf/hetero_gnn_pool.yaml`, `hetero_gnn_pool_reg_categorical_entropy.yaml`, `hetero_gnn_pool_reg_categorical_entropy-sweep_5e4.yaml`, `hetero_gnn_pool-sweep_5e4_02.yaml`, `hetero_gnn_pool-sweep_5e4_03.yaml`. The other soft configs of 003 set `bin: false`.

How the historical runs treated those rows, measured by the same script on a two-row batch whose second row has an all-zero task-1 target:

- `CombinedCELoss` (loss_type "ce"): the zero row counts as valid in the mask, its cross entropy is 0 and the gradient on its logits is exactly [0, 0, 0, 0]. It also divides the task mean by one more row (task-1 mean 1.1815991 = row-0 CE / 2). The strongest gene interactions were silently untrained, not NaN.
- `MseCategoricalEntropyRegLoss` (loss_type "mse_entropy_reg"): the MSE part reads the continuous target and is unaffected. The entropy regularizer adds 1e-10 and renormalizes, so the zero row enters the KL distances as the uniform distribution [0.25, 0.25, 0.25, 0.25] at 4 classes; the total stays finite. Reading the code (not measured), the row adds no tightness term, because every class probability is 0.

No switch reproduces the zero rows: a row of zeros is not a target.

A zero-width bin (duplicate edges) makes sigma 0 and every soft row NaN (zero rows before the fix). `LabelBinningTransform` now refuses a soft label whose minimum bin width is not positive: `Soft labels for label 'fitness' need bins of positive width; the minimum bin width is 0.0 (duplicate bin edges)`. Categorical and ordinal labels keep accepting an empty bin. The 003 data has 0 duplicate edges at 32 bins for both labels, with and without minmax, so no committed config is refused.
