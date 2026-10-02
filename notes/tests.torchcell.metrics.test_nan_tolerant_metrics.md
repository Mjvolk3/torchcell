---
id: 71e4hrbsagm85y013fffp88
title: Test_nan_tolerant_metrics
desc: ''
updated: 1790916511624
created: 1790916511624
---

## 2026.10.01 - Exact values and findings for the NaN-tolerant regression metrics

`tests/torchcell/metrics/test_nan_tolerant_metrics.py` pins `torchcell/metrics/nan_tolerant_metrics.py` (0 percent to 98 percent line coverage in Phase 19 of [[test-campaign.2026.09.25]]). Callers (`fit_int_cell_diffpool_dense_regression`, `fit_int_hetero_gnn_pool_binary_classification`, `fit_int_cell_sagpool_regression`) use the default `num_outputs=1` with 1-D predictions and targets, so those paths are checked against independent oracles: torchmetrics `MeanSquaredError`, `MeanAbsoluteError`, `PearsonCorrCoef`, `R2Score` and `scipy.stats.pearsonr` / `spearmanr` on the hand-masked rows. Every metric masks an element when EITHER the prediction or the target is NaN.

Fixtures. Accumulation: four `update` calls with 2, 0, 3 and 1 valid rows; squared errors sum to 19.25 over 6 rows (MSE 3.2083333, RMSE 1.7911822), absolute errors to 9.5 (MAE 1.5833333); the mean of per-batch MSEs would be 3.0972222, so pooled accumulation is distinguished from it. Pearson: four batches with 2, 0, 3, 1 valid rows, compared with scipy on the six concatenated pairs (sum_x 9.5, sum_y 10). `_final_aggregation`: hand-built shard states (mean, M2, co-moment, n) for x = [1, 2, 3 | 10, 12 | -4] give mean 5.6, M2_x 101.2, M2_y 45.2, C 65.2 for the first two shards and match numpy for all three.

Findings (each test docstring starts "Finding:"):

- Multi-output MSE, RMSE, MAE (lines 25 to 27, 47 to 49): `preds[valid_mask]` flattens the 2-D input, so `sum(dim=0)` is a scalar over all columns and the count is the ROW count. On P2 = [[1, 2], [3, nan], [5, 6]], T2 = [[0, 0], [nan, 0], [1, 1]] MSE is [46/3, 46/3] instead of [8.5, 14.5].
- Multi-output Pearson (lines 336 to 350): both columns receive pooled sums and `compute` raises `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. Spearman with `num_outputs=2` inherits it.
- Pearson raw-moment cancellation (lines 340 to 368): a constant non-representable target [1.3, 1.3, 1.3] leaves var_y = 1.19e-07 and scores 0.0 instead of NaN; an offset of 11000 turns a true correlation of 0.5 into -1.0. The fitness-scale effect size is in the 2026.10.02 section (reproducible from the test file).
- Spearman ties (lines 434 to 435): `argsort(argsort(x))` gives ties distinct ranks; [1, 1, 2, 3] vs [1, 2, 3, 4] scores 1.0 (scipy 0.9486833) and a constant target scores the rank correlation of the predictions with row position (corrected 2026.10.02).
- Spearman `num_outputs` guard (line 396) uses `and`, so `num_outputs=0` is accepted.
- R2 accumulation (lines 543 to 554): each batch is centered on its own target mean; two batches give 0.5 where the concatenation gives 0.9676375. Multi-output R2 pools columns (-45 in both instead of [-33, -57]).
- `_final_aggregation` (lines 202 to 220) is algebraically exact but mutates the caller's `vars_x`, `vars_y`, `corrs_xy` in place; a second call double counts.
- `_nan_tolerant_pearson_update` (lines 261 to 263) omits the prior shard's mean shift: after two batches M2_x is 62.32 instead of 101.2. It and `_final_aggregation` have no caller in `torchcell/` or `experiments/`.

Left uncovered: `_track_device`'s device move (line 288, needs a second device) and Spearman's `except (RuntimeError, IndexError)` branch (lines 429 to 431), unreachable because `num_samples > 0` implies a non-empty buffer.

## 2026.10.02 - Audit 1 corrections, new pins, and reach

Corrections to the 2026.10.01 section:

- Spearman ties: "a constant target scores 1.0" was too general. Tied values get ranks in row order, so the score of tied data depends on row order. With a constant target the score is the rank correlation of the predictions with row position: preds [1, 2, 3] give 1.0 and [3, 2, 1] give -1.0 (scipy nan). With tied predictions, x [1, 1, 2, 3] gives 1.0 against y [1, 2, 3, 4] and 0.8 against y [2, 1, 3, 4]; scipy gives 0.9486833 for both.
- The fitness-scale figure given on 2026.10.01 is withdrawn: it came from a scratch probe whose generator was shared across three scales and is not reproducible from the tests. It is replaced by `test_pearson_fitness_scale_stream_error_against_scipy`: `torch.Generator().manual_seed(0)`, y = 1 + 0.01 *randn(20000), x = y + 0.01* randn(20000), batches of 64. NaNTolerantPearsonCorrCoef gives 0.7097331, scipy 0.7092065, torchmetrics `PearsonCorrCoef` 0.7092068: raw-moment error about 5.3e-4, centered error about 3e-7.

New pins:

- Finding: constant predictions (a model that predicts the mean) give NaN or a number depending on the sign of the float32 residue in `sum_x2/n - mean_x^2` (lines 359 to 368): 1.3 repeated 7 times gives var_x < 0 and NaN; 1.3 repeated 3 times gives 0.0; 0.3 repeated 10 times gives a small nonzero value (0.000555 observed). scipy returns nan in every case.
- Only NaN is masked; +-inf passes: MSE on p [inf, 1] counts 2 rows and returns inf; Pearson with an inf prediction returns NaN.
- `n_samples` is float32, so counts saturate at 2^24 = 16777216.

Reach (from audit 1, verified here):

- Five trainers import these modules: `fit_int_cell_diffpool_dense_regression`, `fit_int_cell_sagpool_regression`, `fit_int_hetero_gnn_pool_binary_classification`, `fit_int_cell_gin_diffpool_dense_binary` (classification metrics only) and `fit_int_gat_diffpool_inception_regression`.
- Finding: `fit_int_gat_diffpool_inception_regression` imports `NaNTolerantPearsonCorrCoef` and `NaNTolerantSpearmanCorrCoef` from `torchcell.losses.multi_dim_nan_tolerant` (lines 25 to 29), where they do not exist, so the module raises ImportError and the trainer cannot load (pinned by `test_gat_diffpool_inception_trainer_cannot_import_its_metrics`; also in the known-failure list of `tests/torchcell/test_import_all.py`).
- Every affected logged metric belongs to experiment 003 runs from 2024-11 to 2025-01; none is in the manuscript or notes-tex.
- Latent (no caller exercises them): findings 1 (multi-output MSE/RMSE/MAE), 2 (multi-output Pearson/Spearman crash), 5 (Spearman `num_outputs` guard), 7 (`_final_aggregation` mutation) and 8 (`_nan_tolerant_pearson_update`). Callers use `num_outputs=1` with 1-D slices.
- Hypothesis (untested): with `dist_sync_on_step=True` and `sync_on_compute=False`, each DDP rank computes its own epoch value; if Lightning then averages per-rank Pearson values, the logged number is not the global Pearson.
