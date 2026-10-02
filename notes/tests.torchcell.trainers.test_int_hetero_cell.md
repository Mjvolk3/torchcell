---
id: 2bdnrd72g6ir18w0szvfobl
title: Test_int_hetero_cell
desc: ''
updated: 1790872904260
created: 1790872904261
---

## 2026.10.01 - Batch sizing in the hetero trainers (issue #567)

New file. For both `RegressionTask` and `DiffusionRegressionTask`: a batch with vector [0, 0, 1] and `num_graphs = 3` is size 3; the dataloader-profiling step logs exactly `{"val/dataloader_profile_loss": (0.0, 3), "val/dataloader_profile_batch_size": (3.0, 3)}`; and the ladder below `num_graphs` gives [5, 3, 2, 1] for `x` rows, perturbed genes, values, empty.

## 2026.10.01 - Phase 19: exact behavior of both tasks on scripted stand-ins

Fixture: `_Fixed` returns p = [1, 3, -2] times one trainable `scale` (1.0) and `{"z_p": [[3, 4], [0, 0], [6, 8]]}` (row norms 5, 0, 10, mean 5); `_coo` is a collated COO batch of three genotypes, targets y = [2, 5, -1], `num_graphs` 3. `log` is replaced by a recorder of every call; optimizer, backward, clipping and schedulers are recording stand-ins except in three `fast_dev_run` tests. The real 006 inverse is `COOInverseCompose([COOLabelNormalizationTransform])` fitted on [0, 4] (mean 2, sd 2), so it maps v to 2v + 2. Diffusion-only behavior lives in [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]]; shared behavior is parametrized over both classes here.

Expected values:

- Squared error (1 + 4 + 1) / 3 = 2.0; mean log cosh (2 log cosh 1 + log cosh 2) / 3 = 0.7308548; Pearson(p, y) = 15 / sqrt(228) = 0.9933993 (numpy oracle).
- Inverse units: model-unit targets [0, 1.5, -1.5] and predictions p give transformed MSE 3.5 / 3 = 7/6; inverted [4, 8, -2] against originals [2, 5, -1] give MSE 14/3.
- Real `MleDistSupCR(embedding_dim=2)` at epoch 50: total 2.0, buffer weight 0.1 + 0.2 * 50/100 = 0.2, temperature 0.1 ** 0.05 = 0.8912509; all 19 components logged under `train/`.
- Sample buffers (ceiling 5, every 2 epochs): nothing at epoch 0; at epoch 1 a full batch of 3, then `randperm(3)[:2]` (seed 7), then nothing. The test buffer keeps every batch (9 rows).
- `training_step`: backward, (clip at `max_norm`), step, zero_grad; schedule {0: 2} backpropagates 2.0 / 2 = 1.0 and steps on odd `batch_idx`; effective batch size 3 *2* 1 = 6.
- `CosineAnnealingWarmupRestarts` (min 1e-4, max 1e-2, warmup 2) in a real `fast_dev_run`: `learning_rate` logged 1e-4, then 1e-4 + (1e-2 - 1e-4) / 2 = 5.05e-3 after the epoch step.

Findings (source `torchcell/trainers/int_hetero_cell.py`, commit 92b1b05be):

- PointDistGraphReg component logger (lines 302-317) drops multi-element tensors that the generic branch (lines 343-370) logs element-wise. Latent: the real loss returns only Python floats (corrected 2026.10.02).
- A non-tensor inverse output is ignored (lines 456 and 1264), and a batch carrying `phenotype_values_original` on a task with no inverse is accepted: both score model-unit predictions against original-unit targets (MSE 878 in the test).
- `phenotype_types` and `phenotype_sample_indices` are never read: a `fitness`-only batch is scored under `gene_interaction`; a genotype without a label fails as a broadcast error.
- The NaN mask applies to the metrics only; one NaN target makes the batch loss NaN.
- Issue #596 is still open on main: `_get_batch_size` returns the node count for a batch with `gene.x` (12 for 3 graphs), feeding `learning_rate` and `effective_batch_size` logs.
- The DDP world-size factor in `training_step` is unreachable: no Lightning 2.5 strategy has `_strategy_name`.
- `_compute_metrics_safely` swallows two `ValueError` messages silently. Both messages exist in torchmetrics 1.8.2 (`functional/regression/r2.py`, `utilities/data.py`), but MSE, RMSE and Pearson never reach them: an empty epoch logs three NaNs (corrected 2026.10.02).
- A `ReduceLROnPlateau` scheduler (no type, an unknown type, or explicit `type: ReduceLROnPlateau`) is stepped by `on_train_epoch_end` with no metric: a real `fast_dev_run` epoch fails with `TypeError`.
- String-keyed accumulation schedules are sorted lexicographically: {"0": 3, "2": 2, "10": 4} gives 2 at epoch 12, and `__init__` misses the "0" key.
- `batch_size` and `device` hyperparameters are stored and never read.

## 2026.10.02 - Audit corrections, new pins, and the reach of each finding

Corrections (the 2026.10.01 list above is edited in place where it was wrong):

- The real `PointDistGraphReg` returns 14 Python floats and logs `train/graph_reg_loss` itself; the earlier docstring said that key was not logged. The vector-dropping branch is latent. Pinned with the real loss: point 2.0, graph 0.25, total 2.25, normalized shares 8/9 and 1/9.
- The DDP docstring's "a sixth of 192" claim was wrong: 006 batches carry `gene.x`, so the logged effective batch size is node rows times the accumulation factor.
- Swallowed messages (H6): both exist in torchmetrics 1.8.2; only MSE, RMSE and Pearson never reach them.

Rewritten tests: gate means use asymmetric columns [0.1, 0.2, 0.9] and [0.9, 0.6, 0.0] (means 0.4 and 0.5, medians 0.2 and 0.6); the accumulation schedule {0: 1, 2: 2, 10: 4} is checked at the boundary epochs 1, 2, 5, 10, 12 giving 1, 2, 2, 4, 4. Global RNG reseeds now run inside `torch.random.fork_rng()`.

New pins:

- Real `ICLoss(lambda_dist=0.1, lambda_supcr=0.001)` passes through the generic branch with `z_p`; all 17 components logged; loss 2.0 + 0.1 dist + 0.001 supcr.
- `nn.MSELoss` with a model emitting `z_p` fails: `TypeError: MSELoss.forward() takes 3 positional arguments but 4 were given`; without `z_p` it returns 2.0.
- `grad_accumulation_schedule: {0:16}` (no space) loads through OmegaConf as `{"0:16": None}`: the task starts at 1 step and `on_train_epoch_start` raises `ValueError: invalid literal for int() with base 10: '0:16'`. Configs: 006 `hetero_cell_bipartite_dango_gi_cabbi_009`, `_cabbi_010`, `_cabbi_012` (`{0:16}`) and `_mmli_011`, `_mmli_013` (`{0:8}`). Hypothesis, not checked against W&B: runs launched from these files as committed stopped at their first epoch start.
- The plateau crash is parametrized over no type and explicit `type: ReduceLROnPlateau`.

Reach of each finding (from audit 2; reach not re-measured here):

| Finding | Reach |
|---|---|
| H1 PointDistGraphReg drops vectors | Latent (real loss returns floats); 006 `cell_graph_transformer_{cabbi_005,006,008,009,gh_004,mmli_007}` use this loss |
| H2 unit mismatches | Latent: 006 scripts always build `COOInverseCompose`, which returns tensors |
| H3 type and sample indices unread | Latent for 006, assuming one label per genotype (assumption, unchecked) |
| H4 `gene.x` batches sized by node rows | Reached: 006 `hetero_cell_bipartite_dango_gi{,_lazy,_diff_gi}.py`, `hetero_cell_nsa_retry.py`, and the 004/005 dango_gi scripts. PR 587 fixed only the `perturbation_indices_batch` branch (#567); the `gene.x` branch (149, 1022) remains |
| H5 DDP factor unreachable | Reached in DDP configs with a schedule (e.g. 006 `hetero_cell_bipartite_dango_gi_cabbi_014.yaml`), swamped by H4 |
| H6 swallowed messages | Latent |
| H7 plateau stepped with no metric | Crash: 006 `hetero_cell_bipartite_dango_gi_mmli.yaml` and `_test.yaml`; 004 and 005 `hetero_cell_bipartite_dango_gi.yaml` |
| H8 string keys sorted as text | Latent: every config has a single key |
| H9 diffusion train metrics score zeros | Reached: 006 `hetero_cell_bipartite_dango_diff_gi.yaml` |
| `{0:16}` schedule | The five configs above |

## 2026.10.02 - Findings retired by the issue #614 fix

Retired `Finding:` pins, now asserting the corrected contract (exact values and full messages):

- Plateau: `test_plateau_scheduler_steps_on_val_mse_at_validation_end_only` (stepped once on the val MSE 2.0, not during the sanity check, not at the train epoch end) and `test_fast_dev_run_with_a_plateau_scheduler_steps_it_on_the_logged_val_mse` (one real epoch completes; `last_epoch` 1, `best` equals the logged val MSE, rate still 1e-2). `test_a_missing_or_unknown_scheduler_type_is_refused_at_construction` replaces the silent-plateau pin.
- Schedules: `test_the_five_corrected_006_configs_load_an_integer_schedule` (16 or 8 steps from epoch 0, from the loaded mapping and from the string-keyed `wandb.config` copy), `test_digit_string_keys_are_integer_epochs_in_numeric_order` ({"0": 3, "2": 2, "10": 4} gives 3, 2, 2, 4, 4 at epochs 1, 2, 5, 10, 12), `test_a_schedule_that_is_not_epochs_to_step_counts_is_refused_at_construction` (ten refusals, `"0:16"` first).
- Batch size: `test_batch_size_is_num_graphs_and_a_batch_without_it_is_refused` (a real `Batch` of 12 `gene.x` rows is 3; the perturbation and COO batches are 3), `test_every_log_of_a_gene_x_batch_carries_the_genotype_count` (all four logs `batch_size=3`, effective batch size 6, not 24), `test_effective_batch_size_counts_the_trainer_world_size` (DDP over two devices: 3 x 2 x 2 = 12).
- Losses: `test_unnamed_losses_receive_only_predictions_and_targets`, `test_plain_mse_loss_gets_two_arguments_with_or_without_z_p`, `test_a_z_p_loss_on_a_model_without_z_p_is_refused_by_name`, `test_point_dist_graph_reg_gets_representations_and_logs_every_component` (`vec_0`, `vec_1` now logged).
- Units: `test_unit_mismatches_are_refused_by_name` (`TypeError` for a list inverse; `ValueError` before the model for originals without an inverse).
- Metrics: `test_log_metrics_logs_and_resets_and_an_empty_epoch_logs_nan`, `test_a_metric_error_propagates_from_the_epoch_end`.
- Hyperparameters: `test_batch_size_and_device_are_accepted_but_neither_saved_nor_read` (the eleven saved names listed).

Still pinned: `test_coo_layout_is_read_positionally_not_by_sample_or_type` (a fitness-only batch is scored as gene interaction). It now also pins the real collated layout (`phenotype_types` one list per genotype, `phenotype_sample_indices` [0, 0]) and the new refusal of a batch whose genotype count differs from the prediction rows. The buffer test runs the diffusion task on validation only; its train epoch end moved to the diffusion tests.
