---
id: j5vblbpgzke16upbl638lug
title: Dcell_regression
desc: ''
updated: 1790812863404
created: 1790812863404
---

## 2026.09.30 - Loss call order, root metrics, and epoch-end fixes (issue #516)

Previous behavior and the fix:

- Loss: the steps called `self.loss(y_hat, y, dcell.parameters())`, the `(outputs, target, weights)` order of the deprecated loss, while `__init__` builds `torchcell.losses.DCellLoss(predictions, outputs, target)`, so every train, validation and test step raised `AttributeError: 'generator' object has no attribute 'size'`. `_loss` now passes the squeezed root head as `predictions` and every squeezed head as `outputs["linear_outputs"]` (the loss skips `GO:ROOT`). On the test fixture the value is 0.75 + 0.3 * 19/12 = 1.225.
- `__init__` no longer calls `tracemalloc.start()`; the memory banner print and `tracemalloc.stop()` in the epoch end are gone. Nothing in the repo read that output.
- RMSE/MSE/MAE score the root prediction only (DCell reports the root term's output as the phenotype); they used to be updated with the subsystem mean and then the root, pooling six predictions per batch of three. The validation box plot also receives the root prediction instead of the subsystem mean.
- Validation logs `val_pearson_root`/`val_spearman_root`, matching `train_*_root` and `test_*_root`. No experiment config or W&B sweep config in the repo reads the old `val_pearson`/`val_spearman` from this task (the 002 readers are the traditional-ML scripts, the 006 reader uses `int_dcell`'s `val/gene_interaction/Pearson`).
- An unknown target raises `ValueError("Unknown target '<name>': expected one of ('fitness', 'genetic_interaction_score').")` in `__init__` instead of `UnboundLocalError` on `fig` at the first plotting epoch.
- `validation_step` skips prediction collection during `trainer.sanity_checking`, so the first box plot holds only the real validation pass.
- Best-checkpoint artifacts: the artifact block ran in `on_validation_epoch_end`, which Lightning calls before `ModelCheckpoint` saves (with validation every epoch the save happens in `on_train_epoch_end`, after the module hooks), so each artifact held the previous epoch's checkpoint; with `enable_checkpointing=False` it read `None.best_model_path` and raised. Now `_log_best_checkpoint` runs from `on_train_epoch_start` and `on_train_end`, the first module hooks after the save, so the artifact `model-global_step-<n>` holds the checkpoint saved at step n, and a missing checkpoint callback logs nothing.

Evidence: `tests/torchcell/trainers/test_dcell_regression.py` (`test_loss_feeds_the_root_as_prediction_and_every_head_as_auxiliary`, `test_one_training_step_logs_closed_form_values_and_takes_one_adam_step`, `test_validate_logs_root_suffixed_correlations_like_train_and_test`, `test_unknown_target_is_rejected_at_construction`, `test_each_epoch_best_checkpoint_is_logged_once_at_its_own_step`, `test_training_without_a_checkpoint_callback_logs_no_artifact`, `test_sanity_check_predictions_stay_out_of_the_first_box_plot`).

Left as is: on non-plotting epochs the box-plot buffers keep accumulating and are plotted, with older epochs' predictions, at the next plotting epoch.
