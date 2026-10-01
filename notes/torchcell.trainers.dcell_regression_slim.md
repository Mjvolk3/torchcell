---
id: 16t4mzc4wf8znjuu5zynedk
title: Dcell_regression_slim
desc: ''
updated: 1790812870956
created: 1790812870956
---

## 2026.09.30 - Loss call order, test root metrics, and checkpoint artifacts (issue #516)

Previous behavior and the fix:

- Loss: the steps called `self.loss(y_hat, y, dcell.parameters())` against the current `DCellLoss(predictions, outputs, target)`, so every step raised `AttributeError: 'generator' object has no attribute 'size'`. `_loss` now passes the squeezed root as `predictions` and every squeezed head as `outputs["linear_outputs"]`; at alpha 0.3 the fixture loss is 1.225, at alpha 0.7 it is 0.75 + 0.7 * 19/12.
- `on_test_epoch_end` logs `test_root_*` beside `test_*` and resets both collections, mirroring train and validation; the root state used to be updated, never logged, and carried into the next test run.
- Best-checkpoint artifacts: the artifact block ran in `on_validation_epoch_end`, which Lightning calls before `ModelCheckpoint` saves (with validation every epoch the save happens in `on_train_epoch_end`, after the module hooks), so each artifact held the previous epoch's checkpoint; with `enable_checkpointing=False` it read `None.best_model_path` and raised. Now `_log_best_checkpoint` runs from `on_train_epoch_start` and `on_train_end`, the first module hooks after the save, so the artifact `model-global_step-<n>` holds the checkpoint saved at step n, and a missing checkpoint callback logs nothing.

Evidence: `tests/torchcell/trainers/test_dcell_regression_slim.py` (`test_loss_feeds_the_root_as_prediction_and_every_head_as_auxiliary`, `test_test_epoch_end_logs_and_resets_the_root_metrics`, `test_each_epoch_best_checkpoint_is_logged_once_at_its_own_step`, `test_training_without_a_checkpoint_callback_logs_no_artifact`).
