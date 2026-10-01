---
id: bji1524wnvt66lb4h38zahm
title: Simple_linear_regression
desc: ''
updated: 1790812878629
created: 1790812878629
---

## 2026.09.30 - Target validation, sanity leak, and checkpoint artifacts (issue #516)

Previous behavior and the fix:

- An unknown target raises `ValueError("Unknown target '<name>': expected one of ('fitness', 'genetic_interaction_score').")` in `__init__` instead of training and validating and then raising `UnboundLocalError` on `fig`.
- `validation_step` skips prediction collection during `trainer.sanity_checking`, so the epoch-0 box plot holds only the real validation pass (three values, not six).
- Model artifacts were logged one epoch late and only on plotting epochs (the block sat behind the plotting early return in `on_validation_epoch_end`). They are now logged on every epoch from `on_train_epoch_start` and `on_train_end`, after `ModelCheckpoint` has saved, so `model-global_step-1` holds `epoch=0-step=1.ckpt` and `model-global_step-2` holds `epoch=1-step=2.ckpt` whatever `boxplot_every_n_epochs` is. A missing checkpoint callback logs nothing.

Evidence: `tests/torchcell/trainers/test_simple_linear_regression.py` (`test_unknown_target_is_rejected_at_construction`, `test_sanity_check_predictions_stay_out_of_the_first_box_plot`, `test_artifact_of_the_current_best_is_logged_on_every_epoch`).
