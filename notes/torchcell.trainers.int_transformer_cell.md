---
id: s8v413ogb64c2agud1zbrw5
title: Int_transformer_cell
desc: ''
updated: 1790815582406
created: 1790815582406
---

## 2026.09.30 - Plateau scheduler, shared loss logging, inverse transform refusal

Four fixes from issue #534.

- **The default `ReduceLROnPlateau` scheduler completes an epoch.** The module runs under manual optimization, so Lightning steps no scheduler and ignores the `monitor` key. `on_train_epoch_end` used to assert the scheduler was not a plateau scheduler, so the default config (no `type`) raised `AssertionError` after the first step. The plateau scheduler is now stepped in `on_validation_epoch_end` (outside the sanity check) on `PLATEAU_MONITOR = "val/gene_interaction/MSE"`, which validation computes before `on_train_epoch_end`; `on_train_epoch_end` steps only epoch-interval schedulers. Without a validation loop the plateau scheduler is never stepped and the learning rate stays fixed.
- **One component logger for every loss.** `_log_loss_components` logs a one-element tensor or a number as `{stage}/{key}` and a multi-element tensor as `{stage}/{key}_{i}`. The `PointDistGraphReg` branch used to drop multi-element components that the generic branch logged.
- **A non-tensor inverse transform output raises `TypeError`** naming the transform class and the returned type. It used to be ignored, scoring the transformed predictions on the original scale (MSE 1285 in the pinned case instead of 250).
- **The edge recovery precision plot is skipped, with one INFO log line, when no graph has a counted node at any k.** It used to be called with empty inner dicts. Skipping rather than raising because an empty accumulator is a legitimate state and the recall and mass plots already skip it.

Tests in [[tests.torchcell.trainers.test_int_transformer_cell]].
