---
id: 4mo8xlmnpi9bubzcawpgu5y
title: Test_dcell_regression_slim
desc: ''
updated: 1790756738743
created: 1790756738743
---

## 2026.09.30 - Phase 11: the slim DCell task on the same fixture

Same fixture as [[tests.torchcell.trainers.test_dcell_regression]]; eight tests. Exact `train_*` from the subsystem mean (MSE 77/108) and `train_root_*` from the root (0.75), the exact first Adam step (-1e-3 times the gradient sign on every parameter), eleven exact validation values, and `wandb.log` never called.

Findings: the same loss argument-order crash (`dcell_regression_slim.py` lines 158, 188, 227); `on_test_epoch_end` drops `test_metrics_root`, never logged or reset (line 240); `enable_checkpointing=False` raises `'NoneType' object has no attribute 'best_model_path'` (line 208).

## 2026.09.30 - Findings retired by the issue #516 fix

The three findings of this file are retired. The tests run with the constructed `DCellLoss` and assert loss 1.225 at alpha 0.3 and 0.75 + 0.7 * 19/12 at alpha 0.7; `trainer.test` returns `test_root_*` (root MSE 0.75, RMSE 0.8660254, MAE 5/6, Pearson and Spearman 0.5) beside `test_*`, and the root collection holds no updates afterwards; with checkpointing disabled a two-epoch fit logs no artifact; with checkpointing each epoch's own checkpoint is logged once at its step.

## 2026.09.30 - Loss values under the paper's sum (issue #554)

The task is built with `aux_reduction="sum"` (the init test checks `"mean"` reaches the loss). Pinned loss is 1.7 at alpha 0.3 and 0.75 + 0.7 * 19/6 = 2.9666667 at alpha 0.7 (were 1.225 and 1.8583333 under the mean). Gradient signs and the Adam-step deltas are unchanged.
