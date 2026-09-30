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
