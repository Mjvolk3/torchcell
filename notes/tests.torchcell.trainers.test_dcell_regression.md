---
id: 7x6mb09mpq41cyd06amptrt
title: Test_dcell_regression
desc: ''
updated: 1790756731181
created: 1790756731181
---

## 2026.09.30 - Phase 11: the DCell regression task on a count-subsystem fixture

Fixture (`tests/torchcell/conftest.py`): `DCellCountSubsystems` and `DCellIdentityHeads` replace the random network with deterministic modules on the conftest DCell graph, and `make_dcell_regression_batch()` holds three knockout sets ({0,1,2,3}, {0}, {2}) against `y = [1, 0, 0.5]`. GO:1 counts [0,1,2], GO:2 counts [0,2,1], the root is GO:1 minus GO:2 = [0,-1,1]; the subsystem mean is [0, 2/3, 4/3]; the loss is 1.7. Pearson and Spearman are -0.5 on the mean and +0.5 on the root, so the tests can tell the two predictions apart. Eleven tests under `L.Trainer(fast_dev_run=True, ...)` on CPU with `wandb.log` recorded.

Findings, pinned as the code behaves:

- The default loss cannot run: the task calls `self.loss(y_hat, y, dcell.parameters())` in the old `(outputs, target, weights)` order while it builds the current `DCellLoss(predictions, outputs, target)`, so every step raises `AttributeError: 'generator' object has no attribute 'size'` (`dcell_regression.py` lines 148, 200, 290). The value tests install the deprecated `dcell_DEPRECATED.DCellLoss` per instance.
- `__init__` starts `tracemalloc` for the whole process (line 106); an autouse fixture stops it.
- The RMSE/MSE/MAE collection is updated with both the subsystem mean and the root (lines 161-162), so it pools six predictions: MSE 158/216.
- Validation logs root correlations as `val_pearson`/`val_spearman` while train and test carry a `_root` suffix (lines 224, 230).
- An unknown target raises `UnboundLocalError` on `fig` (lines 251-254); sanity-check predictions leak into the first box plot (line 246 returns early without clearing).

## 2026.09.30 - Findings retired by the issue #516 fix

All six findings of this file are retired. The tests now run with the `DCellLoss` the task constructs (no deprecated loss is installed) and assert: loss 1.225 (and 0.75 with auxiliary losses off, which pins the root as `predictions`); root-only RMSE/MSE/MAE (0.8660254, 0.75, 5/6); `val_pearson_root`/`val_spearman_root`; no `tracemalloc` tracing after construction; `ValueError` with the exact message for `target="growth_rate"`; a box plot of y against the root, and after one Adam step against the post-step root [0.001, -0.997001, 0.999001] with no sanity-pass values; one artifact per epoch holding that epoch's checkpoint; no artifact and no error with checkpointing disabled.

## 2026.09.30 - Loss values under the paper's sum (issue #554)

`DCellLoss` now sums the auxiliary MSEs, so the pinned loss is 0.75 + 0.3 * 9.5 / 3 = 1.7 (was 1.225 under the mean); the loss test also pins 1.225 under `aux_reduction="mean"` and 0.75 with auxiliary losses off. Auxiliary-head gradients doubled (GO:1 0.8 / 0.3, GO:2 0.9 / 0.3, `scale` [1.0, 0.8, 0.9]); their signs, and therefore the one-Adam-step deltas, are unchanged.
