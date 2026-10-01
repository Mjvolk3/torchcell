---
id: 0gvhe90ps0kmanm3b6lif2n
title: Test_int_transformer_cell_methods
desc: ''
updated: 1790562670531
created: 1790562670531
---

## 2026.09.27 - The trainer's methods on a scripted model

Twenty-five tests on the LightningModule with a scripted stand-in model with fixed outputs and `self.log`, the viz classes and `wandb.log`/`wandb.Image` replaced by recorders. Exact loss, logged payload and metrics on a two-sample batch (MSE 2.5, RMSE sqrt 2.5, Pearson 1), the inverse-transform (MSE 250) and NaN-mask cases; the loss dispatch (`PointDistGraphReg` counts the graph term once; `MleDistSupCR` and `MleWassSupCR` receive `epoch=4` and their components are logged exactly; generic two- and three-argument losses); the validation diagnostics with exact recall-at-degree 0.5, precision-at-k 0.25, edge mass 0.2 and Spearman 0.8944; sample collection and plotting; the epoch hooks and optimizers (the cosine scheduler takes the learning rate from 1e-2 to 5e-3; ReduceLROnPlateau hits its assertion); and a real `fast_dev_run` where accumulation defers the step (effective batch 4), clipping moves SGD by exactly 1e-3, and model-profiling mode skips the optimizer. Module coverage 85% from this file, 87% with [[tests.torchcell.trainers.test_int_transformer_cell]]. Finding: `__init__` reads the epoch-0 accumulation with `.get(0, 1)`, so a string-keyed schedule such as `{"0": 3}` starts at 1 until `on_train_epoch_start` corrects it (int_transformer_cell.py lines 74 to 76). Pinned, not a finding: in this torchmetrics version a one-sample Pearson computes to NaN rather than raising, so the skip branch of `_compute_metrics_safely` is tested with a raising custom metric. Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.30 - Plateau scheduler no longer refused at the training epoch end

`test_train_epoch_end_logs_resets_and_steps_the_scheduler` no longer expects `AssertionError` for a ReduceLROnPlateau scheduler (issue #534). It now asserts the plateau scheduler is left unstepped at the training epoch end (`last_epoch == 0`, rate unchanged), since it is stepped on its monitor in `on_validation_epoch_end`, and that with two schedulers the first in Lightning's list is the one stepped.
