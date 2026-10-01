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

## 2026.10.01 - Batch size reads num_graphs (issue #567)

Previously `_get_batch_size` sized a perturbation batch as `max(perturbation_indices_batch) + 1`. Since PR #560 the CGT sizes its output by `batch.num_graphs`, so a trailing genotype with no perturbed gene (a wild-type record) was dropped from the logged batch size while the model still predicted its row. The perturbation branch now returns `int(batch.num_graphs)`; the ladder below it (perturbed-gene count, value count, 1) is unchanged. Evidence: `test_trailing_genotype_without_a_perturbation_is_counted` in [[tests.torchcell.trainers.test_int_transformer_cell]] (3 genotypes with batch vector [0, 0, 1]: size 3, 3 prediction rows, profiling logs 3 under `batch_size=3`).
