---
id: s8zyxc6dd8zcgng1ap95agn
title: Int_hetero_cell
desc: ''
updated: 1765217693705
created: 1765217693705
---

## 2026.10.01 - Batch size reads num_graphs (issue #567)

`RegressionTask._get_batch_size` and `DiffusionRegressionTask._get_batch_size` sized a perturbation batch (no `gene.x`) as `max(perturbation_indices_batch) + 1`, which drops a trailing genotype with no perturbed gene. Both now return `int(batch.num_graphs)` in that branch; the `x` rows / perturbed-gene count / value count / 1 ladder is unchanged. With the lazy collate now writing `num_graphs` (issue #572), every 006 lazy batch carries it. Tests: [[tests.torchcell.trainers.test_int_hetero_cell]].

## 2026.10.02 - One shared step, plateau stepped on its metric, validated schedules (issue #614)

Owner decision 2026-10-02: "614 sounds like it needs fix." Previous behavior and the fix, item by item. Tests: [[tests.torchcell.trainers.test_int_hetero_cell]], [[tests.torchcell.trainers.test_int_hetero_cell_diffusion]].

- **Plateau scheduler.** `on_train_epoch_end` called `scheduler.step()` on any scheduler, so a `ReduceLROnPlateau` raised `TypeError: ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics'` at the first epoch end, and a missing or misspelled `type` silently built one. Now `scheduler_type` refuses a missing or unknown `type` by name at construction (valid: `CosineAnnealingWarmupRestarts`, `CosineAnnealingLR`, `ReduceLROnPlateau`); `on_train_epoch_end` steps only a non-plateau scheduler; `on_validation_epoch_end` steps a plateau scheduler once per non-sanity validation epoch on `PLATEAU_MONITOR` (`val/gene_interaction/MSE`) just computed, as the CGT trainer does (#534). Under manual optimization Lightning steps no scheduler and drops the `monitor` key with a warning, so the task must do it.
- **Accumulation schedule.** `normalize_accumulation_schedule` runs at construction: a key is an integer epoch >= 0 given as an `int` or a digit string, a value a positive `int`; anything else is refused by name. Digit strings are accepted because `wandb.config` returns every mapping key as a string (checked with wandb 0.30.0: `{0: 16, 5: 8}` reads back as `{'0': 16, '5': 8}`), and the scripts pass `wandb.config["regression_task"]["grad_accumulation_schedule"]`. Keys are sorted numerically (they sorted as text before, item 9). `"0:16"`, what YAML reads from `{0:16}`, is refused with a message naming the missing space. The five 006 configs that wrote `{0:16}` / `{0:8}` are corrected to `{0: 16}` / `{0: 8}`; as committed they crashed at the first epoch start.
- **Batch size (rest of issue #596).** `_get_batch_size` returns `num_graphs` and refuses a batch without it; the `gene.x` node-row rung and both fallbacks are gone. Both collaters the callers use (PyG and `LazyCollater`) write `num_graphs`. `_shared_step` uses it for every log and refuses a model whose prediction rows differ from it.
- **DDP rank factor.** `effective_batch_size` = genotypes on this rank x accumulation steps x `trainer.world_size` (the genotypes behind one optimizer step across ranks, assuming equal per-rank batches). It read `trainer.strategy._strategy_name`, which no Lightning strategy has.
- **Diffusion train metrics.** `DiffusionRegressionTask.scores_train_predictions = False`: `GeneInteractionDiff` returns `zeros_like(targets)` in training mode, so train-stage predictions update no metric collection and keep no plot sample; the skip is logged once at INFO. Validation and test score the sampled predictions as before.
- **One `_shared_step`.** `DiffusionRegressionTask` now subclasses `RegressionTask` and overrides only `_stage_loss` (train: the shared loss call, kept for `train/avg_diffusion_loss`; val/test: `F.mse_loss` of the sampled predictions, logged as `{stage}/inference_mse`) and the two epoch-end hooks that log its buffers. Shared: one loss dispatcher (`PointDistGraphReg` gets representations and epoch; `ICLoss`, `DiffusionLoss` get `z_p`; the MLE losses get `z_p` and epoch; any other loss gets `(pred, target)`, so `nn.MSELoss` no longer receives `z_p`; a `z_p` loss on a model without `z_p` is refused by name), one component logger (vectors element by element, numbers as is; the `PointDistGraphReg` branch dropped vectors, the diffusion copy averaged them and dropped numbers), the graph-regularization term, `ValueError("No loss function provided")` instead of the diffusion copy's bare `assert`.
- **Unit mismatches.** A non-tensor inverse result raises `TypeError` with the #534 message; a batch carrying `phenotype_values_original` on a task without `inverse_transform` raises `ValueError` before the model runs.
- **Metric guard.** `_compute_metrics_safely` is deleted; `_log_metrics` computes, logs and resets, and a metric error propagates. An empty epoch logs the NaN torchmetrics computes.
- **Unused hyperparameters.** `batch_size` and `device` are still accepted (every script and older checkpoints pass them) but excluded from `save_hyperparameters`; nothing read them.

Left open: the COO layout is still read positionally (issue item 7). A real PyG collation carries `phenotype_types` as one list per genotype and leaves `phenotype_sample_indices` un-offset (all 0 for one label per sample), so a type or sample check needs that collated layout pinned on a real 006 batch first.

## 2026.10.02 - Review fixes on PR #637

- `DiffusionRegressionTask` refuses at construction any `loss_func` that is not a `DiffusionLoss` (None stays allowed for evaluation only). The 006 diffusion script also offers `loss: logcosh` and `icloss`; with the train stage no longer scoring placeholders, those would have trained on the all-zero placeholders without an error (on main `LogCoshLoss` raised `TypeError` at the first step).
- A plateau step on a validation epoch whose collection received no rows (all targets NaN) is refused by name instead of stepping on NaN.
- The schedule-key refusal shows the YAML space hint only for a key containing a colon.
- The new `cast` calls are replaced by annotated locals.
- Tests: the plateau test now runs with the real inverse, so the original-unit val MSE 14 / 3 (not the transformed 7 / 6) is pinned as the stepped value; refusal tests for `LogCoshLoss` and `nn.MSELoss` on the diffusion task and for the empty validation epoch. The shared tests give the diffusion task a `DiffusionLoss`-typed squared-error stand-in.
