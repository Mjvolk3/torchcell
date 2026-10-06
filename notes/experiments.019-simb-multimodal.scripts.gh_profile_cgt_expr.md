---
id: x8gfju299fe3e7ay7t71kgb
title: Gh_profile_cgt_expr
desc: ''
updated: 1791268380143
created: 1791268380143
---

## 2026.10.06 - Where an epoch of the expression CGT goes

One v21 reference run (`cgt_expr_v21_small`, arm `S_ref_s0`, 1,103 training strains, 35 steps at batch 32) under Lightning's simple profiler, four epochs, eval-mode train pass every epoch. GilaHyper RTX 6000 Ada, jobs 3302, 3306, 3307, 3311, 3312, 3313; W&B `torchcell_019_profile`. The trainer takes `+trainer.profiler=simple` (or `pytorch`); cells are listed in the script header. Seconds per epoch are the profiler's total divided by four epochs.

One run per card:

| cell | epoch | sample fetch | model step (forward, backward, optimizer) | validation end hook, per call |
|---|---|---|---|---|
| batch 32, zero workers | 151 | 57.5 | 8.2 | 72 |
| batch 32, three workers | 59 | 8.1 | 8.6 | 38 |
| batch 64, three workers | 61 | 11.4 | 6.6 | 38 |
| batch 128, three workers | 65 | 12.7 | 5.6 | 39 |
| batch 32, three workers, operator LOOP | (not read) | 9.0 | 14.3 | (not read) |
| batch 32, zero workers, splits in memory | 32 | 3.2 | 8.1 | 20 |

Four runs per card (per run):

| cell | sample fetch | model step |
|---|---|---|
| batch 32, three workers, batched operator | 6.4 | 18.9 |
| batch 32, three workers, operator LOOP | 2.4 | 25.7 |
| batch 64, three workers | 13.7 | 15.3 |
| batch 128, three workers | 17.9 | 8.7 |

What the tables say:

- **A sample costs about 50 ms to produce and the loader repeats it every epoch.** 1.64 s per batch of 32 at zero workers: an LMDB read, a JSON parse, the pydantic reconstruction of an experiment with thousands of phenotype values, and the graph processor. That is 57 s of a 151 s epoch at zero workers, the setting Delta ran until 2026-10-05.
- **The eval-mode train pass rebuilds its DataLoader on every call** (`_train_eval_pass` calls `datamodule.train_dataloader()`), so with workers it pays a worker spawn and import, about 30 s, every time. The profile ran it every epoch; production runs it every tenth, about 3 s per epoch amortized. Not fixed yet.
- **The model step is 8 s of the epoch at batch 32 on an unshared card**, and it is paced per step, not per sample: 0.21 s per step at batch 32, 0.33 at 64, 0.56 at 128. The kernel profile (job 3311, two validation steps) puts 128 of 183 ms of CUDA time in `scaled_dot_product_attention`, the encoder's self-attention over all 6,607 gene tokens, which runs once per step on the wild-type graph whatever the batch. So a larger batch buys fewer encoder passes per epoch: 8.2, 6.6, 5.6 s per epoch at 32, 64, 128.
- **The batched perturbation operator removes 40 percent of the step** against the per-strain loop: 14.3 to 8.6 s per epoch on an unshared card, 25.7 to 18.9 s at four per card.
- **Four runs on a card are GPU-bound**: the step per run goes from 8.6 to 18.9 s per epoch, so the card does about twice the work of one run, not four times.
- **Holding the splits in memory** (`+data_module.materialize=true`, `MaterializedSplit` in the trainer) costs 63 + 7 + 8 s once at setup and leaves 3.2 s per epoch of collation at zero workers. Packed four per card at zero workers the training bar read 35 to 36 s per epoch at batch 32 (cell 11, partial read) against 26 to 27 s with three workers and no materialization (cell 4), so collation in the main process competes with the step when the card is shared; the v22 round therefore runs materialized splits WITH two persistent workers.
- Precision is already `bf16-mixed` in this config lineage.

Changes made on the strength of this, all in commit of 2026-10-06: the batched operator and `pooled_perturbed` in `torchcell/models/equivariant_cell_graph_transformer.py` (the loop is kept as `_forward_loop`, selected by `TORCHCELL_PERT_OPERATOR=loop`; `test_batched_perturbation_operator_matches_loop` holds the two within 1e-5 on eight cases), `MaterializedSplit` and the `trainer.profiler` pass-through in `train_cgt_multitask.py`.

## 2026.10.06 - The step's real cost was target decoding on the CPU, found by stack sampling

The simple profiler lumps everything inside `training_step`, so the tables above could not see inside the step. Stack samples of two live v22 runs (`py-spy record`, 90 s at 20 Hz, main thread, job 3319: one batch-32 run sharing a card with two others, one batch-128 run likewise):

| share of main-thread samples | batch 32 | batch 128 |
|---|---|---|
| `_extract_targets_and_masks` | 59.0 % | 64.9 % |
| model forward (all of `CellGraphTransformer.forward`) | 20.0 % | 7.8 % |
| `validation_step` | 8.6 % | 7.7 % |
| batch transfer to the device | 6.9 % | 2.9 % |
| `_reduce_epoch_pearson` (CPU metrics, incl. scipy rank for Spearman) | 3.0 % | 7.5 % |
| checkpoint saving | 0.7 % | 1.3 % |

`_extract_targets_and_masks` decoded each graph's target row from the COO phenotype list in a Python loop over the batch, with about ten small operations per graph on DEVICE tensors, each one a point where the CPU waits for the GPU (`.tolist()` of 6,000 type indices, `bool(x.any())`, `unique`), plus a list comprehension over every value. When several processes share a card every one of those waits queues behind the other processes' kernels, which is why packing hurt so much more than the GPU work alone predicted, and why batch 128 (four times the graphs per step) was no faster per epoch than batch 32 in a shared card (19 to 20 s against 29 to 32 s at three per card, job 3319, epochs 15 to 50).

Fix: `torchcell/trainers/coo_targets.py`, `decode_head_targets`, the same decode in a fixed number of tensor operations (name table gather, one group id per graph and experiment, `bincount` for group sizes, one stable sort); the loop is kept as `decode_head_targets_loop` and `tests/torchcell/trainers/test_coo_targets.py` holds the two equal on ten cases (vector and scalar heads, a keep mask, an absent label, a wrong-width group beside a correct one, both error paths). The trainer calls the vectorized form. Timing with the fix is the v22 relaunch (job 3323).

Correction to the section above: its statement that four runs on a card are GPU-bound was an inference from the step time under packing; the step time included these waits, so the GPU-bound floor is not yet measured.

## 2026.10.06 - What each fix bought, and where the floor now is

| change | measured effect | where measured |
|---|---|---|
| batched perturbation operator | model step 14.3 to 8.6 s per epoch, one run per card | profile cells 1 and 7 |
| vectorized target decode | batch 128 at three per card 19 to 20 s per epoch down to 10 to 12; batch 32 at three per card 29 to 32 down to 23 to 24 | jobs 3319 and 3323, W&B `perf/epoch_seconds`, last 20 epochs |
| fused attention on unregularized encoder layers | optimizer step 0.526 to 0.466 s at batch 128 (11 percent) | A/B inside job 3323, five epochs each, card shared with two other runs |
| torch-native ranks, metric reduction on the device | not yet timed in a full run (the CPU reduction was 26 percent of a batch-128 run's main thread) | stack samples, job 3323 |

Stack samples after the decode fix (job 3323): the main thread now waits at the first GPU sync after the encoder and in the prediction gather, so the runs are GPU-bound. The operator alone costs about 55 ms forward and backward at batch 32 and 190 ms at batch 128 on a shared card, independent of the perturbed-set size (1, 3 or 16): it is the feed-forward block and the norms over every gene token of every strain, batch times 6,607 tokens, not the attention. What is left is the architecture's own per-strain, per-gene work; further speed is an architecture arm (a narrower operator feed-forward, fewer tokens), not a code fix.

The encoder change: the graph regularizer reads the attention of the layers named in `regularized_heads[*].layer` (layer 1 here) and no other, but every layer was asked for its weights whenever lambda > 0, so all six took the manual path. Only the regularized layers are asked now; `test_fused_attention_on_unregularized_layers_matches_manual_everywhere` holds outputs and the regularization loss equal in eval mode, and `TORCHCELL_ENCODER_ATTENTION=manual` restores the old path.
