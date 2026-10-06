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
