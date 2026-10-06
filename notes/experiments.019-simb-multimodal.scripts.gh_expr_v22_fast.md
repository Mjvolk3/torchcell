---
id: 1cyfw1pj6ozp6b329ak2zxb
title: Gh_expr_v22_fast
desc: ''
updated: 1791269319963
created: 1791269319963
---

## 2026.10.06 - v22 wave 1 launched on the four GilaHyper cards

Job 3315, submitted 01:58 CDT, W&B project `torchcell_019_expr_v22`, 1,200 epochs, time limit 8 h 45 min (the cards are ours until 11:00). One card per split seed (0 to 3), four arms on each card, every arm the v21 reference with one change, all on the fast code path (batched operator, splits in memory, two persistent workers):

| arm | change against the v21 reference |
|---|---|
| F_ref | none (batch 32, six layers, width 90, lr 3e-4) |
| F_b128 | batch 128, lr 6e-4 |
| F_b128lr4 | batch 128, lr 1.2e-3 |
| F_l4w180 | four layers, width 180 |

Why: the profile ([[experiments.019-simb-multimodal.scripts.gh_profile_cgt_expr]]) found the encoder's self-attention to be a fixed cost per step, so batch 128 takes 9 encoder passes per epoch against 35; whether it learns the same has never been scored. No width above 90 has run past 151 epochs. F_ref on split seeds 0 to 3 is also the check of the fast code against v21's S_ref on Delta (same config and seeds, loop operator, loader from LMDB).

Scoring as v21: mean of `val/expression/pearson_per_feature` over epochs 1,000 to 1,200, paired by split seed against F_ref. Four split seeds size an effect; they do not decide it.

## 2026.10.06 - Relaunches, the wave-1 read at 03:59, and wave 2

Launch history of wave 1: job 3315 (four arms per card, cancelled at ten minutes: the processes time-slice a card and batch 128 was no faster per epoch than batch 32); job 3319 (one arm per card, cancelled at 27 minutes: 20 to 32 s per epoch, target decode found to be 60 percent of wall time, and the fourth batch-128 run of each card out of memory at 12.2 GB per run); job 3323 at 02:25 on the vectorized decode, the one read below. Layout of 3323: F_b128 and F_b128lr4 on split seeds 0 to 2 (three per card), F_ref on 0 to 2, F_l4w180 on 0 and 1.

Read at 03:59 (`v22_readout.py`, PARTIAL, epochs 216 to 461; trailing 20-epoch mean of validation Pearson per feature at the ladder epoch, then the last logged prediction spread ratio and eval-mode train Pearson):

| arm | split | epoch | at 100 | at 200 | at 300 | at 400 | spread ratio | train Pearson | s per epoch |
|---|---|---|---|---|---|---|---|---|---|
| F_b128 | 0 | 461 | 0.007 | 0.028 | 0.001 | 0.001 | 0.000 | 0.024 | 11.6 |
| F_b128 | 1 | 456 | -0.010 | 0.006 | -0.013 | -0.009 | 0.221 | 0.265 | 11.6 |
| F_b128 | 2 | 449 | -0.013 | 0.103 | 0.096 | 0.066 | 0.234 | 0.262 | 11.7 |
| F_b128lr4 | 0 | 436 | -0.001 | 0.003 | 0.002 | -0.001 | 0.000 | -0.002 | 12.3 |
| F_b128lr4 | 1 | 448 | -0.001 | 0.002 | 0.001 | 0.000 | 0.000 | 0.006 | 11.8 |
| F_b128lr4 | 2 | 439 | 0.000 | 0.001 | -0.003 | 0.000 | 0.000 | 0.013 | 11.9 |
| F_l4w180 | 0 | 308 | 0.037 | -0.004 | 0.000 | | 0.000 | -0.076 | 17.5 |
| F_l4w180 | 1 | 300 | -0.010 | -0.007 | -0.001 | | 0.000 | -0.153 | 17.7 |
| F_ref | 0 | 227 | 0.029 | 0.004 | | | 0.154 | 0.258 | 23.7 |
| F_ref | 1 | 216 | -0.008 | -0.017 | | | 0.195 | 0.258 | 24.6 |
| F_ref | 2 | 219 | 0.117 | 0.095 | | | 0.167 | 0.249 | 23.9 |

- A prediction spread ratio of 0.000 with a train Pearson near zero is the collapse to the per-gene mean (the round's own launch gate is 0.05). **F_b128lr4 was collapsed on 3 of 3 split seeds, F_l4w180 on 2 of 2, F_b128 on 1 of 3, F_ref on 0 of 3.** F_l4w180 split 0 and F_b128 split 0 showed signal first (0.037 at epoch 100, 0.028 at epoch 200) and lost it, so these are collapses during training, not runs that never launched.
- So at batch 128 the quadrupled learning rate does not train, and the doubled one is unstable; four layers at width 180 does not train at the reference learning rate. None of this says batch 128 or width 180 cannot work; it says they do not work at these settings without something that stabilizes them.
- The two dead cards were freed at 04:00 (tasks 3323_1 and 3323_3 cancelled; their runs stay in W&B as partial, collapsed runs).

**Wave 2, job 3327, 04:00**, on the freed cards, three split seeds each, on the newer code (ranks on the device, fused encoder attention on the unregularized layers):

| arm | change against the v21 reference |
|---|---|
| F_b128lr1 | batch 128, learning rate unscaled (3e-4) |
| F_b128wu | batch 128, lr 6e-4 behind a 50-epoch linear warmup from 1e-6, flat afterwards |

They bracket the stability question for batch 128: is the collapse the learning rate, and does warmup remove it.
