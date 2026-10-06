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
