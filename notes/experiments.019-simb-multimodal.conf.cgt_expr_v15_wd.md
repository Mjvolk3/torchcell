---
id: sphydds276e5bbp27oh0g4m
title: Cgt_expr_v15_wd
desc: ''
updated: 1789456733687
created: 1789456733687
---

## 2026.09.15 - The weight-decay round: strong L2 on the incumbent at the full budget

Config `experiments/019-simb-multimodal/conf/cgt_expr_v15_wd.yaml` (inherits [[experiments.019-simb-multimodal.conf.cgt_expr_v13_split]]; only the W&B project changes). Arms `W_ref_s<k>` (decay 1e-8, the incumbent), `W_wd1e2_s<k>` (1e-2), `W_wd1e1_s<k>` (1e-1) in `gh_expr_008_arm.sh`; stage `wd` in [[experiments.019-simb-multimodal.scripts.igb_expr_wave5]].

**Why.** The only measurement so far is the v10 grid: 1e-4 against 1e-8 at epochs up to 990, +0.005 at t 0.5, a null at a resolution of 0.02. That left the AdamW range (1e-2 to 1e-1) and the full budget untested. The v13 read at epoch 1,547 puts the largest generalization effect on the partition (five times the within-partition spread), which no regularizer changes; the question here is whether strong decay moves the score at all at 6,000 epochs.

**Design.** Split seeds 1 and 2 (the draws below split 0), two init seeds, 6,000 epochs, twelve runs on three cabbi cards, four per card with arms mixed on a card; contrasts paired within split and seed. The reference arms double as two more draws of v13's reference on splits 1 and 2.

**Expectation, stated before the run.** Hypothesis: no gain, because the loss and metric shapes do not show overfitting on the ranked metric and the model looks information-limited at 1,244 strains; the interesting outcome would be 1e-1 hurting, which would say the weights the readout needs are large.
