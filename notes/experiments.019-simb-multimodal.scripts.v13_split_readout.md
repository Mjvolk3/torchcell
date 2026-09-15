---
id: 1p6irua4ii26tfqxv5q8a2z
title: V13_split_readout
desc: ''
updated: 1789452952069
created: 1789452952069
---

## 2026.09.15 - Mid-run readout of the split round at the matched epoch 1,547

`experiments/019-simb-multimodal/scripts/v13_split_readout.py` reads W&B project `torchcell_019_expr_v13` and writes `results/v13_split_readout.json`. Statistic: `roll_max` (max of a centered 5-epoch rolling mean of `val/expression/pearson_per_feature`) read at the matched epoch, the lowest epoch any run has reached. Runs still training (job 2397311, 6,000-epoch budget); every number below is PARTIAL. One run never left the plateau (`V_ref_s0` seed 1, `825on260`, 0.026) and is set aside where marked.

**Partition.** Mean over both readouts and seeds, plateau run excluded: split 0 0.185 (sd 0.011, n 7), split 1 0.156 (0.009), split 2 0.127 (0.018), split 3 0.132 (0.010). Between-partition sd 0.027, range 0.058, against a pooled within-partition sd of 0.012: the partition moves the score by about five times the replicate spread, and split 0 is the highest draw, as the linear baselines said it would be.

**H_concat minus H_ref**, paired within card and seed: all 12 pairs +0.014 (sd 0.049, driven by the plateau pair at +0.156); the 11 clean pairs +0.001 (sd 0.022, t 0.2, 6 of 11 positive). The v12 advantage of +0.0135 on split 0 at 1,399 epochs does not show across partitions at 1,547 epochs; on split 0 itself two of three clean pairs are negative (-0.029, -0.022, +0.004).

**90/10 minus 80/10/10** on split 0, same validation strains: concat +0.025 and +0.035; reference -0.011 (seed 0) and +0.165 against the plateau run. Three clean pairs average +0.016, inside the within-partition spread.

**Against the linear baselines on the same partition (val):** split 1 the trained arms lead B3 by 0.04 to 0.05; split 0 by 0.05 to 0.06; splits 2 and 3 by 0.01 to 0.02 only (B3 ProtT5 0.111 and 0.126 against H_ref 0.121 and 0.128).

Example runs: split 0 reference seed 0 `wq8y8nd5` (0.205), the plateau run `825on260`, split 2 reference seed 1 `hjx0y9f1` (0.105), 90/10 concat seed 1 `wmmw6ff9` (0.217).
