---
id: p1iwuzhbzbusi2d7ce5f0q3
title: Wandb_seed_histories
desc: ''
updated: 1791808021542
created: 1791808021542
---

## 2026.10.12 - One W&B history per seed, shared by the Figure 2 gate scripts

Job 4111 (the lower-order fit refresh with `--wandb`) failed on the assertion "more than
one seed {0, 1}; pick one per arm": a seed 1 of the arm Q config
(`cgt_030_s3_q_tok_embfit_004`) was launched on Delta on 2026-10-12 (jobs 22812513 to
22812523, run `ddnpwq4z` the rank 0). The three gate readers stitched every tagged run by
epoch regardless of seed, so without the assertion seed 1's first epochs would have
overwritten seed 0's. This module reads the rank-0 runs per seed and never mixes them;
`lower_order_baselines_030.py`, `figure2_gate_numbers.py` and `figure2_gate_sheet.py`
import it. Seed 0 is the headline (`WANDB_HEADLINE_SEED`); every seed is recorded, a seed
with fewer than ten synced epochs is `partial` and drawn nowhere. It imports only pandas
and wandb: importing `lower_order_baselines_030` resets matplotlib's rcParams to a 16 pt
style (font.size 16, axes.labelsize 16, DejaVu Sans), which blew up the sheet's second
page on the first try.
