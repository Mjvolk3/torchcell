---
id: xfkizhtfkv5es5uvyb1yroz
title: Wandb_grid_bench_view
desc: ''
updated: 1791266703119
created: 1791266703119
---

## 2026.10.06 - Views for the depth x operator grid

Labels the sixteen runs of job 3274 (`gh_small_model_bench.slurm` cells 10 to 13) `<operator>_L<layers>`, sets the W&B group to the operator (the launcher tagged only the cell; the operator is read from the config), keeps the checkpoint directory in config `ckpt_group`, and writes two saved Charts views, ids in `results/wandb_view_ids.json` under `grid_bench` (grouped by operator, `rltcn58yxhi`) and `grid_bench/by_depth` (`3ne19cnkxjq`). Sections: validation Pearson, train against validation on one panel, prediction spread against the 0.05 launch gate, loss, throughput. Rerun after any resync; idempotent. Read of the runs in [[experiments.019-simb-multimodal.scripts.gh_small_model_bench]].

- by operator <https://wandb.ai/zhao-group/torchcell_019_grid_bench?nw=rltcn58yxhi>
- by depth <https://wandb.ai/zhao-group/torchcell_019_grid_bench?nw=3ne19cnkxjq>
