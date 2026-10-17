---
id: cr2ew99m4m0v188m0kcscgc
title: Smoke_train
desc: ''
updated: 1792225850772
created: 1792225850772
---

## 2026.10.17 - GPU + W&B smoke test for a fresh torchcell environment

`experiments/042-hpc-smoke-tests/scripts/smoke_train.py --machine <name> --out <json>` proves,
in order and exiting non-zero at the first failure: the stack imports (torch, torch_geometric,
lightning, wandb, torchcell); a GPU is visible (CUDA, else Intel XPU; `--allow-cpu` only for
validating the script itself); a scatter-add runs on it (`torch_scatter` where installed, torch's
`index_add_` otherwise, and the run records which); a 300-step training loop on synthetic
scatter-pooled graphs lowers the loss; every step reaches W&B online (project
`torchcell_042_hpc_smoke`, run name `<machine>-<job id>`, group `<machine>`); and on CUDA a
Lightning `Trainer` with `WandbLogger` on the same run. Synthetic data so no dataset store has to
cross to a machine before its first job.

Runs and results per machine: [[experiments.042-hpc-smoke-tests]].
