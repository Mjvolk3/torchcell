---
id: 5fm4v0us77vau93mjclw9ca
title: Gh_genome_finding_rerun
desc: ''
updated: 1791817931740
created: 1791817931740
---

## 2026.10.12 - Read-only rerun lanes

Array task N verifies the classes on line N of `LANES_FILE` through [[experiments.036-dataset-fixes-before-kg-build.scripts.kg4_pre_build_verification_sweep]], from a frozen detached worktree (`SNAPSHOT_ROOT`) with `DATA_ROOT` pointed at a per-task overlay ([[experiments.036-dataset-fixes-before-kg-build.scripts.verification_overlay]]). Stores are read in place and never rebuilt. 2026-10-12: wave A slurm 4357 (62 stores without a current report or failing, 9 lanes), wave B slurm 4358 (the 9 stores `fix/audit-yeast-refusals-1045` was rebuilding, after jobs 4336, 4337, 4338 and 4314).
