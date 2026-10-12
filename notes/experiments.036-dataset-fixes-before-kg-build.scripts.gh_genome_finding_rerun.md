---
id: 5fm4v0us77vau93mjclw9ca
title: Gh_genome_finding_rerun
desc: ''
updated: 1791817931740
created: 1791817931740
---

## 2026.10.12 - Read-only rerun lanes

Array task N verifies the classes on line N of `LANES_FILE` through [[experiments.036-dataset-fixes-before-kg-build.scripts.kg4_pre_build_verification_sweep]], from a frozen detached worktree (`SNAPSHOT_ROOT`) with `DATA_ROOT` pointed at a per-task overlay ([[experiments.036-dataset-fixes-before-kg-build.scripts.verification_overlay]]). Stores are read in place and never rebuilt. 2026-10-12, from a snapshot at `fac3c689e`: wave A slurm 4394 (61 stores without a current report or failing, 9 lanes), N'Guessan 2025 alone slurm 4393, wave B slurm 4395 (Dmf and Dmi Costanzo 2016, Muenzner 2024, Parsons 2006, Yoshikawa 2009, Nichols 2011, Borchert 2024, after the builds 4336, 4337, 4348, 4374 and 4011 that were rewriting them). Two earlier submissions (4357/4358, 4388/4389) were cancelled before they started: the first predated the `no_viable_deletion_in_` reclassification, the second predated the eight runner registries the sweep did not route.
