---
id: wln0wdz6tb4rn8pzyfhcncs
title: Igb_mmli_cgt_030
desc: ''
updated: 1790417088568
created: 1790417088568
---

## 2026.09.26 - The mmli launcher

`igb_mmli_cgt.slurm` of 025 pointed at 030: runs from a detached worktree prepared on the LOGIN node (`~/projects/torchcell.worktrees/030-per-entry-dataset-token`; compute nodes have no git), preflight requires the build, the composite embedding builds and a `data_module_cache/index_seed_*.json` (refuses to let four DDP ranks compute the split), 4 x A100, 250 GB, 4 days. Usage: `sbatch --export=ALL,PROJECT_ROOT=<worktree> -J 030-r1w1-s0 experiments/030-solid-growth-multi/scripts/igb_mmli_cgt_030.slurm cgt_030_s3_r_tok_embfit_001 +seed=0`. Read MaxRSS and the epoch-1 wall time of seed 0 before seeds 1 and 2.
