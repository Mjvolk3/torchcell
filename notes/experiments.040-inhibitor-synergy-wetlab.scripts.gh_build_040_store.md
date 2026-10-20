---
id: 786osbxxrply5p088r7xev7
title: Gh_build_040_store
desc: ''
updated: 1792470014567
created: 1792470014567
---

## 2026.10.19 - The launcher

`sbatch experiments/040-inhibitor-synergy-wetlab/scripts/gh_build_040_store.slurm` from the
worktree root; 16 CPUs, 64 GB, 12 h on `main`, the 033 build's sizing. Exports
`TORCHCELL_SRC` (default the `build/kg-4-0-v1-8-0` worktree) as `PYTHONPATH` and runs
[[experiments.040-inhibitor-synergy-wetlab.scripts.build_040_store]]. Output under
`$DATA_ROOT/experiments/040-inhibitor-synergy-wetlab/slurm/output/`.
