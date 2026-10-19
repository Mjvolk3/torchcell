---
id: 5io8ae9wojevoahe3zmdllm
title: Gh_within_dataset_ceilings
desc: ''
updated: 1792445295452
created: 1792445295452
---

## 2026.10.19 - Launcher

Runs [[experiments.043-expression-proteome-morphology.scripts.within_dataset_ceilings]]
under slurm (partition `main`, 64 GB, 8 CPUs, 4 h). `STAGE` selects `all` (default),
`jackson`, `kemmeren`, `hu`, `segregants` or `figures`; each stage merges its section
into `results/within_dataset_ceilings.json`, so one stage can be rerun alone.
