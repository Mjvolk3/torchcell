---
id: g5fqwrh4ib1v22u69p50foo
title: Gh_build_043_store
desc: ''
updated: 1792393311628
created: 1792393311628
---

## 2026.10.19 - Launcher

Runs [[experiments.043-expression-proteome-morphology.scripts.build_043_store]] under slurm
on GilaHyper (partition `main`, 96 GB, 16 CPUs, 12 h, job name `043-build-store`), the
shape of the 019 `gh_build_fig3_store.slurm`. Submit from the worktree root:

```bash
sbatch experiments/043-expression-proteome-morphology/scripts/gh_build_043_store.slurm
```

Output: `/scratch/projects/torchcell-scratch/experiments/043-expression-proteome-morphology/slurm/output/043-build-store_<jobid>.out`.
