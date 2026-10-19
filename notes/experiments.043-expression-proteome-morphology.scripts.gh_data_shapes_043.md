---
id: 7ozlw3cy38tbhcqe73z1dr6
title: Gh Data Shapes 043
desc: ''
updated: 1792450005669
created: 1792450005669
---

## 2026.10.19 - Full pass of data_shapes_043.py under slurm

Submits [[experiments.043-expression-proteome-morphology.scripts.data_shapes_043]] in
`--mode full` (every one of the 173,617 raw records, 16 GB, 2 CPUs, `main` partition).
The head-only raw dataset index is reused when
`raw_043/raw/raw_dataset_name_index.json` already exists (written by the login-node sample
run, 563 s). Job 5902 on gilahyper, commit 207093b3a; the result is
`experiments/043-expression-proteome-morphology/results/data_shapes_043.json` with
`"mode": "full"` and `"slurm_job_id": "5902"`.

```bash
sbatch experiments/043-expression-proteome-morphology/scripts/gh_data_shapes_043.slurm
```
