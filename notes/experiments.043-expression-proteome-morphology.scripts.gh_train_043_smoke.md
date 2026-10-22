---
id: lrdg0g1ryehi5nxdit766c9
title: Gh_train_043_smoke
desc: ''
updated: 1792641712246
created: 1792641712246
---

## 2026.10.21 - The smoke cell

`joint_9`, HL-Gauss, equal weighting, seed 0, batch 8, one card, `--mem=48g`, `--time=08:00:00` ([[experiments.043-expression-proteome-morphology.scripts.train_043]]). Epochs are `min(50, what fits 7.6 h)` by the measured first epoch (training plus evaluation) with a 20 min margin; the 30-step projection is printed at startup (`[projection]` line, `setup/probe_seconds_per_step`) and the fixed epoch count with the registered window after epoch 1 (`[budget]` line). The smoke caps marks at 1,024 per record (`data.mark_cap`), which binds on the segregants only; the design has no cap.

Submitted as job 6124 with `--dependency=afterany:6122`, the last of the embedding chain 6106 -> 6115 -> 6121 -> 6122. If any key the arm uses has no vector when it starts, `load_vector_table` raises `KeyError` and the job fails at setup.

Twins, prepared and NOT submitted (one 043 GPU job at a time; chain each behind the previous):

```bash
sbatch --export=ALL,ARM=alone_kemmeren --dependency=afterany:6124 experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
sbatch --export=ALL,LOSS=pinball --dependency=afterany:<previous> experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
sbatch --export=ALL,LOSS=mse --dependency=afterany:<previous> experiments/043-expression-proteome-morphology/scripts/gh_train_043_smoke.slurm
```

The `alone_kemmeren` twin is time-budgeted like the smoke, so it is NOT matched to the joint run on optimizer steps (decision S5); a transfer reading needs the step-matched pair: set `budget.max_epochs` so that epochs times the alone arm's steps per epoch (Kemmeren alone: 1,202 training records, 151 steps at batch 8) equals the joint run's step count, and drop `budget.time_budget_hours`.
