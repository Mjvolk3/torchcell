---
id: 69g037gl55rnskv275hpe9h
title: Gh_train_043_prepare
desc: ''
updated: 1792641720437
created: 1792641720437
---

## 2026.10.21 - Prepare stage on CPU

Runs `train_043.py --stage prepare --route A --workers 30` ([[experiments.043-expression-proteome-morphology.scripts.train_043]]); 32 CPUs, `--mem=96g`, 2 h limit. Job 6123 (commit `402ab04c4`): 3 min 8 s wall. `sacct` MaxRSS for the batch step reads 131 GB, which sums the 30 forked workers' resident pages (copy-on-write pages counted once per worker); the job ran inside its 96 GB cgroup, so the true peak is below that. Lower `--mem` only after reading `memory.peak` on a rerun.
