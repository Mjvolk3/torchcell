---
id: 8o6b36a1octaxngi0c0ipce
title: Gh_smoke_dataset_token
desc: ''
updated: 1790417037079
created: 1790417037079
---

## 2026.09.26 - One GilaHyper GPU, four steps

warm (seeds 0, 1, 2) -> token run `cgt_030_smoke_tok_000` -> control run `cgt_030_smoke_ctl_000` -> report. First submission 2860 sat on Resources: `--mem=96g` with 10 CPUs exceeded MaxMemPerCPU 4.1 GB, which raised the CPU request to 24 while 8 CPUs and one GPU were free. Resubmitted as 2861 with 8 CPUs and 32 GB; started 2026-09-26 after holding and releasing the six queued jobs ahead of it (the FIFO hold/release procedure).
