---
id: va52pg2leiakpnnx5uy8lq2
title: Igb_expression_baselines_split
desc: ''
updated: 1791255596601
created: 1791255596601
---

## 2026.10.05 - Twelve split seeds of linear and nearest-neighbor baselines without a GPU

The Figure 3 benchmark row compares the transformer against ridge and nearest-neighbor baselines on every split seed, and those baselines need only CPU, so they should not wait behind GPU queues. This launcher runs `expression_baselines_split.py` on the IGB `normal` partition as an array over split seeds, parameterized by `BASELINES_TAG`, `BASELINES_LABEL`, `BASELINES_EMBEDDING_SET` and `BASELINES_REQUIRE` (the `--require-labels` restriction) [[experiments.019-simb-multimodal.scripts.expression_baselines_split]]. The compute nodes lack the `GLIBC_2.32` the environment needs (the first submission failed on it), so the script runs inside the Rocky 9 container with `singularity exec`, as the training jobs do. Jobs 2423298 to 2423302 (44 tasks) on 2026-10-04: expression and proteome, both-label restricted and unrestricted. Results land under `results/baselines_split_<tag>_both_<label>[_full]/` in the IGB `019-baselines` worktree and have to be brought back by hand.
