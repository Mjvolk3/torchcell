---
id: l3blo3llnz99o23zc5bmuxq
title: Gh_cgt_030
desc: ''
updated: 1792636621576
created: 1792636621576
---

## 2026.10.21 - GilaHyper launcher for a pool-store 030 arm

`gh_cgt_030.slurm <config> [overrides]` runs one 030 per-entry arm on GilaHyper from a pool store read in place on `$DATA_ROOT` (default the strict S3Q store; `STORE_REL` picks another, `GPUS` the torchrun process count). It is the IGB launcher without the RAM-disk staging and the Delta launcher without the node-local copy.

First use: the affine source-calibration test. Two 1-GPU jobs on the 1/4 S3Q fraction (244,402 training records, 509,458 interaction entry rows), seed 1, 40 epochs, batch 256:

| job | config | source enters as |
|---|---|---|
| 6112 `030-tok-q4-s1` | `cgt_030_s3_q_tok_embfit_004` | 8-dim token in both head inputs (control) |
| 6113 `030-aff-q4-s1` | `cgt_030_s3_q_tok_affine_005` | per-source scale and shift on the head outputs |

Read at matched epoch: pooled `val/gene_interaction/Pearson` (both arms can learn the between-screen offset), `val/token/TmiKuzmin2018Dataset/...` and `.../2020/...` (the within-screen read), `val/crosstoken/...`, and the learned `cal/gene_interaction/<source>/{scale,shift}` of the affine arm.
