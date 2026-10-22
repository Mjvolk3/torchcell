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

## 2026.10.21 - Moved to IGB mmli; GilaHyper keeps the smoke

The user's call: GilaHyper runs a smoke only, the comparison runs where the readout-token controls already run. Jobs 6112 and 6113 were cancelled at epoch 0 (about 20 minutes in) and replaced by `030-aff-smoke` 6114 (affine arm, 1/64 fraction, 2 epochs of 60 batches, 60 validation batches): the end-to-end check of the affine path through a validation epoch end, the `cal/` logging and a checkpoint.

On IGB mmli the two seed-2 readout-token jobs (2429089 Q at epoch 52, 2429091 R at epoch 0) were cancelled and the affine arms queued at seed 1 in their place, so each split carries a matched pair at one A100, batch 512, the full pool:

| split | readout token (control) | affine |
|---|---|---|
| Q | 2429088 `030-r1w2-q-emb-s1` (dhbxhb24) | 2430655 `030-r1w3-q-aff-s1` (`cgt_030_s3_q_tok_affine_005`) |
| R | 2429090 `030-r1w2-r-emb-s1` | 2430656 `030-r1w3-r-aff-s1` (`cgt_030_s3_r_tok_affine_006`) |

IGB worktree `~/projects/torchcell.worktrees/030-token-affine` at 6400bfca5. The controls lead by their head start (Q by about 52 epochs, R by about an hour), so the read is at matched epoch, not matched date.
