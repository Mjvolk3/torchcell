---
id: f0lv6y6pwi2pl2kixqweoob
title: Prot_v26_readout
desc: ''
updated: 1791900844009
created: 1791900844009
---

## 2026.10.13 - Readout of the proteome alignment round v26

Reads project `torchcell_019_prot_v26`: the S_prot arm (v21 small trunk, ProtT5, width 90, six layers, per-gene head) on the full fig3_proteome store at batch 128, 600 epochs, the reference against the latent alignment term ([[torchcell.losses.latent_align]]) at lambda 0.1 and 1 on z_S. The registered window is the mean of `val/proteome/pearson_per_feature` over epochs 500 to 599, declared with the launch on 2026.10.12 before any run was read; a run short of epoch 599 is partial. Each alignment cell is paired against the reference on the same split seed. Writes `results/prot_v26_readout.json` and `notes-tex/figure-3-gate/tables/prot_v26.tex`. Launched as stage `prot_v26` of `igb_expr_wave5.slurm` (cabbi job 2428296).
