---
id: js9kfgxwcmumus734e8pwpu
title: dango-si-completion
desc: ''
updated: 1792994261365
created: 1792994261365
---

## 2026.10.26

- [x] Filled panels e to g of `FigS-dango-full-dataset` (predicted vs measured, absolute error vs |tau| beside the CGT, meta-embedding channel weights v9.1 vs v12.0) from the served IGB checkpoints, copied to GilaHyper and scored from the 006 build's side files because the LMDB no longer deserializes [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_inference]] [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_panels_efg]] [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_compose_figure]]
- [x] ReZero fact check at inference: the v9.1 checkpoint's gates sit at 0.07 but removing the attention branch drops validation Pearson from 0.367 to 0.079 (v12.0: 0.354 to 0.337); the small gate is not a shut branch. Written into the Note `note:dango-full`.
- [ ] DANGO ablation arms on the 006 build, STRING v9.1 (baseline, no reconstruction loss, plain residual, both): GilaHyper jobs 6442 to 6444 pending behind the 043 sweep, 6445 held; W&B project `torchcell_006-kuzmin2018-tmi_dango_ablation` [[experiments.006-kuzmin-tmi.scripts.dango_ablation]]
- [ ] Add the Bachlechner 2020 ReZero reference and the DANGO reference key to `references.bib` (owner curation)
- [x] Owner review: panel f (error vs |tau| beside the CGT) dropped as redundant with Fig. 2d; Supplementary Fig. 9 re-flowed as a 2 x 3 grid of third-width panels (a to f, channel weights now f), 179.9 x 114.0 mm [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_panels_ef]]

## 2026.10.27

- [x] DANGO ablation arms into `note:dango-full`: Supplementary Fig. `fig:dango-ablation` (`FigS-dango-ablation`, 179.9 x 56.5 mm, one row a to c), Supplementary Table `tab:dango-ablation-arms`, and the paragraph "Ablation of the reconstruction loss and the ReZero gate". One seed per arm: without the ReZero gate validation r reaches 0.35 at epoch 6 (baseline 13), best 0.3701 (baseline 0.3654), test 0.3784 (0.3711); without the reconstruction loss it reaches 0.35 at epoch 12 but stalls there and reaches 0.36 only at epoch 126 (baseline 33), best 0.3611, test 0.3705. The both-off arm (6445) has no run [[experiments.006-kuzmin-tmi.scripts.dango_ablation_si]] [[experiments.006-kuzmin-tmi.scripts.dango_ablation_compose_figure]]
