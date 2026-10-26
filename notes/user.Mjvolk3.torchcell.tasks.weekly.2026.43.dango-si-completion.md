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
