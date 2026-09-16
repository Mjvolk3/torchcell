---
id: k6mr30ae8qatnh4007w8xhj
title: '38'
desc: ''
updated: 1789538541983
created: 1789538541983
---

## 2026.09.16

- [x] Gene essentiality moved out of 027 into `experiments/028-gene-essentiality` (worktree `feat/028-gene-essentiality`, W&B `torchcell_028_gene_essentiality`); overview in [[experiments.028-gene-essentiality]]
- [x] FBA single-gene-deletion essentiality on four media [[experiments.028-gene-essentiality.scripts.fba_essentiality_baseline]]: AUROC on our 195 genes 0.654 to 0.666, medium moves it by 0.01
- [x] Split-file protocol and genome-wide labels [[experiments.028-gene-essentiality.scripts.build_essentiality_splits]]
- [ ] Wave 12 (jobs 2073 to 2093): port check, split file, genome labels, weight-decay ladder, batch 32
- [ ] Flux-arm cells (gradients through the metabolic module) after the wave-12 read
- [ ] notes-tex document `028-gene-essentiality` with the FBA-by-medium, AUROC-by-split and gradient figures
