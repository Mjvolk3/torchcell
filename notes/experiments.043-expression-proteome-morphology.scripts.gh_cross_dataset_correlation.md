---
id: poy0s2mwlh1xy7loidr28cc
title: Gh_cross_dataset_correlation
desc: ''
updated: 1792437331815
created: 1792437331815
---

## 2026.10.19 - Launcher

Runs [[experiments.043-expression-proteome-morphology.scripts.cross_dataset_correlation]]
under slurm (partition `main`, 48 GB, 8 CPUs, 4 h). `STAGE` selects `all` (default),
`extract` or `analyze`. The 116 moving CalMorph features are read from the
`multimodal-phenotype-retrospective` worktree, and the output records their sha256
(`312fd7a6...`). Jobs: 5890 (all, 18 min 21 s), 5891 and 5892 (analyze only, about
1 min each, figure layout fixes).
