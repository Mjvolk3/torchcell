---
id: 20d7r1c3u6y6b6zzfwueo56
title: '30'
desc: ''
updated: 1784675068699
created: 1784675068699
---

## 2026.07.21

- [x] Overhauled the SGA colony segmenter to a grid-constrained gitter-style method (6-sided gel-boundary gate + one shared accept predicate) that removes the phantom-boundary artefacts [[torchcell.sga.image]]
- [x] Re-analysed the run-2 volume/timepoint sweep on full-resolution 72 h captures and added Spearman ordering to the reference scatter [[experiments.010-kuzmin-tmi.12_panel_crispr_fitness_assay]]
- [x] Green-lit Cellpose for colony segmentation after a clean zero-shot web-demo result, and drafted the workstation/rsync + integration plan [[experiments.019-echo-crispr-array.cellpose-segmentation-plan]]
- [ ] Move segmentation to Cellpose on gila: rsync the run-2 images, install cellpose, wire a `seg_method='cellpose'` branch, and validate against the classical numbers #high [[experiments.019-echo-crispr-array.cellpose-segmentation-plan]]
- [ ] Re-examine the SVR interaction fits at random (d=1000), where a CV s.d. of 0.383 against a mean of 0.458 sits in the same cell that produced a diverged MSE #medium [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [ ] Commit the gilahyper phenotype/gzip script so the phenotype-dataset and integrated-graph tables come from a script instead of a hand transcription; they are the last numbers in the paper without a generating script, and one of them is sourced from a scratch note #high [[paper.information-accounting]]
- [ ] Reconcile the paper-facing classical-ML plot script with the new figure standard, since its PNG output conflicts with the palette SVG route that now feeds the classical-ML figure #medium [[experiments.smf-dmf-tmf-001.traditional_ml-plot_paper]]
