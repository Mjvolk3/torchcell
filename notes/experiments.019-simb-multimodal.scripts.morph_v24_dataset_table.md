---
id: mlicyplysdzwe5w7pyk7ttl
title: Morph_v24_dataset_table
desc: ''
updated: 1791694380116
created: 1791694380116
---

## 2026.10.11 - The morphology round's data as two tables

Reads the two stores (`fig3_core`, `fig3_morph_proteome`) through their label index and the split seed 0 cache, reproduces the training script's `require_modalities` filter per arm, and writes `tables/morph_v24_dataset.tex` (strain sets; the target and its references) and `results/morph_v24_dataset.json`. The per-arm rows (3,757 / 469 / 469 morphology-only; 1,161 / 132 / 146 expression-and-morphology; 3,464 / 416 / 401 proteome-and-morphology) are the counts the runs printed. Target: 116 moving CalMorph features of 278, ceiling 0.847 on that set (0.611 over all), ridge from the measured proteome 0.280 and from measured expression 0.324 on it, sequence baselines 0.086 / 0.088 on all 278.
