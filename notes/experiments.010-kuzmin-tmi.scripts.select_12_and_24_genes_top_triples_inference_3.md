---
id: 2pvfhbfgk0uq6ip3hsnq8v6
title: Select_12_and_24_genes_top_triples_inference_3
desc: ''
updated: 1781029082786
created: 1781029082786
---

## 2026.06.09 - Selecting Panels From the Relaxed-Threshold Inference_3 Set Sized for the Jonckheere-Terpstra Test

This is the inference_3 counterpart of the panel-selection script, choosing the 12- and 24-gene panels that maximize coverage of the model's extreme triple predictions, but drawing from the relaxed-threshold inference_3 dataset designed so the validation experiment has the statistical power needed for the Jonckheere-Terpstra ordered-trend test (a ~0.04 fitness gap gives roughly 96% power at n=8). It exists because the stricter inference_2 set was too sparse to guarantee enough monotonic-path triples for that test, and the inference_3 table is far larger (465M+ rows), forcing a memory-efficient redesign.

### Specifics worth keeping

- All heavy data work uses PyArrow (sort, take, vectorized `is_in` filtering) on a memory-mapped ~12 GB table instead of an ~84-90 GB pandas load; only small subsets convert to pandas.
- Same greedy-plus-swap selection and Sameith-priority logic as inference_2, over identical panel/k sweeps.
- Additionally emits the full singles/doubles/triples construction tables for the k=200 panels (`singles_table_*`, `doubles_table_*`, `triples_table_*`), which the queried-data and fitness-comparison scripts consume.
- Outputs land in `results/inference_3/`; figures carry the `inference_3` suffix; `--plot-only` redraws from the saved CSV without the multi-GB load.
