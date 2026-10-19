---
id: erh67ylrlpslslhtdwl4c0s
title: Primer_043_tables
desc: ''
updated: 1792445303674
created: 1792445303674
---

## 2026.10.19 - Tables of the 043 datasets primer

Writes the LaTeX tables of `notes-tex/multimodal/043-datasets-primer/tables/` from
`results/store_census.json`, `results/cross_dataset_correlation.json`,
`results/within_dataset_ceilings.json` and two 019 ceiling files on the
`multimodal-phenotype-retrospective` worktree (`expression_ceiling_all.json` for Caudal,
Messner, Zelezniak and Nadal-Ribelles; `morphology_noise_ceiling.json` for Ohya). The
sha256 of every input goes to `tables/provenance.json`. Media names are read from up to
3,000 records of each dev-tree LMDB. Run with `make tables` in the document directory or

```bash
python experiments/043-expression-proteome-morphology/scripts/primer_043_tables.py
```
