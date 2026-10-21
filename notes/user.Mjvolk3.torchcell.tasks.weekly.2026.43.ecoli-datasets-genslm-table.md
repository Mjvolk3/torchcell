---
id: 569z36wtv1y3xy37410m2no
title: ecoli-datasets-genslm-table
desc: ''
updated: 1792614191130
created: 1792614191130
---

## 2026.10.21

- [x] `docs/ecoli-datasets-genslm-table` worktree: notes-tex document `notes-tex/database/ecoli-datasets/` with the table of every served E. coli dataset (68 datasets, 4,561,865 records, snapshot 2026-10-21 of the LMDB probe) and the short list for a gene-sequence encoding benchmark: core four, one per task family (Goodall 2018 essentiality, Nichols 2011 chemical genomics, Schmidt 2016 proteome, Babu 2014 gene interaction), second tier of four (Li 2014, PRECISE-1K, Campos 2018, Wang 2018 gene fitness), every number read from a store by [[experiments.database.scripts.build_ecoli_genslm_benchmark_table]]; `PaperTable.to_latex(table_env="longtable")` added for the A4 page ([[experiments.database.ecoli-datasets]])
