---
id: 5q5vx8qt8xwhcn098i2oh7x
title: Store_against_plan
desc: ''
updated: 1790664850261
created: 1790664850261
---

## 2026.09.29 - Thirteen tables, one per plan claim

Reads the cell table and the 031 result files, and writes `results/*.csv`. `--plan-results`
is required because experiment 031 is on its own branch (PR #436); the commit it was read at
is in `results/store_against_plan_provenance.json` (bf1884c46). Rerun without rebuilding the
cell table through `gh_store_against_plan.slurm` (slurm 3004, 3 m 4 s, 11.4 G resident).
The findings are in [[experiments.033-env-chemgen-pooled]] and typeset in
`notes-tex/033-post-query-analysis`.
