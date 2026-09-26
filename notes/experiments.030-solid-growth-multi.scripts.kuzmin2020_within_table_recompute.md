---
id: 7kc40d4jryiljdc049i21ev
title: Kuzmin2020_within_table_recompute
desc: ''
updated: 1790387158411
created: 1790387158412
---

## 2026.09.25 - What the script does

Recomputes the Kuzmin 2020 trigenic score on the raw Tables S1 and S3 with every term taken
inside the row's own table: f_ijk, f_ij and f_k from the trigenic row, eps_ik and eps_jk from the
single-mutant control queries' rows (query `<gene>+YDL227C`) against the same array strain in the
same table. Reports, per table and pooled: the within-table form, the controls pooled over both
tables (a screen-level match), and f_k averaged over the tables. Reads the dmf_kuzmin2020 raw
mirror; writes `results/kuzmin2020_within_table_recompute.{csv,json}`. Finding and its consequence
for the loaders: [[experiments.030-solid-growth-multi]].
