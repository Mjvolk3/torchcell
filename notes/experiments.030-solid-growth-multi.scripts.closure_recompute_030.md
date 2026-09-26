---
id: 70lnuoki5qwunoa0xn5tq0o
title: Closure_recompute_030
desc: ''
updated: 1790358615010
created: 1790358615010
---

## 2026.09.25 - What the script does

Recomputes the trigenic score of every Kuzmin triple in the 030 build from the build's own
fitness and digenic entries, under the published identity tau = f_ijk - f_ij f_k - eps_ik - eps_jk
and the Kuzmin-first `LabelPolicy` with the triple's own year promoted. Two stages, cache under
`$DATA_ROOT/data/torchcell/experiments/030-solid-growth-multi/closure/`:

- `scan`: every triple, the doubles whose gene pair lies inside some triple, every single; one row
  per stored entry with dataset, temperature, perturbation markers, value, sd, n, p and the strain
  identifier when the entry's perturbations share one; plus the query pair, array gene and query
  strain of each triple from `torchcell.data.label_table.triple_roles`.
- `analyze`: f_ij chosen by `LabelPolicy.select_double` (the double-mutant query strain the screen
  used, matched on the tm token) beside f_ij with the query-strain entries excluded (the 029
  reading) and the symmetric form, per screen and per stratum (all, deletion arrays only, triples
  whose query double is in the build). Reference rows for 025 and 029 are read from their result
  files and recorded with sha256.

Outputs in `experiments/030-solid-growth-multi/results/`: `closure_030_by_screen.csv`,
`closure_030_summary.json` (rows, coverage counts, reference provenance), `t10-030-closure.tex`.
Launcher `scripts/gh_closure_recompute_030.slurm` (`STAGE`, `BUILD`, `CACHE`, `LIMIT`, `LABEL`,
`REF_025` in the environment). Smoke test on the 029 store with 20,000 triples: slurm 2818, 0.516
on Kuzmin 2018 against the published 0.511 on all 57,451, the two f_ij readings identical because
that store carries no query-strain double. Findings and the table:
[[experiments.030-solid-growth-multi]].
