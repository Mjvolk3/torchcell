---
id: hygtna7f9avv8mrporbohks
title: Linear_probes
desc: ''
updated: 1795561811848
created: 1795561811848
---

## 2026.11.24 - Linear probes

Ridge regression on five frozen gene representations for every task and split; penalty chosen on validation mean squared error; test unlocked once per split. Output: `results/probes/<task>.<feature>.csv`. Rendered by [[experiments.050-ecoli-benchmark.scripts.results_page]].

## 2026.11.25 - Ordinary least squares replaces ridge

Owner decision (2026-11-25): one standard, hyperparameter-free evaluation for every model, so the probes are ordinary least squares (minimum-norm solution when the fit is not unique), with no selection on validation. 760 runs; 172 board rows replaced on staging. Wide representations score lower than under ridge on the single-gene tasks (growth rate, mean test Pearson: codon frequency 0.059, ESM-2 0.027, Nucleotide Transformer v1 0.027 with test R2 -12.7).
