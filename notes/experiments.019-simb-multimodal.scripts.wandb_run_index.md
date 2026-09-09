---
id: jhzusd9iq75qwm1k2de554f
title: Wandb_run_index
desc: ''
updated: 1788913458464
created: 1788913458464
---

## 2026.09.08 - Every run behind the expression document, linked

Script: `experiments/019-simb-multimodal/scripts/wandb_run_index.py`. Reads the same result
files the sections of `notes-tex/019-simb-multimodal-expression/` read
(`short_budget_spread.json`, `round_leaderboards.csv`, `mech_round_readout.csv`,
`pearson_round_readout.csv`, `v10_grid_factorial.csv`), writes `results/wandb_run_index.json`
and the LaTeX table `tables/wandb_runs_019.tex` (one row per round and arm, every entry a
seed and its run id linked to the run page, final epoch in brackets). 74 runs: the eight v9
mask-schedule arms, 18 objective-round runs (fresh plus resume), 8 mechanism, 8
metric-aligned, 32 v10 grid. Building this table is what exposed the eight "replicates"
as eight arms; see [[experiments.019-simb-multimodal.expression-strand-retrospective]].
