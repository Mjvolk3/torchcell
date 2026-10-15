---
id: a1ltsh7ehekhqbugqphc9bs
title: Scaling_plot_catalog_030
desc: ''
updated: 1791805097544
created: 1791805097544
---

## 2026.10.12 - The plot catalog of the 030 scaling grid

Script: `experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030.py`. One typed spec (`PlotSpec`: id, family, title, axes, series, read, W&B keys, the rows of the grid it feeds on, status, what it shows) for every plot the queued grid can feed: 16 plots in four families, surface S1-S4, compute C1-C4, diagnostic D1-D6, pair P1-P2. Nine the grid feeds in full, seven in part (a stated limit narrows the claim: ten-epoch full-pool points, one seed, four points on the D axis).

Writes the catalog table of `notes-tex/030-scaling-plots/` (`tables/t1-catalog.tex`) and `results/scaling_plot_catalog_030.json`, which `notes/assets/drawio/030-scaling-plots.gen.py` reads for its panel titles, so the table and the four sketch figures name the same plots the same way. Nothing in it is a result.

```bash
PYTHONPATH=$PWD python experiments/030-solid-growth-multi/scripts/scaling_plot_catalog_030.py
```

The companion study document is `notes-tex/030-scaling-study/` ([[experiments.030-solid-growth-multi.scripts.scaling_grid_table_030]]); the readout the plots draw on is [[experiments.030-solid-growth-multi.scripts.scaling_readout_030]].
