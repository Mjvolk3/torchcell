---
id: xo6hmaelh7nsf7b3uxd9exf
title: Visual_regression
desc: ''
updated: 1739316854226
created: 1739316838369
---
For general plots at end of run

## 2026.11.17 - No seaborn; umap imported only in plot_umap

`plot_distribution` drew two `seaborn.histplot(kde=True, stat="density")` layers. They are now `_density_histogram`, plain matplotlib: `ax.hist(density=True)` with the same `bins`, plus a `scipy.stats.gaussian_kde` line (Scott's bandwidth, 200 points between the series' minimum and maximum, histplot's `cut=0`), skipped for a zero-variance series as histplot skips it. Against seaborn on 500 float32 normal draws the bars differ by at most 3e-7 and the KDE line by 8e-8 (float32 versus seaborn's float64 cast); on the 9-point test fixture both are identical.

`umap` (which loads numba and llvmlite) is imported inside `plot_umap`, so importing `Visualization` needs neither umap-learn nor seaborn; `test_importing_the_module_loads_neither_umap_nor_seaborn` checks that in a fresh interpreter.
