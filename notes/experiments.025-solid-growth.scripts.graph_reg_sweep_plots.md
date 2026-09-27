---
id: zrfiats364pdgiugbueop5c
title: Graph_reg_sweep_plots
desc: ''
updated: 1790475563979
created: 1790475563979
---

## 2026.09.26 - Figures of the sweep

Three figures from the readout's CSVs into `$ASSET_IMAGES_DIR/025-solid-growth/` (true-size SVG + PNG, timestamped copies): `graph_reg_sweep_ladder` (2 x 2, full width: accuracy across the ladder at both readings, edge recall at degree, divergence, gradient-norm ratio), `graph_reg_sweep_curves` (validation Pearson, training Pearson, validation point loss by epoch, mean +- sd over seeds), `graph_reg_sweep_control` (random-graph control at lambda 1e-3, half_plus width). Tick labels are standalone mathtext (`$10^{-5}$`); prose labels spell lambda in decimals because Arial has no superscript minus or nabla glyph, and a missing glyph renders as a box in the PNG. The notes-tex document converts the SVGs with `make plots`.

![](./assets/images/025-solid-growth/graph_reg_sweep_ladder.svg)

![](./assets/images/025-solid-growth/graph_reg_sweep_curves.svg)

![](./assets/images/025-solid-growth/graph_reg_sweep_control.svg)

Readings: [[experiments.025-solid-growth.graph-reg-sweep]].
