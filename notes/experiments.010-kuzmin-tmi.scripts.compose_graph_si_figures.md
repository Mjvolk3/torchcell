---
id: i0kwf6ha3q7o6ray9441jc1
title: Compose_graph_si_figures
desc: ''
updated: 1788466933982
created: 1788466933982
---

## 2026.09.03 - Composing the two graph SI figures

Script: `experiments/010-kuzmin-tmi/scripts/compose_graph_si_figures.py`. Reads the true-size panel
SVGs from [[experiments.010-kuzmin-tmi.scripts.graph_statistics]] and
[[experiments.005-kuzmin2018-tmi.scripts.dango_string_version_sweep]] and writes two plain `.drawio`
files with image cells at exact physical size plus 8 pt bold panel letters (draw.io `fontSize=11.1`):

- `notes/assets/drawio/FigS-graph-attention-priors.drawio`: (a) sizes, (b) degree CCDF, (c) Jaccard,
  (d) containment, (e) edge multiplicity; 180 x 170 mm.
- `notes/assets/drawio/FigS-dango-string-versions.drawio`: (a) STRING release drift, (b) DANGO
  replication by release; 180 x 72 mm.

`make -C paper/nature-biotech fig` exports them to `paper/nature-biotech/figures/FigS-*.pdf`;
`check-figures.sh` passes both (181.0 x 171.8 mm and 181.0 x 73.4 mm, inside the +2 mm grace).
Rerun this script after regenerating any panel; do not edit the `.drawio` by hand.
