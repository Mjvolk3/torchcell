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
  replication by release; 180 x 72 mm. Superseded the same day by `FigS-dango-reproduction`
  (composed by [[experiments.005-kuzmin2018-tmi.scripts.compose_dango_si_figures]]); the file
  was trashed and is not written by this script.

`make -C paper/nature-biotech fig` exports them to `paper/nature-biotech/figures/FigS-*.pdf`;
`check-figures.sh` passes both (181.0 x 171.8 mm and 181.0 x 73.4 mm, inside the +2 mm grace).
Rerun this script after regenerating any panel; do not edit the `.drawio` by hand.

## 2026.09.03 - Two graph figures, aligned grid, DANGO figure handed over

The composer now places panels on a two-column grid: column 2 starts at the half-panel width plus
`COL_GAP` (8 units, 2 mm), so panel b sits flush beside panel a instead of at the far right edge;
rows share a height (set in `graph_statistics.py`) so axes tops and bottoms align; a row of unequal
widths (e, third + f, wide) is right-aligned to column 2's right edge.

- `FigS-graph-attention-priors.drawio`: (a) sizes, (b) degree CCDF, (c) Jaccard, (d) containment,
  (e) edge multiplicity, (f) structure; 178.0 x 159.0 mm (export 178.9 x 160.5 mm).
- `FigS-graph-attention-priors-2.drawio`: (a) shared pairs, (b) recurring hubs, (c) regulatory vs
  TFLink, (d) STRING releases in time order; 178.0 x 117.5 mm (export 178.9 x 118.9 mm).

`FigS-dango-string-versions` is no longer composed here; the DANGO release sweep belongs to the
DANGO reproduction note. Both new figures pass `check-figures.sh` and
`drawio_font_band.py --check` (only the 11.1 panel letters).

## 2026.09.03 - White-cross layout, three-row figures

Author feedback: panel b's letter hung over panel a's axis title, and panels c to e took too much
room. The composer now lays every figure out on explicit white gutters (the "white cross") with
the letters inside them:

- `COL_GAP = 12` units (3 mm) between columns, `ROW_GAP = 22` units (5.5 mm) between rows,
  `TOP_STRIP = 16` units above the first row, and each letter at `(panel_x, panel_y - LETTER_STRIP)`
  with `LETTER_STRIP = 16`, so a letter is always 16 units above its own panel's top-left corner
  and never over a neighbor's labels or the panel's own y-axis label.
- Rows advance by the tallest panel in the row plus `ROW_GAP`; panels of one row share a height set
  in `graph_statistics.py` (`F1_ROW1_H, F1_ROW2_H, F1_ROW3_H = 54, 44, 50` mm; `F2_ROW_H, F2_ROW3_H
  = 52, 40` mm). Columns are placed left to right at their own widths, so a row of three
  third-width panels (3 x 227.6 + 2 x 12 = 707 units) and a row of two halves (2 x 346.5 + 12 = 705)
  both fill the 709-unit page.
- `FigS-graph-attention-priors.drawio`: (a) sizes, (b) degree CCDF; (c) Jaccard, (d) containment,
  (e) edge multiplicity at third width; (f) other components at full width; 707 x 643 units
  (179.5 x 163.2 mm; export 179.9 x 163.7 mm).
- `FigS-graph-attention-priors-2.drawio`: (a) shared pairs, (b) recurring hubs with SGD
  descriptions; (c) regulatory vs TFLink, (d) STRING releases; (e) structure at full width;
  705 x 627 units (179.0 x 159.2 mm; export 179.2 x 159.4 mm).

The height gate is `MAX_HEIGHT + HEIGHT_GRACE` = 669 + 8 units; both figures pass it,
`check-figures.sh`, and `drawio_font_band.py --check`.
