---
id: gphr7rq8xfra1wyx30oce83
title: Compose_dango_si_figures
desc: ''
updated: 1788482551949
created: 1788482551950
---

## 2026.09.03 - Composing the DANGO reproduction SI figure

Script: `experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py`. Writes
`notes/assets/drawio/FigS-dango-reproduction.drawio`, which `make -C paper/nature-biotech fig`
exports to `paper/nature-biotech/figures/FigS-dango-reproduction.pdf` for the Supplementary Note
`note:dango-repro`.

- (a) DANGO schematic in the manuscript's notation, authored as draw.io cells in this script
  (palette fills, Arial, body text at the 8.3 ladder size, panel letters at 11.1 bold): the six
  STRING channels `A^(k)` as message-passing edges of one GraphSAGE encoder each, the
  reconstruction head with zero weight `lambda_k` (pretraining loss), the meta-embedding into `H`,
  the perturbation as row selection with no operator `T_psi`, the Hyper-SAGNN readout with the
  log-cosh loss, and a strip with the scheduled objective.
- (b) `dango_decreased_zeros.svg` from [[experiments.005-kuzmin2018-tmi.scripts.dango_construction_si]].
- (c) `dango_string_version_sweep.svg` and (d) `dango_string_version_curves.svg` from
  [[experiments.005-kuzmin2018-tmi.scripts.dango_string_version_sweep]].

Image cells are placed at the exact size each SVG declares (100 draw.io units per inch), so the
export is WYSIWYG. The script refuses to write a figure wider than 708 or taller than 669 units
(180 x 170 mm). Rerun it after regenerating any panel; the `.drawio` is never edited by hand.

## 2026.09.03 - White-cross layout (layout only, no content change)

Author review asked for one explicit layout convention across the composed SI figures so no
panel letter sits over a y-axis label or a neighbor's title: `COL_GAP = 12` (3 mm),
`ROW_GAP = 22` (5.5 mm), `TOP_STRIP = 16`, letters at `(panel_x, row_top)` in the strip above
each row. The schematic now starts at y = 16 under the strip, panels b and c sit one
`COL_GAP` apart (second column at 358.5), and `FULL_WIDTH` is 705 (the schematic's readout
column is 147 wide instead of 148 so the objective strip ends at the same edge). Figure
705 x 630 units = 179.1 x 160.0 mm; exported PDF 179.6 x 160.2 mm; `check-figures.sh` and
`drawio_font_band.py --check` pass. Panel placement in `editing.pdf` is for the author to
review.

## 2026.09.04 - Second author review: STRING releases first, MathJax schematic, curves kept

Re-lettered to a STRING releases, b schematic, c decreased zeros, d sweep, e curves, all five
within 170 mm (705 x 651 units = 179.1 x 165.2 mm; exported PDF 179.6 x 165.4 mm).

- (a) is `notes/assets/images/010-kuzmin-tmi/graphs_string_releases.svg` from
  `experiments/010-kuzmin-tmi/scripts/graph_statistics.py` (owned by the graphs note), embedded at
  its declared half width; it motivates the lambda rule of (c), and the graphs note references it as
  `\suppfig{fig:dango-repro}a`.
- (b) is now a half-width vertical pipeline (346 x 205 units, the size of panel a) of six stage
  boxes, each a plain-text bold heading (Arial 8.3) over one line of real LaTeX: the model sets
  `math="1"` and every equation is a `$$...$$` label at fontSize 7 (MathJax renders about 1.19x, so
  ~6 pt on the page; `drawio_font_band.py` reads it as 5.04 pt, on the ladder). Rows B and E split
  into two boxes (encoder -> reconstruction head; readout -> interaction loss). Every `\sum` is
  prefixed `\textstyle` so limits stay inline; box height 29.1, heading strip 13, gap 6. The words
  that used to overflow the gray boxes (lookup size, GraphSAGE layers, what lambda weights, the
  static/dynamic embeddings, the three schedules) moved to the caption. MathJax rendered in the
  headless PDF export (glyphs as vector paths; verified by rendering the PDF and reading it).
- (c) legend now sits in an opaque white box (see
  [[experiments.005-kuzmin2018-tmi.scripts.dango_construction_si]]).
- (d), (e) unchanged panels.

`check-figures.sh` and `drawio_font_band.py --check` pass (sizes 7 / 8.3 / 11.1 only).
