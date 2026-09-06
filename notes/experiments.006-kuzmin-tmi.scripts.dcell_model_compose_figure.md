---
id: oaitnrec8srvpcks1erilrt
title: Dcell_model_compose_figure
desc: ''
updated: 1788478118043
created: 1788478118043
---

## 2026.09.03 - FigS-dcell-model composition

Writes `notes/assets/drawio/FigS-dcell-model.drawio` (702 x 389 draw.io units = 178.3 x 98.9 mm; exported PDF 179.2 x 99.5 mm) and is exported to `paper/nature-biotech/figures/FigS-dcell-model.pdf` by `make -C paper/nature-biotech fig` or by hand with `draw.io -x -f pdf --crop`. Panel a is the DCell-in-TorchCell schematic authored as mxGraph XML in the script (palette slots 1-6, Arial, font ladder 8.3 body / 9.7 headers / 11.1 panel letters; `drawio_font_band.py --check` passes). Panels b-d are the true-size SVGs from [[experiments.006-kuzmin-tmi.scripts.dcell_model_go_stats]], placed at exact size, three `third`-width panels across the 180 mm row. The two lines of measured numbers on the schematic (59,986 annotation rows; 2,655 subsystems, 3,208 edges, 13 strata, 20.6 M parameters) come from that script's `dcell_model_size.csv`.

Rerun after regenerating any panel; the `.drawio` is overwritten, never hand-edited.

## 2026.09.03 - Panel a rebuilt around the real DAG, equations as MathJax, white-cross layout

Author review: panel a was too wordy, the toy DAG should be the real ontology, the equations should be real LaTeX, and letters must never sit over a y-axis label. The script was rewritten:

- Panel a is now the true-size `dcell_model_go_dag.svg` from [[experiments.006-kuzmin-tmi.scripts.dcell_model_go_stats]] (118.9 x 69 mm, the whole filtered DAG with one triple deletion highlighted) plus a 227-unit column of boxes to its right: a "perturbation enters as data" chip row (ten gene states, three zeroed in red) with the gene-state rule, the subsystem equations, the root readout, the auxiliary head, and the loss. Nearly all of the old prose moved into the figure caption in `si-note-dcell-model.tex`.
- Equations are `$$...$$` labels typeset by MathJax: the model carries `math="1"`, and the headless `draw.io -x -f pdf` export renders them as vector paths (verified with draw.io 31.3.1; `pdffonts` shows only Arial because the math is outlined). Calibration: MathJax renders about 1.19x the cell `fontSize` (cap height of `\mathrm{H}` against Arial `H`, measured on a 4x PNG export), so math cells are typed at `fontSize=7` and print at ~6 pt; `drawio_font_band.py` reads them as 5.04 pt, on the ladder. At `8.3` the math would print at ~7.1 pt, over Nature's maximum.
- Layout constants shared with the other three composed SI figures: `COL_GAP = 12`, `ROW_GAP = 22`, `TOP_STRIP = 16`; every letter sits in the strip at `(panel_x, row_top)`. Row 1 is panel a (DAG + equations, 16 + 272 units), row 2 the three `third`-width panels b-d at 12-unit gaps (3 x 227.6 + 24 = 706.8 units). Figure 707 x 483 units = 179.6 x 122.7 mm; exported PDF 179.9 x 123.1 mm; `check-figures.sh` and `drawio_font_band.py --check` pass (8 cells at 7, 16 at 8.3, 4 letters at 11.1).

## 2026.09.04 - Loss on one line; the column ends where panel a ends

Author review: the loss box hung below panel a's bottom edge. `equations()` now takes the DAG panel's height and gives the loss box whatever remains, with `alpha = 0.3` typeset on the same line as the loss (`\mathcal{L} = \ldots,\quad \alpha = 0.3`), and `main()` asserts that the column's bottom equals `y1 + h_dag` (271.65 units, the 69 mm DAG panel). Figure 707 x 483 units = 179.6 x 122.7 mm as before; the DAG panel itself carries the second review's changes ([[experiments.006-kuzmin-tmi.scripts.dcell_model_go_stats]]).

## 2026.09.05 - Third author review: DANGO box colors, panel a flush with the column, clear letter strip

Three changes to the composition:

- **Box colors by role, the DANGO scheme.** `ROLE_COLOR` maps the roles of the DANGO schematic (`FigS-dango-reproduction` panel b, `experiments/005-kuzmin2018-tmi/scripts/compose_dango_si_figures.py`) to the palette pairs it uses: input data and the learned encoder stage orange (`#D79B00`/`#FFE6CC`), the merged embedding purple (unused here), the perturbation's entry yellow (`#D6B656`/`#FFF2CC`), the readout blue (`#6C8EBF`/`#DAE8FC`), a head with its own loss red (`#B85450`/`#F8CECC`), the combined objective gray (`#666666`/`#F5F5F5`). DCell's boxes take the same roles: "Perturbation enters as data" yellow (was gray), "Subsystem t" orange (was yellow), "Root readout" blue (was orange), "Auxiliary head" red (was purple), "Loss" gray. Box style also matches DANGO's `spacingLeft=3;spacingRight=3`.
- **Panel a flush with the column.** The DAG SVG is now 80 mm tall (314.96 units) with its visible content flush with the image edges (axes frame at the top, legend at the bottom; see [[experiments.006-kuzmin-tmi.scripts.dcell_model_go_stats]]), and `equations()` sizes the column to exactly that height: the boxes' natural heights (74 / 62 / 30 / 30 / 36 at 6-unit gaps = 260 units) plus an equal share of the remainder (11 units each), so the column's top and bottom coincide with the DAG frame and legend. The letter `a` sits in the 16-unit `TOP_STRIP` above both; `Canvas.box` and `Canvas.image` now refuse any `y < TOP_STRIP`, so nothing can enter a letter strip.
- Figure 707 x 526 units = 179.6 x 133.7 mm (exported PDF 179.9 x 134.0 mm); `check-figures.sh` and `drawio_font_band.py --check` pass (7 math cells at 7, 16 at 8.3, 4 letters at 11.1).
