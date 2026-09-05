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
