---
id: 6a35txadolkpsn2czhyrzny
title: Dcell_training_compose_figure
desc: ''
updated: 1788478281564
created: 1788478281565
---

Composes `notes/assets/drawio/FigS-dcell-training.drawio` from the four true-size panel SVGs of [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]] plus a lettered placeholder box. Script: `experiments/006-kuzmin-tmi/scripts/dcell_training_compose_figure.py`.

## 2026.09.03 - Layout and export

- Two rows of two 88 x 52 mm panels (a, b; c, d) at 6-unit gaps, then (e) a dashed placeholder box (fontSize 8.3, prints at 6 pt) for the per-operation profiler breakdown of a DCell training step, which needs a `torch.profiler` run of `dcell.py` on a cluster GPU node. Whole figure 709 x 491 draw.io units = 180 x 125 mm.
- Panel letters fontSize 11.1 bold lowercase. `drawio_font_band.py --check` passes (one 8.3 label, five 11.1 letters).
- Export: `"/Applications/draw.io.app/Contents/MacOS/draw.io" -x -f pdf --crop -o paper/nature-biotech/figures/FigS-dcell-training.pdf notes/assets/drawio/FigS-dcell-training.drawio` gives a 181.0 x 126.3 mm page; `check-figures.sh` passes (within the 2 mm grace).
- The draw.io file is overwritten on every run; never hand-edit it.

## 2026.09.03 - White-cross layout

Author review: panel letters must never sit over a y-axis label or a neighbor's title. The script now uses the layout constants shared by all four composed SI figures: `COL_GAP = 12` (3 mm), `ROW_GAP = 22` (5.5 mm), `TOP_STRIP = 16`, letters at `(panel_x, row_top)` in the strip above each row, second column at `half_w + COL_GAP` (so the figure is 705 units wide, not flush to 709), and the placeholder box (e) spans the full width from x = 0 with its letter in the strip. Figure 705 x 539 units = 179.0 x 137.0 mm; exported PDF 179.6 x 137.6 mm; `check-figures.sh` and `drawio_font_band.py --check` pass.

## 2026.09.04 - Three rows: the placeholder becomes the CPU profile, plus a data-effect panel

Second author review. The layout is now a loop over three rows of two half-width panels: (a, b) as before at 52 mm; (c, d) at 50 mm, with d the redesigned stage table of [[experiments.006-kuzmin-tmi.scripts.dcell_training_wandb]]; (e) `dcell_training_cpu_profile.svg` from [[experiments.006-kuzmin-tmi.scripts.dcell_training_cpu_profile]] and (f) `dcell_training_data_effect.svg`, both 48 mm. The placeholder cell and its style are gone. Same constants (`COL_GAP = 12`, `ROW_GAP = 22`, `TOP_STRIP = 16`, letters at `(panel_x, row_top)`). Figure 705 x 651 units = 179.0 x 165.2 mm; exported PDF 179.2 x 165.4 mm, under the 170 mm cap; `check-figures.sh` and `drawio_font_band.py --check` pass (six letters at 11.1, no other text cells).
