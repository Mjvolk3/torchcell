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
