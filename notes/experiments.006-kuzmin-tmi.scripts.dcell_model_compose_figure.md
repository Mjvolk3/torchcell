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
