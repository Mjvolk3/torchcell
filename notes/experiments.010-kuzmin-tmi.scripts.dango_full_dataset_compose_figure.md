---
id: 90pwtai1e41c8ssl89kko9u
title: Dango_full_dataset_compose_figure
desc: ''
updated: 1788477900338
created: 1788477900338
---

## 2026.09.03 - Composing FigS-dango-full-dataset

`experiments/010-kuzmin-tmi/scripts/dango_full_dataset_compose_figure.py` writes `notes/assets/drawio/FigS-dango-full-dataset.drawio` from the three true-size panel SVGs of [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_si]] (a: training curves, b: best validation Pearson per run, c: convergence epochs) and three reserved boxes (d, e, f) for panels that need a trained DANGO checkpoint and inference on the validation split, which cannot run on the Mac (gilahyper down, no GPU). Each box's label states the checkpoint, dataset LMDB, and script that fill it:

- d: predicted vs measured trigenic interaction, DANGO STRING v9.1 run `014mprap`, best checkpoint under `DATA_ROOT/models/checkpoints/compute-3-3-1941704_ff855402.../014mprap-best-*.ckpt`, 006 dataset LMDB, `experiments/006-kuzmin-tmi/scripts/dango.py` with `regression_task.execution_mode=inference`.
- e: absolute error vs |tau|, DANGO (`014mprap`) beside CGT (`c7671wgj-best-pearson-epoch=24` checkpoint under `compute-3-3-2036902_bd9e6c66...`), needs both prediction dumps (`dango.py` inference and `equivariant_cell_graph_transformer_eval.py`).
- f: per-STRING-channel weight of DANGO's meta-embedding attention, v9.1 (`014mprap`) vs v12.0 (`9jpfy547`), needs a forward hook on `torchcell.models.dango.Dango`.

Layout: two columns of 88 mm panels, 180 x 145 mm total (709 x 571 draw.io units). Panel letters fontSize 11.1 bold lowercase; placeholder labels fontSize 8.3. Export with `"/Applications/draw.io.app/Contents/MacOS/draw.io" -x -f pdf --crop -o paper/nature-biotech/figures/FigS-dango-full-dataset.pdf notes/assets/drawio/FigS-dango-full-dataset.drawio` (exported page 181.0 x 146.4 mm; `check-figures.sh` and `drawio_font_band.py --check` pass).

## 2026.09.03 - White-cross layout (layout only, no content change)

Same convention as the other composed SI figures: `COL_GAP = 12` (3 mm), `ROW_GAP = 22` (5.5 mm), `TOP_STRIP = 16`, letters at `(panel_x, row_top)` in the strip above each row; the second column is at `half + COL_GAP` (358.5) and the placeholder boxes d-f fill their whole cell from the panel's left edge (previously inset by 14 units to make room for the letter). Figure 705 x 619 units = 179.0 x 157.2 mm; exported PDF 179.9 x 157.6 mm; `check-figures.sh` and `drawio_font_band.py --check` pass. Panel placement in `editing.pdf` is for the author to review.

## 2026.09.04 - Data-effect panel added; c narrowed; placeholders at third width

Layout: row 1, (a) curves and (b) best per run at half width; row 2, (c) convergence at third
width beside (d) the data-effect panel at wide width (57.8 + 3 + 118.9 mm tile the row); row 3,
(e)-(g) the reserved boxes (predicted vs measured, absolute error vs |tau|, meta-embedding channel
weights; unchanged text, re-lettered from d-f) at third width, 140 units tall. Same white-cross
constants. Figure 708 x 609 units = 179.7 x 154.8 mm; exported PDF 180.3 x 155.2 mm (the
third + wide row is 707.9 units, inside the 709 cap and the +2 mm grace); `check-figures.sh` and
`drawio_font_band.py --check` pass.

## 2026.09.05 - Data-effect panel at third width; reserved boxes re-flowed

Author review: (d) was too big. It is now third width (57.8 mm, from
[[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_si]]), so row 2 is (c) convergence, (d) data
effect, and (e) the first reserved box, three thirds; row 3 holds (f) and (g) at half width (140 units
tall, same text). The script refuses a data-effect SVG that is not third width. Same white-cross
constants; every letter in its strip. Figure 707 x 609 units = 179.5 x 154.8 mm; exported PDF
179.9 x 155.2 mm; `check-figures.sh` and `drawio_font_band.py --check` pass.

## 2026.10.26 - Panels e to g filled

The three reserved boxes are now image cells for the panels of [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_panels_efg]], which reads the checkpoint scoring of [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_inference]]: (e) predicted against measured tau, third width, closing row 2 beside (c) and (d); (f) absolute error against |tau| for DANGO beside the CGT and (g) the per-gene meta-embedding channel weights, v9.1 against v12.0, sharing row 3 at half width. The `PLACEHOLDERS` dict, `PLACEHOLDER_STYLE` and `placeholder_cell` are gone, and the script now refuses an e panel that is not third width and a b, f or g panel that is not half width. Row 3 takes its height from the f and g SVGs (50 mm each; 52 mm would have put the figure at 171.2 mm, over the 170 mm cap). White-cross constants unchanged.

Figure 707 x 666 units = 179.5 x 169.2 mm; exported PDF 179.9 x 169.3 mm with the headless recipe (`env -u DBUS_SESSION_BUS_ADDRESS timeout 180 dbus-run-session -- xvfb-run -a $DRAWIO notes/assets/drawio/FigS-dango-full-dataset.drawio --password-store=basic --no-sandbox --disable-gpu --disable-dev-shm-usage -x -f pdf --crop -o paper/nature-biotech/figures/FigS-dango-full-dataset.pdf`, PDF checked to be newer than a stamp taken before the call). `drawio_font_band.py --check` reports the 7 panel letters at 7.99 pt and nothing else; `check-figures.sh` passes; the thesis `common/drawio_text_clearance.py --check` reports all 7 labels clear every line by 0.4 mm. Fonts in the export: Arial, Arial Bold, Arial Italic, plus a LiberationSerif subset holding only the space glyph (draw.io's whitespace inside an embedded image). The figure and its caption fit on one page of `editing-nature-biotech.pdf` (page 47 in the 2026.10.26 build); the review page is `paper/nature-biotech/review/FigS-dango-full-dataset.pdf` (ignored by git).
