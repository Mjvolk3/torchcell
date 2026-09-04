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
