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
