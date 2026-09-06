---
id: prnk01a2z1b1rag4f5gdqgq
title: Fba_baseline_compose_figure
desc: ''
updated: 1788665931978
created: 1788665931978
---
Composes `notes/assets/drawio/FigS-yeast9-fba.drawio`, the figure of Supplementary Note `note:fba`, from `results/fba_baseline_si/stats.json` and the panel SVGs of [[experiments.007-kuzmin-tm.scripts.fba_baseline_si]]. Script: `experiments/007-kuzmin-tm/scripts/fba_baseline_compose_figure.py`.

## 2026.09.05 - Layout

- **a** is a draw.io schematic of the pipeline (six boxes, left to right: Yeast9 and medium, gene-reaction rules, deletion sets, FBA growth, fitness proxy, interaction of Fig. 2a). Equations are MathJax (`math="1"`, `$$...$$`, `fontSize=7` so they print at about 6 pt); every number in the boxes is read from `stats.json`, none is typed by hand. Box colors follow the DANGO/DCell schematic roles (input orange, perturbation yellow, readout blue, derived purple, score red).
- **b**-**e** are the true-size panels (`fba_baseline_tau`, `fba_baseline_fitness`, `fba_baseline_growth_bands`, `fba_baseline_coverage`), three third-width panels on row 2 and one on row 3.
- **f** is a dashed, lettered placeholder: "[placeholder: rerun with corrected medium]", the same pipeline on the Kuzmin 2018 final-selection medium (SD/MSG synthetic medium, glutamate nitrogen, 0.2% amino-acid supplement lacking His/Arg/Lys/Ura, 2% glucose, 26 C). Not run; it reserves the space for b and c recomputed on that medium.
- White-cross layout (COL_GAP 12, ROW_GAP 22, TOP_STRIP 16); extent 707 x 580 units (179.6 x 147.2 mm). Export with `draw.io -x -f pdf --crop` to `paper/nature-biotech/figures/FigS-yeast9-fba.pdf` (179.9 x 147.8 mm, passes `check-figures.sh`; `drawio_font_band.py --check` clean: 7 / 8.3 / 11.1 only).

## 2026.09.06 - Re-lettered to a-g after author feedback

- Row 3 is now **e** (`fba_baseline_evaluable`: genes, doubles, and triples Yeast9 can score), **f** (`fba_baseline_landscape`: measured interactions by coverage class, log counts), and **g**, the dashed placeholder, third-width. The old "|tau| > 1e-3" annotations left the bars; those facts are in b's caption.
- The placeholder's first line is in the palette red (`#B85450`, `<font color>` inside the HTML label, same 8.3 = 6 pt size, the treatment of FigS-dcell-training panel e): "Rerun required: FBA used the model's default ammonium minimal medium; the screen used SD/MSG with amino-acid supplement at 26 C." The rest of the text stays black.
- Panel a's "Deletion sets" box now says the triples are Kuzmin 2018 and 2020; the numbers are unchanged (they are read from `stats.json`). Extent still 707 x 580 units; PDF 179.9 x 147.8 mm; size and font audits pass.
