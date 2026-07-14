---
id: q9ssz69m1m686tpken0ezb4
title: Traditional_ml Plot_svr_palette
desc: ''
updated: 1783996992863
created: 1783996992863
---

## 2026.07.13 - SVR benchmark bars, and one color per encoding across the whole figure

The SVR half of the classical-ML SI figure gets the same true-size treatment as its random-forest twin, and fixes a defect that would have been visible in the assembled figure: the original built its color map by zipping the palette to the **alphabetically sorted** encoding list, so an encoding could take one color in the RF panel and a different one in the SVR panel sitting right beside it. This version pins the canonical `FEATURE_ORDER` and derives colors from it, so a gene encoding keeps a single identity across every panel of Fig 7.

- **Reads cached results only:** `experiments/smf-dmf-tmf-001/results/svr/combined_df_{mse,spearman}_{1e3,1e4,1e5}.csv`; the W&B pull is behind `main(is_overwrite=True)` and is off by default. Dedup writes `deduplicated_*.csv` back to the same results dir.
- **Writes** `SVR_{1e03,1e04,1e05}_{mse,spearman}_{spearman,pearson,mse}_{add_cv,no_cv}_palette.svg` to `ASSET_IMAGES_DIR` via `savefig_true_size_svg` ([[torchcell.utils.utils]]). Note the fitness SVR files carry **no experiment prefix** (historical); the interaction twins in `experiments/002-dmi-tmi/` are prefixed `002-dmi-tmi_`.
- **Same panel contract as the RF twin:** `figsize=(2.276, 6.5)` in = 57.8 x 165 mm (`PANEL_WIDTHS_MM["third"]`, under the 170 mm cap), Arial 6 pt, 0.5 pt frame, in-plot group headers instead of y-tick labels, no in-plot title, distinct hatches per pooling representation, metrics cut to spearman/pearson/mse, green-free Extended 17-series.
- **Where it lands:** panels for `notes/assets/drawio/Fig7-Traditional-ML-justification-of-CGT.drawio.svg` (SI classical-ML figure, `fig:classical-ml`); paper uses the $10^5$ spearman-selected panel. Any style change here must be mirrored in [[experiments.smf-dmf-tmf-001.traditional_ml-plot_random_forest_palette]] -- the two are read side by side.

### Ordered draw.io palette, line-face bars, tenth gridlines

Colors now come from the single ordered `PLOT_PALETTE` / `PLOT_PALETTE_FILL` in [[torchcell.utils.utils]] (green-free, primaries-first, draw.io `(line, fill)` pairs matching Fig 1) instead of a local hex list, so every plot shares one palette. Bars are drawn with the line/border color on the face and the light fill as the lighter test member of the validation/test pair; hatch and edges are solid black -- the previous `alpha=` on the whole patch grayed them, so alpha now applies only to a face color when a lighter bar is needed. Correlation axes gain tenth (0.1) gridlines labelled every 0.2. See CLAUDE.md "Figure & Plotting Standards".
