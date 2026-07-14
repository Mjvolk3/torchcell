---
id: c87qjql0rn61gvoruugwd80
title: Traditional_ml Plot_svr_palette
desc: ''
updated: 1783996982226
created: 1783996982226
---

## 2026.07.13 - SVR control: the ceiling is the representation, not the model class

A random forest failing on gene interactions could be blamed on the model. This SVR panel removes that escape route - same 17 gene encodings, same 4 aggregations, same 1e3-1e5 sizes, an entirely different learner, the same low ceiling - and it is the publication-grade rendering of that control for the interaction (002) side of the classical-ML supplementary figure.

- **Reads** cached wandb pulls `experiments/002-dmi-tmi/results/svr/combined_df_{mse,spearman}_{1e3,1e4,1e5}.csv`. `main(is_overwrite=True)` re-pulls from `zhao-group/torchcell_002-dmi-tmi_trad-ml_svr_{1e03,1e04,1e05}`; ships with `is_overwrite=False` so re-rendering is offline.
- **Writes** deduplicated frames to `experiments/002-dmi-tmi/results/svr/deduplicated_combined_df_*.csv`, and panels to `notes/assets/images/002-dmi-tmi_SVR_{1e03,1e04,1e05}_{mse,spearman}_{spearman,pearson,mse}_{add_cv,no_cv}_palette.svg` (`ASSET_IMAGES_DIR`) via `savefig_true_size_svg` from [[torchcell.utils.utils]].
- **Changed from the non-palette sibling:** the same treatment applied to the random-forest palette script - green-free 17-series palette, metrics trimmed to `spearman` / `pearson` / `mse`, figure 12x14 in -> 2.276x6.5 in (2.276 in = 57.8 mm = `PANEL_WIDTHS_MM["third"]`), fixed `FEATURE_ORDER` (reversed) replacing the alphabetical sort, in-plot group headers replacing long y-tick labels, mutually distinct hatches (`///` `...` `+++` `xxx`), 0.5 pt boxed frame, no in-figure title.
- Panels are deliberately kept interchangeable with the random-forest panels (same width, ordering, hatches, and colors) so the two learners can be stacked side-by-side in draw.io without re-scaling.

Example panel (1e4, spearman selection criterion):

![](./assets/images/002-dmi-tmi_SVR_1e04_spearman_spearman_no_cv_palette.svg)

### Ordered draw.io palette, line-face bars, tenth gridlines

Colors now come from the single ordered `PLOT_PALETTE` / `PLOT_PALETTE_FILL` in [[torchcell.utils.utils]] (green-free, primaries-first, draw.io `(line, fill)` pairs matching Fig 1) instead of a local hex list, so every plot shares one palette. Bars are drawn with the line/border color on the face and the light fill as the lighter test member of the validation/test pair; hatch and edges are solid black -- the previous `alpha=` on the whole patch grayed them, so alpha now applies only to a face color when a lighter bar is needed. Correlation axes gain tenth (0.1) gridlines labelled every 0.2. See CLAUDE.md "Figure & Plotting Standards".
