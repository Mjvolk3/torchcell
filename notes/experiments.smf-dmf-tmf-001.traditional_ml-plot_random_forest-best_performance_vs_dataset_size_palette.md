---
id: fc1csdohvrt284o8p77qwnp
title: Traditional_ml Plot_random_forest Best_performance_vs_dataset_size_palette
desc: ''
updated: 1783996991457
created: 1783996991457
---

## 2026.07.13 - The scaling panel: does more data rescue any encoding?

This panel carries the second claim of the classical-ML Supplementary Note -- fitness keeps climbing with data while gene interactions flatline regardless of encoding -- and that claim is only visible if the fitness and interaction progressions can be read side by side on **identical axes**. The palette version is what makes the pairing possible: a shared fixed y-scale, a detachable legend, and true-size SVG output that drops into the draw.io canvas at true mm.

- **Reads** `experiments/smf-dmf-tmf-001/results/random_forest/random_forest_processed_df_{1000,10000,100000}.csv`. Per encoding it takes the best run by `val_r2` (validation-selected, no test leakage) and plots that run's test metric at each of the three dataset sizes on a log x-axis -- one line per encoding, 17 lines.
- **Writes** to `ASSET_IMAGES_DIR` (`notes/assets/images/`): `smf-dmf-tmf-001_node_embedding_performance_{test_spearman,test_pearson,test_mse}[_shared_0_1][_no_legend]_palette.svg` plus the standalone `node_embedding_legend_palette.svg`, all via `savefig_true_size_svg` ([[torchcell.utils.utils]]).
- **Two canvases per metric:** with the inline 17-entry legend (4.6 x 2.4 in, right ~40% reserved so the plot keeps real width) and a plot-only `_no_legend` (2.6 x 2.2 in, ~66 x 56 mm). Two no-legend panels plus one shared legend fit a 180 mm row -- which is why `create_legend()` emits the legend separately, cropped to the legend artist's own window extent rather than to the (invisible) axes box.
- **`_shared_0_1`** pins `test_pearson` to $y \in [0, 1]$ so fitness and interaction sit on the same scale. **This `shared_pearson_ylims` dict must stay identical to the twin script in `experiments/002-dmi-tmi/`** or the comparison is a lie.
- **Deliberate choices:** minor log gridlines are off -- each line has only three measured sizes, so the connecting segments are a trend guide, not interpolation; metrics cut from twelve to three (`test_spearman`, `test_pearson`, `test_mse`); 6 pt Arial throughout (the old `labelsize=14` tick override was the text-mis-sizing bug, do not reintroduce); green-free Extended 17-series.

### Ordered draw.io palette (single source of truth)

Line colors now come from `PLOT_PALETTE` in [[torchcell.utils.utils]] (green-free, primaries-first, draw.io line colors matching Fig 1) instead of a local list, so the progression lines, the bar panels, and the standalone encoding legend all share one palette. See CLAUDE.md "Figure & Plotting Standards".
