---
id: t5vo3nb5iqltmzsgo091pvh
title: Traditional_ml Plot_random_forest_palette
desc: ''
updated: 1783996979101
created: 1783996979101
---

## 2026.07.13 - Publication panel: no gene encoding rescues random forests on interactions

The classical-ML supplementary note claims the bottleneck for gene interactions is the *representation*, not the learner: 17 gene encodings crossed with 4 aggregations all land at the same low Spearman. This is the publication-grade rendering of that evidence for the interaction (002) side - the original script emitted a screen-sized PNG that cannot go into a Nature figure, so this variant re-emits the same result as a true-size, standard-conforming panel.

- **Reads** cached wandb pulls `experiments/002-dmi-tmi/results/random_forest/combined_df_{mse,spearman}_{1e3,1e4,1e5}.csv`. `main(is_overwrite=True)` re-pulls from `zhao-group/torchcell_002-dmi-tmi_trad-ml_random-forest_{1e03,1e04,1e05}`; it ships with `is_overwrite=False` so a re-render never depends on the network.
- **Writes** deduplicated frames back to `experiments/002-dmi-tmi/results/random_forest/deduplicated_combined_df_*.csv`, and panels to `notes/assets/images/002-dmi-tmi_Random_Forest_{1e03,1e04,1e05}_{mse,spearman}_{spearman,pearson,mse}_{add_cv,no_cv}_palette.svg` (`ASSET_IMAGES_DIR`) via `savefig_true_size_svg` from [[torchcell.utils.utils]].
- **Changed from the non-palette sibling:** green-free 17-series palette (was a 20-color pastel set); metrics trimmed 6 -> 3 (`spearman`, `pearson`, `mse`); figure 12x14 in -> 2.276x6.5 in (2.276 in = 57.8 mm = `PANEL_WIDTHS_MM["third"]`); encodings pinned to a fixed `FEATURE_ORDER` (reversed, so the panel reads top-down in canonical order) instead of an alphabetical sort; y-tick labels replaced by in-plot left-aligned group headers to reclaim width at column size; hatches made mutually distinct (`///` `...` `+++` `xxx`) so the four aggregations stay separable on thin bars; markers and error bars shrunk; in-figure title dropped (it belongs in the caption); frame and ticks at 0.5 pt.
- `add_cv` variants add cross-validation error bars plus NaN markers, and reserve extra top headroom for the two additional legend rows.

Example panel (1e4, spearman selection criterion):

![](./assets/images/002-dmi-tmi_Random_Forest_1e04_spearman_spearman_no_cv_palette.svg)
