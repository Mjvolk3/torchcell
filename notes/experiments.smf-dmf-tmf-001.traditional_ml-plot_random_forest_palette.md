---
id: nh2k8w3mptbyybx9clc49vr
title: Traditional_ml Plot_random_forest_palette
desc: ''
updated: 1783996990000
created: 1783996990000
---

## 2026.07.13 - Random-forest benchmark bars authored at true Nature panel size

These are the RF panels of the classical-ML SI figure -- 17 encodings x 4 pooling representations, validation against test -- and the original chart was a 12x14 in exploratory canvas with 20 pt type. Placing that in a 180 mm page means rescaling, which silently shrinks the fonts below Nature's 6 pt floor. This version authors the panel at its final print size and exports true-size SVG, so what the script emits is what the figure carries, 1:1.

- **Reads cached results only:** `experiments/smf-dmf-tmf-001/results/random_forest/combined_df_{mse,spearman}_{1e3,1e4,1e5}.csv`. The W&B pull is gated behind `main(is_overwrite=True)` and now defaults to `False`, so regeneration is offline. Dedup keeps one config per (encoding, representation) by the selection criterion and writes `deduplicated_*.csv` back to the same results dir.
- **Writes** true-size SVGs to `ASSET_IMAGES_DIR` (`notes/assets/images/`) as `smf-dmf-tmf-001_Random_Forest_{1e03,1e04,1e05}_{mse,spearman}_{spearman,pearson,mse}_{add_cv,no_cv}_palette.svg`, via `savefig_true_size_svg` ([[torchcell.utils.utils]]) so they import at true mm in draw.io.
- **Panel geometry:** `figsize=(2.276, 6.5)` in = **57.8 x 165 mm**, i.e. `PANEL_WIDTHS_MM["third"]` wide (three across a 180 mm page) and under the 170 mm `MAX_HEIGHT_MM` cap. Arial 6 pt everywhere, 0.5 pt frame and ticks, `svg.fonttype: none` so text stays editable.
- **Restyling forced by that width:** encoding names moved out of the y-tick margin into left-aligned in-plot group headers (the long labels were eating the panel); the in-plot title dropped (the figure caption names the panel); hatches made mutually distinct (`///` pert_sum, `...` pert_mean, `+++` intact_sum, `xxx` intact_mean) so the four representations stay separable on thin bars; metrics cut from six to spearman/pearson/mse; the ad-hoc 20-color list replaced by the green-free Extended 17-series.
- **Where it lands:** these `_palette` SVGs are the panels composed in `notes/assets/drawio/Fig7-Traditional-ML-justification-of-CGT.drawio.svg` (the SI classical-ML figure); the paper uses the $10^5$, spearman-selected pair. Keep in lockstep with its SVR twin, [[experiments.smf-dmf-tmf-001.traditional_ml-plot_svr_palette]].
