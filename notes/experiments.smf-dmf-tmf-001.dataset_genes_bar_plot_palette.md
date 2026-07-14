---
id: kqpkoybdog8xmec417uijrz
title: Dataset_genes_bar_plot_palette
desc: ''
updated: 1783996983814
created: 1783996983814
---

## 2026.07.13 - Showing which genes the fitness splits actually cover

The classical-ML Supplementary Note argues that knockout fitness is saturated by gene identity alone, and a reader can only judge that claim if they can see how much of the genome each dataset size perturbs and whether the splits cover the same genes. This is that gene-coverage bar chart, recolored to the repo palette so it can sit inside the classical-ML SI figure instead of clashing with the benchmark panels next to it.

- **Reads** the cached one-hot splits under `DATA_ROOT`: `data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene/sum_{1e03,1e04,1e05}/{all,train,val,test}/{X,y}.npy`. Inverts the intact one-hot (`X -> 1 - X`) to a per-gene deletion count and annotates the number of genes perturbed on each panel.
- **Writes** 12 PNGs (3 sizes x 4 splits) to `ASSET_IMAGES_DIR` (`notes/assets/images/`) as `smf-dmf-tmf-traditional-ml_gene-count_{size}-{split}-bar_palette.png`. They are not on disk right now; regenerating requires the cached splits under `DATA_ROOT`.
- **Only delta vs `dataset_genes_bar_plot.py`:** bars drawn in Base-primary gold `#BD8800` (from `notes/assets/images/color-palette.svg`) and the `_palette` output suffix.
- **Known standard gap:** still a 10x6 in PNG at default type size, not a `PANEL_WIDTHS_MM`-sized true-size SVG. Convert it through [[torchcell.utils.utils]] `savefig_true_size_svg` before it goes into a paper figure.
