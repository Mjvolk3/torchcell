---
id: 251px62b1wg7rsqjf3yxr8h
title: Dataset_size_histograms_palette
desc: ''
updated: 1783996985385
created: 1783996985385
---

## 2026.07.13 - Showing what the fitness splits contain, in the paper palette

The classical-ML benchmark's numbers are only trustworthy if the train/val/test splits sample the same fitness range and the same mix of single, double, and triple deletions. This is the split-composition histogram, recolored to the repo palette so it can join the classical-ML SI figure as a dataset-construction panel rather than reading as a leftover exploratory plot.

- **Reads** the same cached one-hot splits as the gene-count bar plot: `$DATA_ROOT/data/torchcell/experiments/smf-dmf-tmf-traditional-ml/one_hot_gene/sum_{1e03,1e04,1e05}/{all,train,val,test}/{X,y}.npy`.
- **Plots** two histograms per split from the inverted one-hot: `num_genes_deleted` (row sum, i.e. perturbation order 1--3) and the fitness label `y`, 30 bins.
- **Writes** 12 PNGs to `ASSET_IMAGES_DIR` (`notes/assets/images/`) as `smf-dmf-tmf-traditional-ml_{size}-{split}-histogram_palette.png`.
- **Only delta vs `dataset_size_histograms.py`:** `color="#BD8800"` (Base-primary gold) and the `_palette` suffix. Same standard gap as its bar-plot sibling: PNG at default size, not yet a true-size SVG panel ([[torchcell.utils.utils]]).
