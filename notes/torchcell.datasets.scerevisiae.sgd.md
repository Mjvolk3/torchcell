---
id: 8jawh92k2f1y87z9b8vdlms
title: Gene_essentiality_sgd
desc: ''
updated: 1726426740612
created: 1726426684541
---
## 2024.07.23 - Gene Essentiality Duplicates

We have duplicates but since they come from different experiments they might be taken under different environment conditions or gene knockout conditions etc.

```python
len([i['experiment']['genotype']['perturbations'][0]['systematic_gene_name'] for i in dataset])
1329
len(set([i['experiment']['genotype']['perturbations'][0]['systematic_gene_name'] for i in dataset]))
1140
```

## 2024.07.23 - We Could Manually Check Meta Data From Yeast Gene Essentiality

We should be able to manually check 113 different datasets for meta data population after automated population of meta data from the paper pdf.

We could then use the automated procedure from this dataset, checked manually, on the synthetic lethality datasets [[2024.07.23 - We Will Not Be Able to Manually Populate Meta Data for all Synthetic Lethality and Synthetic Rescue Experiments|dendron://torchcell/torchcell.datasets.scerevisiae.syn_leth_db_yeast#20240723---we-will-not-be-able-to-manually-populate-meta-data-for-all-synthetic-lethality-and-synthetic-rescue-experiments]]

```python
len(set([i['experiment']['pubmed_id'] for i in dataset]))
113
```

## 2024.09.15 - Rename to Match other Datasets

We have been naming datasets with label(phenotype) and then source and date.

## 2026.10.01 - Issue #533 No overwrite=True and No Relative Fallback

- `process` read the gene directory from `os.environ.get("DATA_ROOT", "data")`, a relative fallback; it now uses `torchcell.graph.sgd.data_root()`, which refuses an unset `DATA_ROOT` with a named `ValueError`.
- `main` built the genome with `overwrite=True`, which races any concurrent reader of the shared genome build, and the dataset at the relative default root. It now passes `overwrite=False` and roots the dataset at `$DATA_ROOT/data/torchcell/gene_essentiality_sgd`.
- Left open (adapter scope): the content hash in `cell_adapter.py` excludes the publication, so records that differ only in publication collapse to one graph node (1,329 records to 1,140 nodes on the datasets page).
