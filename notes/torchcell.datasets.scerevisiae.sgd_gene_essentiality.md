---
id: i0sc3g3kg9yqa2wcywlmwjr
title: Sgd_gene_essentiality
desc: ''
updated: 1721782831924
created: 1721598684492
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
