---
id: i0sc3g3kg9yqa2wcywlmwjr
title: Sgd_gene_essentiality
desc: ''
updated: 1721771439661
created: 1721598684492
---
## 2024.07.23

We have duplicates but since they come from different experiments they might be taken under different environment conditions or gene knockout conditions etc.

```python
len([i['experiment']['genotype']['perturbations'][0]['systematic_gene_name'] for i in dataset])
1329
len(set([i['experiment']['genotype']['perturbations'][0]['systematic_gene_name'] for i in dataset]))
1140
```
