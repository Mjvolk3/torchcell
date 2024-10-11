---
id: o5tw8l0zwoo59864tjt0iv5
title: Yaml
desc: ''
updated: 1726692160459
created: 1726691988229
---
## 2024.09.18 - Few Phenotype Nodes

We only get 7 phenotypes. This makes sense because they are deduplicated and we have an additional for reference.

```python
df['r.statistic_score'].unique()
array([0.9 , 0.8 , 0.5 , 0.25, 0.75, 1.  ])
len(df['r.statistic_score'].unique())
6
```
