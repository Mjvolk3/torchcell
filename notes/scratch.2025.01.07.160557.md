---
id: 1cvj5c73sjxtoz1hsb151dt
title: '160557'
desc: ''
updated: 1736287585678
created: 1736287559197
---
```python
unique_genes = set()
for edge in hypergraph.edges:
    unique_genes.update(hypergraph.edges[edge].properties['genes'])
unique_genes_list = sorted(unique_genes)
print(unique_genes_list)
```