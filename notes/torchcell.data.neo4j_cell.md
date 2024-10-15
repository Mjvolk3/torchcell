---
id: p64k8x1u54badkcybt1vcpn
title: Neo4j_cell
desc: ''
updated: 1728960279783
created: 1727139151784
---
## 2024.10.14 - Idea for Supporting Different Base Graphs

```python
def get_init_graphs(self, raw_db, genome):
    # Setting priority
    if genome is None:
        cell_graph = create_graph_from_gene_set(raw_db.gene_set)
    elif genome:
        cell_graph = create_graph_from_gene_set(genome.gene_set)
    return cell_graph
```