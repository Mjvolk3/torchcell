---
id: hl40po9soz6mghzvcymg9h4
title: Graph
desc: ''
updated: 1697763177830
created: 1697604307905
---
## Not using MultiDiGraph

While the graph structure is naturally a `multidigraph`, we don't model it this way because the conversion functions in `PyG` are for `nx.Graph` and `nx.DiGraph`.

```python
def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
```

## Supported Graphs

| $\textbf{Dataset Name}$ | genotypes  | environment | phenotype (label)             | label type             | description                | supported |
| :---------------------- | :--------- | :---------- | :---------------------------- | :--------------------- | :------------------------- | :-------: |
| baryshnikovna2010       | 6,022      | 1           | $\text{smf}$                  | global                 | growth rate                |     ✔️     |
| YeastTract              | ~6,000     | 1,144       | $\text{smf}$                    | global       | growth rate                |           |

YeastTract
356,180
