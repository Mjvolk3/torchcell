---
id: jejpj6sz9tibe9rmcrsw7z2
title: Graph
desc: ''
updated: 1745968856680
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

## Adding GO Union from GAF

```python
>>> sum([len(graph.G_raw.nodes[node]['go_details']) for node in graph.G_raw])
197396
```

After running `compare_go_terms`

```bash
5577 out of 6607 nodes have matching GO term sets.
A total of 306 new GO terms would be added to G_raw from G.
```

306 accounts for 0.16% of 197,396 and it takes a bit of care to combine these two different annotations since their fields are slightly different and/or have different formats. I will leave this task for a future date.

## 2025.04.29

The reason for `GeneMultiGraph` is that `pyg` `from_networkx` only takes type `nx.Graph` and `nx.Digraph`, if we make multigraph just a list of these objects then we just loop over them and use function to get multigraph object.
