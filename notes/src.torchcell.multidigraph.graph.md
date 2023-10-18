---
id: fpgsoq4b7jr8f8c2m7oh2jt
title: Graph
desc: ''
updated: 1697604409696
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
