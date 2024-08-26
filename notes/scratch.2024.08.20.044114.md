---
id: 66wf82a1gh93cvvfx5ro7mc
title: '044114'
desc: ''
updated: 1724147237943
created: 1724146884340
---
When node_embeddings 

```python
self.cell_graph
HeteroData(
  gene={
    num_nodes=6579,
    node_ids=[6579],
    x=[6579, 1],
  }
)
[type(i) for i in self.cell_graph['gene']]
[<class 'str'>, <class 'str'>, <class 'str'>]
[type(i) for i in self.cell_graph['gene'].values()]
[<class 'int'>, <class 'list'>, <class 'torch.Tensor'>]
```

Without `node_embeddings`

```python
self.cell_graph
HeteroData(
  gene={
    num_nodes=6579,
    node_ids=[6579],
    x=[6579, 0],
  }
)
[type(i) for i in self.cell_graph['gene'].values()]
[<class 'int'>, <class 'list'>, <class 'torch.Tensor'>]
```