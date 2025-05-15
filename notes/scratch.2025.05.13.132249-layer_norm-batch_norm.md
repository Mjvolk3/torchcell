---
id: eaksmd8qz8acvac536ob7pu
title: 132249 Layer_norm Batch_nor
desc: ''
updated: 1747161983649
created: 1747160583179
---
```python
>>> import torch
... import torch.nn as nn
...
... # Input: 3 instances, 4 features each
... x = torch.tensor([
...     [1.0, 2.0, 3.0, 4.0],
...     [2.0, 3.0, 4.0, 5.0],
...     [3.0, 4.0, 5.0, 6.0],
... ])
...
... # BatchNorm1d: normalize over batch (dim=0), per feature
... bn = nn.BatchNorm1d(num_features=4, affine=False, track_running_stats=False)
... x_bn = bn(x)
...
... # LayerNorm: normalize per instance (dim=1)
... ln = nn.LayerNorm(normalized_shape=4, elementwise_affine=False)
... x_ln = ln(x)
...
... print("Input:\n", x)
... print("BatchNorm (per feature across batch):\n", x_bn)
... print("LayerNorm (per instance across features):\n", x_ln)
Input:
 tensor([[1., 2., 3., 4.],
        [2., 3., 4., 5.],
        [3., 4., 5., 6.]])
BatchNorm (per feature across batch):
 tensor([[-1.2247, -1.2247, -1.2247, -1.2247],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.2247,  1.2247,  1.2247,  1.2247]])
LayerNorm (per instance across features):
 tensor([[-1.3416, -0.4472,  0.4472,  1.3416],
        [-1.3416, -0.4472,  0.4472,  1.3416],
        [-1.3416, -0.4472,  0.4472,  1.3416]])
```

When the std is the same over the dim batch norm gives same value for each row and layer norm gives same value over each column. This means that if stds are the same we cannot differentiate instances using layer norm because rows are identical.

``` python
>>> import torch
...
... # Input tensor: (batch_size=3, features=4)
... x = torch.tensor([
...     [1.0, 2.0, 3.0, 4.0],
...     [2.0, 3.0, 4.0, 5.0],
...     [3.0, 4.0, 5.0, 6.0],
... ])
...
... print("Input:\n", x)
...
... # BatchNorm: normalize each column (feature) across the bat
... # ch
... mean_batch = x.mean(dim=0)
... std_batch = x.std(dim=0, unbiased=False)  # Use population 
... # std like nn.BatchNorm1d
...
... x_bn_manual = (x - mean_batch) / std_batch
...
... print("\nBatchNorm (manual):")
... print("Mean per feature:", mean_batch)
... print("Std per feature:", std_batch)
... print("Normalized:\n", x_bn_manual)
...
... # LayerNorm: normalize each row (instance) across its featu
... # res
... mean_layer = x.mean(dim=1, keepdim=True)
... std_layer = x.std(dim=1, keepdim=True, unbiased=False)
...
... x_ln_manual = (x - mean_layer) / std_layer
...
... print("\nLayerNorm (manual):")
... print("Mean per sample:\n", mean_layer)
... print("Std per sample:\n", std_layer)
... print("Normalized:\n", x_ln_manual)
Input:
 tensor([[1., 2., 3., 4.],
        [2., 3., 4., 5.],
        [3., 4., 5., 6.]])

BatchNorm (manual):
Mean per feature: tensor([2., 3., 4., 5.])
Std per feature: tensor([0.8165, 0.8165, 0.8165, 0.8165])
Normalized:
 tensor([[-1.2247, -1.2247, -1.2247, -1.2247],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.2247,  1.2247,  1.2247,  1.2247]])

LayerNorm (manual):
Mean per sample:
 tensor([[2.5000],
        [3.5000],
        [4.5000]])
Std per sample:
 tensor([[1.1180],
        [1.1180],
        [1.1180]])
Normalized:
 tensor([[-1.3416, -0.4472,  0.4472,  1.3416],
        [-1.3416, -0.4472,  0.4472,  1.3416],
        [-1.3416, -0.4472,  0.4472,  1.3416]])
>>>
```

When stds are different over dim we get unique values.

```python
>>> import torch
...
... # Asymmetric input to break symmetry
... x = torch.tensor([
...     [1.0, 100.0, 5.0, 7.0],
...     [10.0, 50.0, 6.0, 8.0],
...     [3.0, 10.0, 7.0, 9.0],
... ])
...
... # BatchNorm: normalize per feature (column) across batch (rows)
... mean_batch = x.mean(dim=0)
... std_batch = x.std(dim=0, unbiased=False)
... x_bn = (x - mean_batch) / std_batch
...
... # LayerNorm: normalize per row across features (columns)
... mean_layer = x.mean(dim=1, keepdim=True)
... std_layer = x.std(dim=1, keepdim=True, unbiased=False)
... x_ln = (x - mean_layer) / std_layer
...
... print("Input:\n", x)
... print("\nBatchNorm (normalize over batch, per feature):\n", x_bn)
... print("\nLayerNorm (normalize over features, per row):\n", x_ln)
Input:
 tensor([[  1., 100.,   5.,   7.],
        [ 10.,  50.,   6.,   8.],
        [  3.,  10.,   7.,   9.]])

BatchNorm (normalize over batch, per feature):
 tensor([[-0.9503,  1.2675, -1.2247, -1.2247],
        [ 1.3822, -0.0905,  0.0000,  0.0000],
        [-0.4319, -1.1770,  1.2247,  1.2247]])

LayerNorm (normalize over features, per row):
 tensor([[-0.6569,  1.7297, -0.5605, -0.5123],
        [-0.4660,  1.7268, -0.6853, -0.5756],
        [-1.5853,  1.0258, -0.0933,  0.6528]])
```
