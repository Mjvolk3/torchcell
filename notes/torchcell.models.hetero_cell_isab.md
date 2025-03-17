---
id: bo1lmoehw7j51xtebq8etkh
title: Hetero_cell_isab
desc: ''
updated: 1741158292172
created: 1741149073493
---
## 2025.03.04 - Model Example Architecture

```python
Model architecture:
HeteroCell(
  (gene_embedding): Embedding(6607, 64)
  (reaction_embedding): Embedding(4881, 64)
  (metabolite_embedding): Embedding(2534, 64)
  (preprocessor): PreProcessor(
    (mlp): Sequential(
      (0): Linear(in_features=64, out_features=64, bias=True)
      (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (2): ReLU()
      (3): Dropout(p=0.0, inplace=False)
      (4): Linear(in_features=64, out_features=64, bias=True)
      (5): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (6): ReLU()
      (7): Dropout(p=0.0, inplace=False)
    )
  )
  (convs): ModuleList(
    (0-2): 3 x HeteroConv(num_relations=4)
  )
  (global_aggregator): SortedSetTransformerAggregation(
    (aggregator): SetTransformerAggregation()
    (proj): Identity()
  )
  (prediction_head): Sequential(
    (0): Linear(in_features=64, out_features=64, bias=True)
    (1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (2): ReLU()
    (3): Dropout(p=0.0, inplace=False)
    (4): Linear(in_features=64, out_features=64, bias=True)
    (5): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (6): ReLU()
    (7): Dropout(p=0.0, inplace=False)
    (8): Linear(in_features=64, out_features=64, bias=True)
    (9): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    (10): ReLU()
    (11): Dropout(p=0.0, inplace=False)
    (12): Linear(in_features=64, out_features=2, bias=True)
  )
)
Parameter count: 1167493
```

## 2025.03.04 - Model Mermaid

![](./assets/images/mermaid_pdf/hetero_cell_isab_mermaid.pdf)