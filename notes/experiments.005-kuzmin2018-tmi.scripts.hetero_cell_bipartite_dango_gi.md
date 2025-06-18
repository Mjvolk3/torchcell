---
id: e4mjp4xhnn3oxv8q34l4a68
title: Hetero_cell_bipartite_dango_gi
desc: ''
updated: 1749757466522
created: 1748992674368
---
## ICLoss NaN Issue Summary (2025.06.10)

### Root Cause

The NaN issue occurs when `is_weighted_phenotype_loss: true` due to problematic weight calculation:

1. **Weight Formula Issue**: The formula `weights = [1 - v/sum(phenotype_counts.values()) for v in phenotype_counts.values()]` can produce:
   - Zero weights when a phenotype represents 100% of data
   - Very small weights when a phenotype dominates (e.g., 90% → weight = 0.1)
   - This causes numerical instability in ICLoss

2. **Gene Interaction Data Characteristics**:
   - Likely has severe class imbalance
   - Most interactions might be neutral (close to 0)
   - Few strong positive/negative interactions
   - This imbalance leads to problematic weights

### Evidence

- Model works perfectly with `is_weighted_phenotype_loss: false`
- Model works with ICLoss in standalone mode (uses `weights = torch.ones(1)`)
- NaN appears immediately in first forward pass when weighted loss is enabled

### Temporary Solution

Set `is_weighted_phenotype_loss: false` in the config to use uniform weights

### Proper Fix Needed

Replace the weight calculation with a more stable approach:

- Use inverse frequency weighting: `weight = total_count / phenotype_count`
- Apply sqrt or log scaling to moderate extreme weights
- Ensure minimum weight threshold (e.g., 0.1)
- Normalize weights appropriately

### Related Files

- `/experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi.py` (lines 348-360)
- `/experiments/005-kuzmin2018-tmi/conf/hetero_cell_bipartite_dango_gi.yaml`
- `/torchcell/losses/multi_dim_nan_tolerant.py` (WeightedSupCRCell)
- `/torchcell/models/hetero_cell_bipartite_dango_gi.py`
