---
id: j7ag2z1rgofx48iryne1mts
title: '24'
desc: ''
updated: 1749535026431
created: 1749503722566
---


- [ ] Add Gene Expression datasets
- [ ] For @Junyu-Chen consider reconstructing $S$? Separate the metabolic graph construction process into building both $S$ then casting to a graph... Does this change properties of S? You are changing the constrains but also changing the dimensionality of the matrix... → don't know about this... I think the the constrained based optimization won't be affected from the topological change much. It is mostly a useful abstraction for propagating genes deletions.
- [ ] #ramble Need to start dumping important experimental results into the experiments folder under `/experiments` - Do this for `004-dmi-tmi` that does not work
- [ ] Add concern about graph connectivity to [[Report 003-fit-int.2025.03.03|dendron://torchcell/experiments.003-fit-int.2025.03.03]]
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] Inquiry about web address for database.
- [ ] Export and `rsync` this to linked delta drive
- [ ] Mount drive and spin up database. Check if database is available on ports and over http.
- [ ] HeteroCell on String 12.0
- [ ] Contrastive DCell head on HeteroCell.
- [ ] Add morphology. Only Safari browser works. Respond to maintainers about solved problem of downloading database. Might want to store a backup.
- [ ] Morphology Random Forest Baseline
- [ ] Morphology animation ? for fun...

***

## 2025.06.09

- [x] Run 006 query. → Had to set graphs to null → 332,313 trigenic interactions
- [ ] Sync experiments for `005`, see if syncing crashes offline runs. Fastest run is on epoch 153.  → [[2025.06.03 - Launched Experiments|dendron://torchcell/experiments.005-kuzmin2018-tmi.results.hetero_cell_bipartite_dango_gi#20250603---launched-experiments]] jobs stopping will also be sign of cancellation. Keeping `slurm id: 1820177`, cancelling `slurm id: 1820176`, `slurm id: 1820165`. →
- [ ] After sync check if the gating of loss looks dramatic. If so we should parameterize concatenation.  
- [ ] Run Dango on `006` query
- [x] Debugged ICLoss NaN issue with hetero_cell_bipartite_dango_gi
- [ ]

1824013 - lr: 1e-4
lr: 1e-5

## 2025.06.10

- [ ] Fix issue with ddp on genome.

#TODO

## DDP Gene Set JSON Race Condition Issue (2025.06.10)

### Problem

When running with DDP (distributed data parallel), one of the processes encounters a `JSONDecodeError` when trying to read `gene_set.json`:

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

This occurs in `/torchcell/data/neo4j_cell.py` at line 404 when trying to load the gene set.

### Root Cause

This is a race condition in distributed training where:
1. Multiple processes are trying to access the dataset simultaneously
2. The `gene_set.json` file exists but appears empty to process 3
3. This suggests file system synchronization issues between processes

### Temporary Solutions

1. **Add rank-based delays**: Have non-zero ranks wait before accessing the dataset
2. **Use barrier synchronization**: Ensure all processes wait for rank 0 to finish setup
3. **Pre-create the dataset**: Ensure the dataset is fully processed before starting DDP training

### Proper Fix Needed

The `Neo4jCellDataset` needs better handling for concurrent access in DDP mode:
- Add file locking or process synchronization
- Ensure rank 0 completes all file writes before other ranks read
- Add retry logic with delays for file reads
- Consider using shared memory or broadcast for gene_set data

### Related Files

- `/torchcell/data/neo4j_cell.py` (lines 399-412)
- `/experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi.py`

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
