---
id: owd5bmp58fknlx3fpfubh6h
title: Test_go_topological_batching
desc: ''
updated: 1747610034907
created: 1747610020769
---
```python
(torchcell) michaelvolk@M1-MV torchcell % python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/test_go_topological_batching.py
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
<frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute
Loading genome and graph...
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2025-03-16) 43,544 Terms
Original GO graph has 5660 nodes and 6674 edges

Applying DCellGraphProcessor filters...
INFO:torchcell.graph.graph:Filtering result: 3193 GO terms removed (had < 5 contained genes)
INFO:torchcell.graph.graph:Filtering result: 4 redundant GO terms removed
INFO:torchcell.graph.graph:Filtering result: 25 GO terms and 1576 IGI gene annotations removed
  Original graph: 5660 nodes, 6674 edges
  Filtered graph: 2438 nodes, 2944 edges

After filtering, found 14 different levels in the GO hierarchy
Level -1: 1 nodes
Level 0: 3 nodes
Level 1: 15 nodes
Level 2: 148 nodes
Level 3: 376 nodes
Level 4: 505 nodes
Level 5: 508 nodes
Level 6: 393 nodes
Level 7: 285 nodes
Level 8: 138 nodes
Level 9: 35 nodes
Level 10: 20 nodes
Level 11: 10 nodes
Level 12: 1 nodes

Assigned nodes to 13 batches for parallel processing
First 10 batches:
  Batch 0: 1 nodes
  Batch 1: 3 nodes
  Batch 2: 567 nodes
  Batch 3: 474 nodes
  Batch 4: 366 nodes
  Batch 5: 314 nodes
  Batch 6: 266 nodes
  Batch 7: 203 nodes
  Batch 8: 119 nodes
  Batch 9: 58 nodes

Batch efficiency analysis:
  Total nodes: 2438
  Total batches: 13
  Average batch size: 187.54
  Maximum batch size: 567
  Minimum batch size: 1
  Parallel efficiency: 0.33
  Sequential ratio: 0.0053

Batch size distribution plot saved to: /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/go_batch_distribution_2025-05-18-18-11-30.png

All batches validated: Nodes within each batch have no dependencies on each other.

Level-based vs Batch-based grouping comparison:
  Total groups (levels vs batches): 14 vs 13
  Max group size (level vs batch): 508 vs 567
  Avg group size (level vs batch): 174.14 vs 187.54

Comparison plot saved to: /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/go_level_vs_batch_comparison_2025-05-18-18-11-30.png

Loading sample data to analyze actual graph processor operations...
DATA_ROOT: /Users/michaelvolk/Documents/projects/torchcell
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2025-03-16) 43,544 Terms
INFO:torchcell.graph.graph:Filtering result: 119 GO terms and 1921 IGI gene annotations removed
After IGI filter: 5541
INFO:torchcell.graph.graph:Filtering result: 106 redundant GO terms removed
After redundant filter: 5435
INFO:torchcell.graph.graph:Filtering result: 2780 GO terms removed (had < 4 contained genes)
After containment filter: 2655

Normalization parameters for gene_interaction:
  mean: -0.048011
  std: 0.053502
  min: -1.081600
  max: 0.000000
  q25: -0.061951
  q75: -0.015263
  strategy: standard
INFO:torchcell.datamodules.cell:Loading index from /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_seed_42.json
INFO:torchcell.datamodules.cell:Loading index details from /Users/michaelvolk/Documents/projects/torchcell/data/torchcell/experiments/005-kuzmin2018-tmi/001-small-build/data_module_cache/index_details_seed_42.json
Setting up PerturbationSubsetDataModule...
Loading cached index files...
Creating subset datasets...
Setup complete.
  0%|                                                                                                                               | 0/10001 [00:01<?, ?it/s]
Loaded dataset with 91050 samples and batch with 4 graphs
Processed GO graph in dataset has 2655 nodes
Mutant state tensor shape: torch.Size([239944, 3])
Number of unique GO terms in mutant_state: 2654
Number of unique genes in mutant_state: 6607

=== OPTIMIZATION STRATEGY SUMMARY ===
1. Original level-based approach would require 14 sequential processing steps
2. Topological batch approach requires 13 sequential processing steps
3. Topological batching reduces sequential steps by 7.1%

=== IMPLEMENTATION RECOMMENDATIONS ===
1. Replace the current sequential processing with batch processing:
   - Group GO terms by topological ranks (as done in this script)
   - Process each batch in a single GPU operation
   - Update the forward pass to handle batch-by-batch processing

2. Implementation approach:
   a. During model initialization, compute and store the batch assignments
   b. In forward pass, iterate over batches instead of individual nodes
   c. For each batch, concat inputs, process together, then split outputs
(torchcell) michaelvolk@M1-MV torchcell %                                 
```
