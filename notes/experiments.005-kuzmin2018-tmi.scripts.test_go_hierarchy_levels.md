---
id: ldw263lc3uvubk8wzrjavt4
title: Test_go_hierarchy_levels
desc: ''
updated: 1747605913843
created: 1747605908439
---
```python
(torchcell) michaelvolk@M1-MV torchcell % python /Users/michaelvolk/Documents/projects/torchcell/experiments/005-kuzmin2018-tmi/scripts/test_go_hierarchy_levels.py
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
Loading genome and graph...
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2025-03-16) 43,544 Terms
GO graph has 5660 nodes and 6674 edges
Found 14 different levels in the GO hierarchy
Level -1: 1 nodes
Level 0: 3 nodes
Level 1: 16 nodes
Level 2: 239 nodes
Level 3: 682 nodes
Level 4: 1014 nodes
Level 5: 1391 nodes
Level 6: 1055 nodes
Level 7: 735 nodes
Level 8: 369 nodes
Level 9: 89 nodes
Level 10: 43 nodes
Level 11: 20 nodes
Level 12: 3 nodes

Total edges: 6674
Same-level edges: 531
Cross-level edges: 6143

Found cross-level connections!
Sample of cross-level edges (node1, node2, level1, level2):
  1. (GO:0000001, GO:0008150, 5, 0)
     - mitochondrion inheritance -> biological_process
  2. (GO:0000002, GO:0007005, 6, 5)
     - mitochondrial genome maintenance -> mitochondrion organization
  3. (GO:0007005, GO:0006996, 5, 4)
     - mitochondrion organization -> organelle organization
  4. (GO:0000011, GO:0007033, 6, 5)
     - vacuole inheritance -> vacuole organization
  5. (GO:0007033, GO:0006996, 5, 4)
     - vacuole organization -> organelle organization
  6. (GO:0000012, GO:0006281, 6, 5)
     - single strand break repair -> DNA repair
  7. (GO:0006281, GO:0006974, 5, 4)
     - DNA repair -> DNA damage response
  8. (GO:0000017, GO:0008150, 7, 0)
     - alpha-glucoside transport -> biological_process
  9. (GO:0000018, GO:0008150, 7, 0)
     - regulation of DNA recombination -> biological_process
  10. (GO:0000019, GO:0000018, 8, 7)
     - regulation of mitotic recombination -> regulation of DNA recombination

Level difference statistics:
  Min difference: 1
  Max difference: 11
  Average difference: 2.08

Edge counts by level difference:
  Difference 1: 4436 edges
  Difference 2: 145 edges
  Difference 3: 307 edges
  Difference 4: 277 edges
  Difference 5: 348 edges
  Difference 6: 280 edges
  Difference 7: 234 edges
  Difference 8: 87 edges
  Difference 9: 16 edges
  Difference 10: 8 edges
  Difference 11: 5 edges

Edge direction analysis:
  Higher level to lower level: 5769 edges
  Lower level to higher level: 374 edges

Plot saved to: /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/go_cross_level_edges_2025-05-18-17-02-27.png

Can process levels in order (higher to lower): False

Found 374 problematic edges (lower level to higher level)
Sample of problematic edges:
  1. (GO:0006631, GO:0032787, 5, 7)
     - fatty acid metabolic process -> monocarboxylic acid metabolic process
  2. (GO:0000054, GO:0051168, 3, 6)
     - ribosomal subunit export from nucleus -> nuclear export
  3. (GO:0000070, GO:0000819, 4, 5)
     - mitotic sister chromatid segregation -> sister chromatid segregation
  4. (GO:0000096, GO:0019752, 4, 6)
     - sulfur amino acid metabolic process -> carboxylic acid metabolic process
  5. (GO:0000097, GO:0046394, 5, 6)
     - sulfur amino acid biosynthetic process -> carboxylic acid biosynthetic process

Found 531 same-level connections
Sample of same-level edges:
  1. (GO:0006281, GO:0006259) at level 5
     - DNA repair -> DNA metabolic process
  2. (GO:0000025, GO:0000023) at level 7
     - maltose catabolic process -> maltose metabolic process
  3. (GO:0000045, GO:1905037) at level 6
     - autophagosome assembly -> autophagosome organization
  4. (GO:0000054, GO:0051656) at level 3
     - ribosomal subunit export from nucleus -> establishment of organelle localization
  5. (GO:0000054, GO:0033750) at level 3
     - ribosomal subunit export from nucleus -> ribosome localization

Same-level edge counts by level:
  Level 2: 3 edges
  Level 3: 43 edges
  Level 4: 142 edges
  Level 5: 132 edges
  Level 6: 111 edges
  Level 7: 57 edges
  Level 8: 42 edges
  Level 9: 1 edges

NOTE: Same-level connections mean nodes at the same level might depend on each other.
For optimization, we would need to check if these create cycles within a level.

No cycles found within any level.
This confirms we can safely process all nodes within a level in a single batch.

=== OPTIMIZATION STRATEGY SUMMARY ===
1. GO nodes at different levels are connected, with some connections going from lower to higher levels.
2. This creates cycles in the level dependency graph, preventing simple level-by-level processing.
3. For optimization, we would need to use a more complex approach like strongly connected components.
3. Nodes within the same level may have connections but no cycles.
4. For optimization, we can use topological sorting within each level.
```
