---
id: hnsslicuqnmhh7ldob1nauv
title: Dango_lambda_determination
desc: ''
updated: 1746736829975
created: 1746736779136
---
```python
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell/e
xperiments/005-kuzmin2018-tmi/scripts/dango_lambda_determination.py
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
  Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
  Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
  warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
2025-05-08 15:40:17,141 - __main__ - INFO - Loading gene graphs...
/Users/michaelvolk/Documents/projects/torchcell/data/go/go.obo: fmt(1.2) rel(2024-11-03) 43,983 Terms
2025-05-08 15:40:19,602 - __main__ - INFO - Calculating zero decrease percentages...
2025-05-08 15:40:19,602 - __main__ - INFO - 
Analyzing neighborhood network...
2025-05-08 15:40:19,884 - __main__ - INFO - Network type: neighborhood
2025-05-08 15:40:19,884 - __main__ - INFO - Total nodes: 2353
2025-05-08 15:40:19,884 - __main__ - INFO - Possible edges: 2767128
2025-05-08 15:40:19,884 - __main__ - INFO - Edges in v9.1: 45601
2025-05-08 15:40:19,884 - __main__ - INFO - Edges in v11.0: 120965
2025-05-08 15:40:19,884 - __main__ - INFO - New edges in v11.0: 100651
2025-05-08 15:40:19,884 - __main__ - INFO - Zero edges in v9.1: 2721527
2025-05-08 15:40:19,884 - __main__ - INFO - Percentage of decreased zeros: 3.6983%
2025-05-08 15:40:19,892 - __main__ - INFO - 
Analyzing fusion network...
2025-05-08 15:40:19,904 - __main__ - INFO - Network type: fusion
2025-05-08 15:40:19,904 - __main__ - INFO - Total nodes: 2589
2025-05-08 15:40:19,904 - __main__ - INFO - Possible edges: 3350166
2025-05-08 15:40:19,904 - __main__ - INFO - Edges in v9.1: 1358
2025-05-08 15:40:19,904 - __main__ - INFO - Edges in v11.0: 3914
2025-05-08 15:40:19,904 - __main__ - INFO - New edges in v11.0: 3522
2025-05-08 15:40:19,904 - __main__ - INFO - Zero edges in v9.1: 3348808
2025-05-08 15:40:19,904 - __main__ - INFO - Percentage of decreased zeros: 0.1052%
2025-05-08 15:40:19,904 - __main__ - INFO - 
Analyzing cooccurence network...
2025-05-08 15:40:19,916 - __main__ - INFO - Network type: cooccurence
2025-05-08 15:40:19,916 - __main__ - INFO - Total nodes: 1454
2025-05-08 15:40:19,916 - __main__ - INFO - Possible edges: 1056331
2025-05-08 15:40:19,916 - __main__ - INFO - Edges in v9.1: 2659
2025-05-08 15:40:19,916 - __main__ - INFO - Edges in v11.0: 4668
2025-05-08 15:40:19,916 - __main__ - INFO - New edges in v11.0: 3440
2025-05-08 15:40:19,916 - __main__ - INFO - Zero edges in v9.1: 1053672
2025-05-08 15:40:19,916 - __main__ - INFO - Percentage of decreased zeros: 0.3265%
2025-05-08 15:40:19,916 - __main__ - INFO - 
Analyzing coexpression network...
2025-05-08 15:40:20,496 - __main__ - INFO - Network type: coexpression
2025-05-08 15:40:20,496 - __main__ - INFO - Total nodes: 6013
2025-05-08 15:40:20,496 - __main__ - INFO - Possible edges: 18075078
2025-05-08 15:40:20,496 - __main__ - INFO - Edges in v9.1: 313688
2025-05-08 15:40:20,496 - __main__ - INFO - Edges in v11.0: 554833
2025-05-08 15:40:20,496 - __main__ - INFO - New edges in v11.0: 441450
2025-05-08 15:40:20,496 - __main__ - INFO - Zero edges in v9.1: 17761390
2025-05-08 15:40:20,497 - __main__ - INFO - Percentage of decreased zeros: 2.4854%
2025-05-08 15:40:20,552 - __main__ - INFO - 
Analyzing experimental network...
2025-05-08 15:40:20,985 - __main__ - INFO - Network type: experimental
2025-05-08 15:40:20,985 - __main__ - INFO - Total nodes: 6159
2025-05-08 15:40:20,985 - __main__ - INFO - Possible edges: 18963561
2025-05-08 15:40:20,985 - __main__ - INFO - Edges in v9.1: 219450
2025-05-08 15:40:20,985 - __main__ - INFO - Edges in v11.0: 392772
2025-05-08 15:40:20,985 - __main__ - INFO - New edges in v11.0: 309524
2025-05-08 15:40:20,985 - __main__ - INFO - Zero edges in v9.1: 18744111
2025-05-08 15:40:20,985 - __main__ - INFO - Percentage of decreased zeros: 1.6513%
2025-05-08 15:40:21,028 - __main__ - INFO - 
Analyzing database network...
2025-05-08 15:40:21,078 - __main__ - INFO - Network type: database
2025-05-08 15:40:21,078 - __main__ - INFO - Total nodes: 3338
2025-05-08 15:40:21,078 - __main__ - INFO - Possible edges: 5569453
2025-05-08 15:40:21,078 - __main__ - INFO - Edges in v9.1: 33486
2025-05-08 15:40:21,078 - __main__ - INFO - Edges in v11.0: 46816
2025-05-08 15:40:21,078 - __main__ - INFO - New edges in v11.0: 34245
2025-05-08 15:40:21,078 - __main__ - INFO - Zero edges in v9.1: 5535967
2025-05-08 15:40:21,078 - __main__ - INFO - Percentage of decreased zeros: 0.6186%
2025-05-08 15:40:21,082 - __main__ - INFO - 
--- SUMMARY OF ZERO DECREASE PERCENTAGES ---
2025-05-08 15:40:21,082 - __main__ - INFO - neighborhood: 3.6983%
2025-05-08 15:40:21,082 - __main__ - INFO - fusion: 0.1052%
2025-05-08 15:40:21,082 - __main__ - INFO - cooccurence: 0.3265%
2025-05-08 15:40:21,082 - __main__ - INFO - coexpression: 2.4854%
2025-05-08 15:40:21,082 - __main__ - INFO - experimental: 1.6513%
2025-05-08 15:40:21,082 - __main__ - INFO - database: 0.6186%
2025-05-08 15:40:21,082 - __main__ - INFO - 
--- DETERMINED LAMBDA VALUES ---
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_neighborhood: 0.1
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_fusion: 1.0
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_cooccurence: 1.0
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_coexpression: 0.1
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_experimental: 0.1
2025-05-08 15:40:21,082 - __main__ - INFO - string9_1_database: 1.0
2025-05-08 15:40:21,668 - __main__ - INFO - Plot saved to /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/string_v9.1_vs_v11.0_comparison_2025-05-08-15-40-21.png
2025-05-08 15:40:21,668 - __main__ - INFO - Generated plot at: /Users/michaelvolk/Documents/projects/torchcell/notes/assets/images/string_v9.1_vs_v11.0_comparison_2025-05-08-15-40-21.png
michaelvolk@M1-MV torchcell %                    
```
