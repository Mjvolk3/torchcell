---
id: kxrphvj8y7wcmlb1t4mrl0y
title: smf_tmi_combine_kg
desc: ''
updated: 1723180260853
created: 1723178576805
---
## 2024.08.08 - Checking that Combine Produces the Same Import Summary

This should eventually be moved to tests somehow although it is quite a complicated thing to put in tests.

Output from combining the two individual `biocypher-out` dirs:

```bash
IMPORT DONE in 5s 178ms. 
Imported:
  323416 nodes
  841603 relationships
  1826530 properties
Peak memory usage: 1.035GiB
```

Output from `torchcell/knowledge_graphs/dmf_tmi_combine_kg.py`

```bash
IMPORT DONE in 7s 664ms. 
Imported:
  323417 nodes
  953196 relationships
  1826533 properties
Peak memory usage: 1.035GiB
Build and run process completed.
```

Not good! They are different...
