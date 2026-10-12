---
id: oikmii157g7gglfzxuze5jj
title: Yoshikawa2009_adapter
desc: ''
updated: 1791802866677
created: 1791802866677
---

## 2026.10.12 - The Yoshikawa 2009 adapter

`EnvChemgenYoshikawa2009Adapter` serves [[torchcell.datasets.scerevisiae.yoshikawa2009]] with `conf/env_chemgen_yoshikawa2009_adapter.yaml`, the Parsons 2006 method list (17 node and 15 edge methods): gene-keyed `genotype` and `perturbation` nodes for the barcoded kanMX4 deletions, one `environment perturbation` node per stress (ethanol at 8% v/v, sodium chloride at 1 M) so the two join the other datasets that add the same compounds, and three `experiment reference` nodes (one per condition, each the standard strain's Table 1 rate) on one BY4742 MATalpha `genome` node. It is in the yeast section of `dataset_adapter_map`, so it is in the KG 4.0 rebuild membership.

`tests/torchcell/adapters/test_yoshikawa2009_adapter.py` runs the loader's 24-record synthetic build through the real adapter: 24 experiment nodes, 3 references on 1 genome node with rates 0.452, 0.178 and 0.203, the two compound nodes' InChIKeys and doses, and 27 distinct phenotype ids (24 records plus 3 reference phenotypes, which differ by condition here, unlike Parsons' shared log2 ratio 0).
