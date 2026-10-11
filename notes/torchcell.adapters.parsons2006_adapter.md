---
id: s4za042w10lsnr7f2mplxva
title: Parsons2006_adapter
desc: ''
updated: 1791746764194
created: 1791746764194
---

## 2026.10.11 - The Parsons 2006 adapter

`EnvChemgenParsons2006Adapter` serves [[torchcell.datasets.scerevisiae.parsons2006]] with `conf/env_chemgen_parsons2006_adapter.yaml`, the Wildenhain 2015 method list (17 node and 15 edge methods): gene-keyed `genotype` and `perturbation` nodes for the barcoded kanMX4 deletions, one `environment perturbation` node per compound so its CAS-resolved InChIKey joins the other chemogenomic datasets, three `experiment reference` nodes (one per medium: YPD, SC, SC lacking lysine) on one BY4741 `genome` node. It is in `dataset_adapter_map`, so it is in the KG 4.0 rebuild membership.

`tests/torchcell/adapters/test_parsons2006_adapter.py` runs the loader's 11-record synthetic build through the real adapter: 11 experiment nodes, 3 references on 1 genome node, the three compound nodes' names, InChIKeys and doses, and 12 phenotype node emissions (11 records plus the one reference phenotype the three references share, since phenotype ids are content hashes).
