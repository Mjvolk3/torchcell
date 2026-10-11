---
id: hhgmx0x5ynfmt1ycys4qda3
title: Boocock2025_adapter
desc: ''
updated: 1791727626355
created: 1791727626355
---

## 2026.10.11 - Boocock 2025 adapter

`Boocock2025Adapter` serves [[torchcell.datasets.scerevisiae.boocock2025]]: a `segregant genotype` node per record (the haplotype mosaic in `serialized_data`), an `environment` with its `media` and `temperature`, and the new `single cell expression phenotype` node (`CellAdapter._single_cell_expression_phenotype_node` and its reference collector; the three dicts serialize to JSON strings, the totals and the feature-universe pin stay typed). The conf (`torchcell/adapters/conf/boocock2025_adapter.yaml`) enables no `perturbation (chunked)` pair, since a segregant has no gene-keyed perturbation, and no environment-perturbation methods, since no culture carries one. Registered in `dataset_adapter_map` (`Boocock2025Dataset: Boocock2025Adapter`) and in `segregant_adapters`.
