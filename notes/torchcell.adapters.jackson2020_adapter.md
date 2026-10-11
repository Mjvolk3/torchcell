---
id: 2r2osdo3tj1g6g9gfkauam8
title: Jackson2020_adapter
desc: ''
updated: 1791732541451
created: 1791732541451
---

## 2026.10.11 - Jackson 2020 pseudobulk adapter

`TfDeletionPseudobulkJackson2020Adapter` serves `TfDeletionPseudobulkJackson2020Dataset` ([[torchcell.datasets.scerevisiae.jackson2020]]) from `conf/tf_deletion_pseudobulk_jackson2020_adapter.yaml`: the full genotype (one barcoded homozygous deletion per strain, a `perturbation` node), environment-perturbation nodes (RAPA's rapamycin) and the new `pseudobulk umi count phenotype` node and reference collector (`CellAdapter._pseudobulk_umi_count_phenotype_node`, `_get_pseudobulk_umi_count_phenotype_reference_nodes`; dicts and per-cell lists as JSON strings, totals typed). The temperature methods are enabled as in every conf and emit nothing: every environment gaps its temperature. The FY4/FY5 background rides on the `genome` node's `serialized_data`. Mapped in `dataset_adapter_map` after Peter 2018. Tests: `tests/torchcell/adapters/test_jackson2020_adapter.py` (init harness, `Shape("pseudobulk umi count phenotype", perturbation=True, env_perturbation=True)`) and `tests/torchcell/adapters/test_pseudobulk_umi_count_node.py` (the node's exact properties and the reference collector's dedup).
