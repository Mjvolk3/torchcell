---
id: ihnynf0vhqnrbpy4is8arc7
title: Vaishnav2022_adapter
desc: ''
updated: 1791808968142
created: 1791808968142
---

## 2026.10.12 - The Vaishnav 2022 adapter (#991)

`PooledPromoterExpressionVaishnav2022Adapter` +
`conf/pooled_promoter_expression_vaishnav2022_adapter.yaml`: the variable-promoter
reporter construct as a first-class `perturbation` node (it carries the released 110 bp
sequence), an SC-Ura `environment` with its media and temperature nodes, and the `pooled
sort seq expression phenotype` with its reference.

No environment-perturbation methods are enabled, which is the one thing to notice about
this conf: the assay varies the genome's content and never the medium, so every record's
`Environment.perturbations` is empty and the lane would write nothing.

The leaf is a yeast one, so it projects onto the served `perturbation` graph class and
the adapter adds no graph class of its own. Four pin places were re-derived from the
merged tree when the dataset joined: `dataset_adapter_map` (154 pairs),
`tests/torchcell/knowledge_graphs/test_build_time_projection.py` (the map total's addend
list and the uncalibrated-dataset set), `tests/torchcell/verification/test_runners.py`
(the registry-root addends and `RUN_ALL_ORDER`) and
`tests/torchcell/adapters/_adapter_init_harness.py` (the phenotype-to-method map).
