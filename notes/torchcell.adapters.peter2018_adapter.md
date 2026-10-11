---
id: zt6x7n4jyhwrhnm64ox41za
title: Peter2018_adapter
desc: ''
updated: 1791716769610
created: 1791716769610
---

## 2026.10.11 - Peter 2018 isolate-growth adapter

`IsolateGrowthPeter2018Adapter` serves `IsolateGrowthPeter2018Dataset` ([[torchcell.datasets.scerevisiae.peter2018]]) from `conf/isolate_growth_peter2018_adapter.yaml`: 16 node and 14 edge methods. Compact genotype, as the Caudal 2024 conf: no `perturbation (chunked)` / `perturbation to genotype (chunked)` pair, because an isolate carries ~5,000 perturbations and the variant sequences stay off-graph in the pinned Peter tarball. Environment perturbations (Supplementary Table 2's chemical stresses) are nodes. Mapped in `dataset_adapter_map` beside Bloom 2019. Test: `tests/torchcell/adapters/test_peter2018_adapter.py` (the shared init harness rebuilds the expected conf from `Shape("environment response phenotype", perturbation=False, env_perturbation=True)`).
