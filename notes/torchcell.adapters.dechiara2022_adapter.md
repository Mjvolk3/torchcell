---
id: t38nvt4q4z3mt5yepr2m6m1
title: Dechiara2022_adapter
desc: ''
updated: 1791728053702
created: 1791728053702
---

## 2026.10.11 - Three adapters, one module

`LifeCycleDeChiara2022Adapter`, `DoublingTimeDeChiara2022Adapter` and `CellYieldDeChiara2022Adapter` serve the three De Chiara 2022 stores ([[torchcell.datasets.scerevisiae.dechiara2022]]). Each class names its own conf in its body (`conf_name`), which is how `kg_manifest.adapter_conf_name` binds a class to its enable-list in a multi-class module.

The confs follow Caudal 2024's compact genotype: no per-perturbation node and no perturbation-to-genotype edge, because each 1,011-panel isolate carries about 5,000 perturbations off S288C and the full set is already on the genotype node (its sequences off-graph in Peter 2018's pinned tarball). The two growth confs serve the carbon or nitrogen source as `environment perturbation` nodes; the life-cycle conf serves `life cycle phenotype` and no environment perturbation, since its five protocols carry none.
