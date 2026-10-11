---
id: p1jtgrnzeq8p1cw7dyiluun
title: Piotrowski2017_adapter
desc: ''
updated: 1791737271559
created: 1791737271559
---

## 2026.10.11 - Piotrowski 2017 chemical-genetic adapter

`EnvChemgenPiotrowski2017Adapter` serves `EnvChemgenPiotrowski2017Dataset` ([[torchcell.datasets.scerevisiae.piotrowski2017]]) from `conf/env_chemgen_piotrowski2017_adapter.yaml`: 17 node and 15 edge methods, the method list of the Vanacloig 2022 conf. Gene-keyed genotype WITH the `perturbation (chunked)` / `perturbation to genotype (chunked)` pair, because each record holds one barcoded diagnostic-pool deletion. The dosed compound is an environment-perturbation node carrying its InChIKey, or its typed InChIKey gap for the 33 kept conditions that release no InChI. The pdr1 pdr3 snq2 background rides on the genome node, shared with Vanacloig 2022. Mapped in `dataset_adapter_map` beside Vanacloig 2022. Test: `tests/torchcell/adapters/test_piotrowski2017_adapter.py` (the shared init harness rebuilds the expected conf from `Shape("environment response phenotype", perturbation=True, env_perturbation=True)`).
