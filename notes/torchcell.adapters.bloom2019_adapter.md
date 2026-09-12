---
id: aq3ngygi47urq8nmkh77a8b
title: Bloom2019_adapter
desc: ''
updated: 1789203220476
created: 1789203220476
---

## 2026.09.12 - Adapter for the segregant growth panel

`torchcell/adapters/bloom2019_adapter.py` + `torchcell/adapters/conf/bloom2019_adapter.yaml`, registered in `torchcell/knowledge_graphs/dataset_adapter_map.py`. Serves [[torchcell.datasets.scerevisiae.bloom2019]].

Three additions to `CellAdapter`, all new methods (no served method was edited, so `kg_manifest admit` sees no adapter drift):

- `segregant genotype (chunked)`: a `segregant genotype` node (graph schema `is_a: genotype`) projecting `cross`, `segregant_id`, `parent_1`, `parent_2`, `n_blocks`; the blocks travel in `serialized_data`. The node id is the sha256 of the whole `model_dump`, hashed once per record. The existing `genotype to experiment (chunked)` edge hashes the same way and reads no gene-keyed attribute, so it is reused; `genotype member of` gained `segregant genotype` as a source label.
- `environment response phenotype (chunked)` and its reference method: the `environment response phenotype` node class did not exist in the graph schema before, though the schema class and 13 chemogenomic loaders did. Properties: `environment_response`, `environment_response_se`, `measurement_type`, `assay_type`, plus the label fields and `serialized_data`.
- No `perturbation (chunked)` / `perturbation to genotype (chunked)` in the conf: a segregant carries no gene-keyed perturbation.

Tests: `tests/torchcell/adapters/test_bloom2019_adapter.py`; the conf-vs-schema consistency test in `tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py` covers the new class.
