---
id: n35ar6mgsrfe8kzcpmtt020
title: _adapter_init_harness
desc: ''
updated: 1791269087787
created: 1791269087787
---

## 2026.10.06 - Phase 21 tests

Shared constructor checks for the single-dataset adapters (not a test module).

- Fixture: `SimpleNamespace(name="FakeDataset")` as the dataset; `cell_adapter.wandb` and `cell_adapter.datetime` replaced by the recorder and pinned clock of `_sga_adapter_harness` (start time `2026-09-28 12:00:00`).
- `expected_conf(Shape)` rebuilds the exact conf from the graph shape (phenotype kind; gene-keyed or segregant genotype; perturbation, CRISPR-construct and environment-perturbation nodes; `memory_reduction_factor` 1.0 or absent). Base shape: 15 node and 13 edge methods; no perturbation -1/-1; CRISPR +1/+1; environment perturbation +2/+2. Order is the `CellAdapter` registration order restricted to the enabled set.
- `assert_construction`: exact `OmegaConf.to_container(config)`, every name registered by the built adapter, no dangling edge (`EDGE_ENDPOINTS`), every chunked entity node linked (`NODE_LINK`), phenotype method equals `PHENOTYPE_METHOD[experiment_class.phenotype]`, attributes (3, 2, 500, 50), one `wandb.init`, the method table rows `[event, name, node|edge, factor or NaN]` and the start-time payload, stdout.
- `assert_missing_conf`: `osp.exists` faked False; exact `Config file not found: <adapters dir>/conf/<file>`; no `wandb.init`.
- All 31 adapter classes in 30 modules pass, so every served conf is consistent with its dataset's schema shape at 4b293d343.
- Audit 1 revision: the dataset each adapter is checked against now comes from `torchcell.knowledge_graphs.dataset_adapter_map` (inverted, exactly one pairing per adapter), with each case asserting the class it expects; the segregant flag is derived from the paired dataset's `experiment_class` genotype hint (`SegregantGenotype`); the registered-name subset check (already in `tests/torchcell/knowledge_graphs/test_adapter_schema_consistency.py`) is replaced by an order check: the enabled names in registration order equal the conf list, the order `_yield_methods` runs them in.
- Limit: the Het and Hom Hillenmeyer 2008 confs differ only in their header comment and the two Lopez 2024 confs are byte-identical, so a swapped conf in those pairs is caught only by the missing-conf test (exact file name in the refusal path).
