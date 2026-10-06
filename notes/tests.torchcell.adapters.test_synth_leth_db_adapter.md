---
id: f9tl6236xzwd3fact5i6tk6
title: Test_synth_leth_db_adapter
desc: ''
updated: 1791269205789
created: 1791269205789
---

## 2026.10.06 - Phase 21 tests

- Constructor tests appended: exact conf content, wiring, logged method table, refusal on a missing conf; checks in [[tests.torchcell.adapters._adapter_init_harness]]. The expected conf for each class is listed in the test docstring.
- Finding (synth_leth_db_adapter.py:73): `SynthRescueYeastSynthLethDbAdapter`'s `dataset` is annotated `SynthLethalityYeastSynthLethDbDataset` (phenotype `SyntheticLethalityPhenotype`) while its conf serves `synthetic rescue phenotype`.
- Both classes print `<Class> initialized with config: <config>` once.
- Reach (audit 1): type-level only; `dataset_adapter_map.py:149` pairs the rescue adapter with `SynthRescueYeastSynthLethDbDataset` at runtime.
