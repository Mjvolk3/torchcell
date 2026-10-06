---
id: lzheuobi3dds0iqheucsp3a
title: Test_hillenmeyer2008_adapter
desc: ''
updated: 1791269003480
created: 1791269003480
---

## 2026.10.06 - Phase 21 tests

- Constructor tests appended: exact conf content, wiring, logged method table, refusal on a missing conf; checks in [[tests.torchcell.adapters._adapter_init_harness]]. The expected conf for each class is listed in the test docstring.
- A conf parsing to a YAML list is refused with `het_hillenmeyer2008_adapter.yaml must parse to a mapping, got <class 'omegaconf.listconfig.ListConfig'>`.
