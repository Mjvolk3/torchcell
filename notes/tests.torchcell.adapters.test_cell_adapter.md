---
id: sc3s45nt0z64z8xvcqnc0aa
title: Test_cell_adapter
desc: ''
updated: 1790562635396
created: 1790562635396
---

## 2026.09.27 - CellAdapter on an in-memory dataset with wandb recorded

Twenty-one test functions (55 cases, two parametrized over eighteen cases of thirteen phenotype kinds) on a tiny in-memory dataset built from real schema records. Expected BioCypher nodes and edges are built by hand and compared whole (id, label, preferred id, every property); `wandb.init` and `wandb.log` are replaced by recorders on the module's `wandb` name and every logged payload is asserted. The chunked handlers fork one loader worker each; the pool path runs with three one-record chunks (two workers, so the per-group pool rebuild runs) and with one chunk; an autouse fixture calls `gc.unfreeze()` because the adapter calls `gc.freeze()`. Module coverage from this file 93%, 99% with the existing adapter tests (the CRISPR, media, temperature and environment-perturbation builders are theirs). Findings: the chunk-size `ValueError` joins its two message pieces with no space (cell_adapter.py lines 72 to 73); the chunking decorator looks up the memory reduction factor without `is_edge`, so an edge method's factor scales its chunk size but never its loader batch size (line 404); `get_nodes` runs methods in registration order filtered by the config, not in config order as its docstring says (lines 446 to 447); only the environment-response reference collector deduplicates, so the fitness collector and the others emit identical duplicate reference nodes (line 1103 against 1118). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
