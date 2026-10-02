---
id: sc3s45nt0z64z8xvcqnc0aa
title: Test_cell_adapter
desc: ''
updated: 1790562635396
created: 1790562635396
---

## 2026.09.27 - CellAdapter on an in-memory dataset with wandb recorded

Twenty-one test functions (55 cases, two parametrized over eighteen cases of thirteen phenotype kinds) on a tiny in-memory dataset built from real schema records. Expected BioCypher nodes and edges are built by hand and compared whole (id, label, preferred id, every property); `wandb.init` and `wandb.log` are replaced by recorders on the module's `wandb` name and every logged payload is asserted. The chunked handlers fork one loader worker each; the pool path runs with three one-record chunks (two workers, so the per-group pool rebuild runs) and with one chunk; an autouse fixture calls `gc.unfreeze()` because the adapter calls `gc.freeze()`. Module coverage from this file 93%, 99% with the existing adapter tests (the CRISPR, media, temperature and environment-perturbation builders are theirs). Findings: the chunk-size `ValueError` joins its two message pieces with no space (cell_adapter.py lines 72 to 73); the chunking decorator looks up the memory reduction factor without `is_edge`, so an edge method's factor scales its chunk size but never its loader batch size (line 404); `get_nodes` runs methods in registration order filtered by the config, not in config order as its docstring says (lines 446 to 447); only the environment-response reference collector deduplicates, so the fitness collector and the others emit identical duplicate reference nodes (line 1103 against 1118). Phase 8 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.10.01 - Phase 19: chunk sizing, pool recycling and the single pass

80 cases (55 before). The pool-side tests replace `ProcessPoolExecutor` at its import site with a synchronous stand-in (each chunk runs on submit; nothing forks) and a chunk function that returns (method, first record, length), so the yielded list is the chunk boundary list. Records `{"i": i, "x": "a" * k}` dump to 16 + digits(i) + k characters.

- `_estimate_record_bytes`: median of evenly spaced samples (k = [10, 0, 30, 0, 20], samples 2: [27, 47, 37] -> 37; default samples: 27), one LMDB close, cached.
- Single pass: factor min / count (0.5 / 3); chunk shrinks to the byte budget (601 records of median 30 bytes, budget 300 records: [0, 300), [300, 600), [600, 601) and the log line), floors at 256, stays when the budget is larger.
- In-process rule by records and, when set, bytes (81 <= 81 in process, 81 > 80 to the pool); an empty dataset yields nothing and builds no pool.
- Memory recycling: scripted cgroup readings give pools of [3, 6, 1] chunks, seven readings, three `gc.freeze` calls, the exact log line; the submission window is workers + 2 then one per consumed chunk, in order and (as multisets) under `completion_order`.
- `data_chunker` in process, `_all_chunked` (per record, each folded handler's output in table order), `_pack_chunk` with and without `row_specs`, `_yield_methods` single pass for nodes and edges (collector in the loop, one folded traversal, events and `BuildPhase` transitions), and `cgroup_memory_fraction` on v2 files (0.25), `max` (refused before `memory.current` is read) and a v1 tree (`FileNotFoundError` naming `memory.max`, then `memory.current`).

Finding: a single-pass traversal over an empty dataset raises `IndexError` in `_estimate_record_bytes` (cell_adapter.py:411, 617-626) where the per-method path yields nothing. Not measured whether a served dataset can be empty at build time.

Coverage of `torchcell/adapters/cell_adapter.py` from this file: 84% -> 94%.

## 2026.10.02 - Audit round 2 corrections

82 cases. The count in the 2026.10.01 section is corrected to 80.

- The budget test is parametrized over 800 and exactly 500 records (the chunk). The shrink rule is strict, so neither logs anything, and the `<=` mutant (it would log "500 -> 500") dies.
- The single-pass test asserts the exact first item: the `dataset` node with id `ToyDataset` for nodes, and the `genome member of` edge from the genome to the reference for edges.
- The `sync_pool` fixture docstring is corrected: it records pools and does not count `gc.freeze`. `dataset: Any` replaces the `arg-type` ignore.
- New, reach of the Phase 8 finding at cell_adapter.py:584. `data_chunker` looks the factor up without `is_edge`. With the shipped `torchcell/adapters/conf/dmf_costanzo2016_adapter.yaml` (loaded by `costanzo2016_adapter.py`; its chunked edge methods carry 0.5), the edge lookup gives 0.5 but the decorator's gives 1.0, and the "experiment to dataset" loader is built with batch 2 instead of 1. Not measured: the memory this costs on a real build.
- Reach of the empty-dataset single-pass `IndexError`: the live `torchcell/knowledge_graphs/conf/kg_uncapped.yaml` sets `single_pass: true`. It is a crash only, and only if a dataset is empty, which is not measured.
