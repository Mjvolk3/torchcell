---
id: xah8qndm6rzskcx3gyqilus
title: Test_generate_adapter
desc: ''
updated: 1792055330786
created: 1792055330786
---

## 2026.10.15 - What the tests pin

`tests/torchcell/knowledge_graphs/test_generate_adapter.py` covers the helpers of [[torchcell.knowledge_graphs.generate_adapter]] with the shared fakes of `_kg_build_fakes`: `--adapter-index` and `--out-dir` leave `sys.argv` before hydra parses it (absent, twice, without a value), the worker count is the `SLURM_CPUS_PER_TASK` the array script sets from the tier, `resolve_slot` reads the order at the index, `instantiate` builds the one dataset with `genome` and `scerevisiae_graph` injected for a yeast loader and `ecoli_genome` through the bacterial injector, and a missing LMDB is an error rather than a skip. `main` itself (hydra, wandb, BioCypher, the sink, the sampler) is the serial build's own path; its contract, the marker written last, is pinned in `test_generation_array` and the slurm text test.
