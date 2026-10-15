---
id: u3kjsdqq59mj3vivnv7s893
title: Generate_adapter
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - One array task: one dataset, one adapter, one directory, DONE last

`torchcell/knowledge_graphs/generate_adapter.py` is the container-side entry of the per-adapter generation ([[torchcell.knowledge_graphs.generation_array]]). The array script runs it as

```bash
python -m torchcell.knowledge_graphs.generate_adapter --config-name kg_uncapped \
    --adapter-index 64 --out-dir /var/lib/neo4j/biocypher-out/<ts>/adapters/064_EnvChemgenHoepfner2014Adapter \
    --include-private --organisms scerevisiae,ecoli,pputida [hydra overrides]
```

- `--adapter-index` and `--out-dir` are taken off `sys.argv` before hydra parses it, as `create_scerevisiae_kg_small` does for `--include-private` and `--organisms` (`take_value_flag`). The index is a position in `generation_array.adapter_order` for the same membership flags.
- `instantiate` builds that one dataset at its loader's default root under `DATA_ROOT`, full and never subset, with the genomes injected by parameter name exactly as the serial build does (`genome` and `scerevisiae_graph` for yeast, `ecoli_genome` / `pputida_genome` through `BacterialGenomeInjector`). A missing LMDB is an error: the task was asked for this dataset.
- The adapter gets the config's `adapters` knobs (`chunk_size`, `loader_batch_size`, `single_pass`, `single_pass_chunk_budget_mb`, `chunks_per_worker`, `pool_memory_fraction`, `completion_order`, the in-process thresholds); the worker count is `SLURM_CPUS_PER_TASK`, which the array script sets to the tier's cpus. A hydra override on the command line (`adapters.single_pass_chunk_budget_mb=256`) is how a measurement varies one knob without touching the adapter.
- Nodes then edges go through `FastCsvSink` into `--out-dir`; the resource sampler writes `telemetry/resource_samples.csv` and `phase_timings.json` beside them, and W&B gets one run per task named `<array job>_<task>_<Adapter>`.
- After `sink.finish()`, `bc.write_import_call()` and `bc.write_schema_info()`, the `DONE` marker (`DoneMarker`) is written. Nothing the merge reads is written after it. A directory that already carries a marker is refused, never regenerated in place.

Tests: `tests/torchcell/knowledge_graphs/test_generate_adapter.py` (the flags leave argv, the worker count comes from the tier, the slot resolves by index, the one dataset is built with its genomes injected by name, a missing LMDB is an error).
