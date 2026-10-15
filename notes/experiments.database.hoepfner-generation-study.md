---
id: 07fca8rgc57b76xtn7e9qxt
title: Hoepfner Generation Study
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - Why Hoepfner 2014 throttles, and what the array measures

The calibration run of the full build, job 4901 (170 datasets, 64 CPUs, a 256 G container), spent 4 h 14 min of node writing and 2 h 14 min of edge writing on `EnvChemgenHoepfner2014Adapter` (30.3 M experiments) at a mean 19.4 cores against 64, a mean 213 G against the 256 G limit, with 139 pool recycles in the node pass and 49 in the edge pass ([[experiments.database.scripts.kg_build_adapter_resources]], `kg_build_adapter_resources_4901.md`). Costanzo DMF (20.7 M experiments) took 9 min of nodes at 60.8 cores. The pass is memory-throttled, not compute-bound and not parent-bound (parent 0.75 cores).

### What the 4901 telemetry says about the memory

Folding `resource_samples.csv` over the Hoepfner node phase (1,936 samples): `mem_gb` mean 213, max 256; `anon_gb` mean 85, min 70, max 123. The container's anonymous memory (heap, the part that is OOM-killed) was a third of its cgroup memory; the other 128 G is file-backed page cache, and the Hoepfner dev LMDB is 118 G (`du`, `env_chemgen_hoepfner2014/processed/lmdb`). `cgroup_memory_fraction` (`cell_adapter.py`) reads `memory.current`, which counts that page cache, so the 0.8 recycle threshold (205 G) was crossed by cache plus 85 G of heap, not by the workers' heaps alone. The 70 G of anonymous memory at the start of the phase is the serial container's cumulative state after 64 adapters (every dataset instantiated up front, the sink's global seen-id sets).

### Hypotheses (untested until the runs below report)

1. **Page cache in the limit.** With the 118 G LMDB mapped by the loader children, `memory.current` sits near the limit whatever the workers hold, so the pool recycles on cache, and at the limit the kernel reclaims LMDB pages that the next chunk re-reads from NVMe. Prediction: at 320 G (threshold 256 G) the cache plus heap fits under the threshold and recycles fall by an order of magnitude; the node pass shortens toward Costanzo's rate per record.
2. **The cumulative parent.** The serial container's 70 G parent heap makes every `gc.freeze` plus fork of 62 workers slower and privatizes more pages per worker; a per-adapter container starts with an empty parent. Prediction: the array baseline at 256 G would already recycle less than 4901; the 320 G baseline cannot separate this from hypothesis 1, so the comparison is 4901 versus the array baseline, with the limit change confounded.
3. **Chunk size.** The 128 MiB chunk budget cuts Hoepfner to 3,446 records per chunk (39 KB per resolved record), about 8,800 chunks per pass; per-chunk overhead (a DataLoader with forked children per chunk, a pickle round trip per result) is paid 8,800 times. Prediction: `adapters.single_pass_chunk_budget_mb=256` halves the chunk count and the overhead with no change to the rows written.

### The runs

Each run is one task of the generation array ([[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]]) on the current main dev store (fresh, read-only), into `/db/database/biocypher-out/hoepfner-study-<ts>/adapters/064_EnvChemgenHoepfner2014Adapter/`, at the Hoepfner tier (64 CPUs, 320 G, 36 h), chained with `--dependency=afterany` so no two run at once. Resource knobs only; the adapter and its output are unchanged.

| run | knob | job | status |
|---|---|---|---|
| baseline | config defaults (128 MiB chunk budget, 64 workers, recycle at 0.8) | see the section below | |
| variant A | `adapters.single_pass_chunk_budget_mb=256` | | |
| variant B | 32 workers (`GEN_CPUS=32`, same 320 G) | | |

Readout per run: `kg_build_adapter_resources.py --telemetry <dir>/telemetry/resource_samples.csv --generate-log <task log> --label hoepfner-<job>` (hours, cores mean, memory mean and max, recycles), the `DONE` marker's `generation_wall_s` and `peak_memory_gb`, and the node and edge row counts, which must equal the baseline's.
