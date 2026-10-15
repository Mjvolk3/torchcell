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
| baseline | config defaults (128 MiB chunk budget, 64 workers, recycle at 0.8) | 4919 (task 4919_64), `hoepfner-study-2026-10-15_04-20-20` | submitted 2026-10-15 04:20 CDT |
| variant A | `adapters.single_pass_chunk_budget_mb=256` | 4920, `hoepfner-study-2026-10-15_04-21-22-chunk256`, after 4919 | chained |
| variant B | 32 workers (`GEN_CPUS=32`, same 320 G) | 4921, `hoepfner-study-2026-10-15_04-21-22-w32`, after 4920 | chained |

All three run at commit `9a7d1c2e` from the detached worktree `~/Documents/projects/torchcell.worktrees/kg-generation-array-study` (removed after the study). The chain was set with `scontrol update JobId=<v> Dependency=afterany:<prev>` after `SBATCH_DEPENDENCY` in the environment proved not to be honored; the conductor now takes `DEPENDENCY=`. Logs: `/scratch/projects/torchcell/database/slurm/output/<job>_64_hoepfner_*.out` and `<job>_64_generate.log`.

Readout per run: `kg_build_adapter_resources.py --telemetry <dir>/telemetry/resource_samples.csv --generate-log <task log> --label hoepfner-<job>` (hours, cores mean, memory mean and max, recycles), the `DONE` marker's `generation_wall_s` and `peak_memory_gb`, and the node and edge row counts, which must equal the baseline's.

### Baseline, partial readout at 21 min into the node pass (job 4919, task 64; measured, the run is in flight)

- Container start to the first generator line: 278 s (pip installs; see the startup section of [[torchcell.knowledge_graphs.generation_array]]). Dataset instantiation under one second (the LMDB is opened lazily).
- `single-pass chunk 11111 -> 3446 records (38940 resolved bytes per record)` at 09:30:48 UTC: the same chunk size as job 4901.
- 5,065 completed chunk bars by 1,200 s into the pass, about 250 chunks per minute. Job 4901's node pass ran at 33 chunks per minute (about 8,800 chunks in 4 h 14 min).
- Telemetry at 1,247 s: 49.1 cores in use (4901 mean 19.4), 206 G resident of which 73.6 G anonymous (4901: 213 G of which 85 G), parent process 1.04 cores (4901: 0.75), 142 processes, 0 pool recycles (4901: 139 in the node pass).

Reading, labeled: the resident memory sits at the same 206 G (the LMDB page cache plus the heap) but under a 320 G limit the 0.8 threshold is 256 G, so no recycle has fired, and the pass runs at 2.5x the cores and about 7x the chunk rate. That is consistent with hypothesis 1 (the page cache crossed the threshold) and does not separate hypothesis 2 (the empty parent); the parent now sits at one core, so the next bound is the parent's dedup-and-append loop (hypothesis, untested). Final wall time, peak memory and the edge pass are reported when the task writes DONE.

### Results (jobs 4919, 4920 complete; 4921 in flight)

Each run wrote the same 60,650,204 node rows and 181,879,823 edge rows (the sink dropped 73,662,645 duplicate node rows and 73,648,500 duplicate edge rows within the adapter, 0 duplicated experiment ids), so the knobs changed nothing in the output. Per-phase tables: `experiments/database/results/kg_build_adapter_resources_hoepfner-4919.md` and `-4920.md` ([[experiments.database.scripts.kg_build_adapter_resources]]).

| run | job | generation wall | node pass | edge pass | peak memory | recycles |
|---|---|---|---|---|---|---|
| job 4901 (serial container, 256 G, 64 CPUs) | 4901 | 6 h 29 min | 4.24 h, 19.4 cores mean, 213 G mean | 2.25 h, 21.8 cores, 213 G mean | 256 G (the limit) | 139 + 49 |
| baseline (320 G, 64 workers, 128 MiB chunks) | 4919_64 | **4,558 s (1 h 16 min)** | 0.60 h, 40.5 cores mean, parent 1.00 core, 172 G mean | 0.66 h, 61.7 cores, 230 G mean | 257.2 GiB | 4 + 0 |
| variant A (chunk budget 256 MiB) | 4920_64 | 4,617 s (1 h 17 min) | 0.39 h, 48.9 cores, 148 G mean | 0.88 h, 35.9 cores, 259 G mean, 298 G max | 297.9 GiB | 2 + 57 |
| variant B (32 workers, 320 G) | 4921_64 | running since 07:06 CDT (1 h 06 min at 08:12), 0 recycles so far | | | | |

Job elapsed for 4919_64 was 1 h 20 min 59 s including the 278 s of container startup.

What the numbers say, labeled:

- **Measured.** The baseline is 5.1x faster than job 4901's Hoepfner pass (1.26 h against 6.49 h) with the same chunk size (3,446 records) and the same adapter code, under a 320 G limit instead of 256 G and in a container of its own. Memory sat at a mean 172 G in the node pass (4901: 213 G) and recycled 4 times instead of 139; the edge pass ran at 61.7 cores with no recycle. Hypothesis 1 (the page cache crossing the 0.8 threshold) and hypothesis 2 (the cumulative parent) are both consistent with it and this design cannot separate them, since the array changed the limit and the parent together; the 4901 anon memory (85 G) against 4919's parent-free start is the only evidence that leans to hypothesis 2 being part of it.
- **Measured.** Variant A buys 13 min in the node pass (0.39 h at 48.9 cores) and loses them in the edge pass (0.88 h, 57 recycles at a mean 259 G against the 256 G threshold): the 256 MiB chunks double the in-flight set and the edge pass, whose rows are the wider ones, crosses the recycle threshold that the 128 MiB baseline stayed under. Net 1.3 % slower and 41 G more peak memory. Not adopted; hypothesis 3 (per-chunk overhead) is real in the node pass and is paid back in memory.
- **Hypothesis (untested).** The baseline's node pass runs the parent at 1.00 core while the workers sit at 40.5 of 64: the parent's dedup-and-append loop (`FastCsvSink._add_nodes`, a set lookup per row plus the write) is now the bound of the node pass, and the next lever is in the sink, not in the pool knobs.
- Variant B is reported when 4921 writes `DONE`; at 32 workers the question is whether the node pass holds its rate with half the forks per recycle and the edge pass stays unthrottled.

For the tier table the baseline sets the Hoepfner tier's expectation at about 1.5 h with 320 G, well inside the 36 h reservation; the reservation stays as it is (the owner's rule: overestimate).
