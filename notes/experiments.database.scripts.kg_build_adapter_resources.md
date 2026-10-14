---
id: c696cifioq0mrspvetskdit
title: Kg_build_adapter_resources
desc: ''
updated: 1791994344837
created: 1791994344837
---

## 2026.10.14 - Per-adapter resources of the KG 4.0 generation (job 4901)

The generator's cgroup sampler (`torchcell/build_telemetry.py`, 5 s interval) tags every sample with the adapter and phase current at that instant, so a generation run ends with the material for a per-adapter resource table without parsing the tqdm bars. This script folds `biocypher-out/<ts>/telemetry/resource_samples.csv` into one row per adapter and phase and counts the pool recycles the generate log reports inside each phase's window. Output: `experiments/database/results/kg_build_adapter_resources_<label>.{json,md}`.

```bash
PYTHONPATH=. ~/miniconda3/envs/torchcell/bin/python experiments/database/scripts/kg_build_adapter_resources.py \
    --telemetry /db/database/biocypher-out/2026-10-14_03-53-33/telemetry/resource_samples.csv \
    --generate-log /scratch/projects/torchcell/database/slurm/output/4901_generate.log --label 4901
```

Job 4901 (170 datasets, three organisms, v1.7.4, 64 CPUs, a 256 G container limit) was killed by its 12 h slurm limit at adapter 65 of 170, so the table below covers the first 12 h 14 min of generation (6,506 samples, 108 phases, 205 recycles). The 14 longest phases, from `experiments/database/results/kg_build_adapter_resources_4901.md`:

| adapter | phase | hours | cores mean | cores max | parent cores | mem mean (GiB) | mem max (GiB) | procs | recycles |
|---|---|---|---|---|---|---|---|---|---|
| EnvChemgenHoepfner2014Adapter | node | 4.24 | 19.4 | 64.6 | 0.75 | 213 | 256 | 190 | 139 |
| EnvChemgenHoepfner2014Adapter | edge | 2.25 | 21.8 | 64.5 | 0.61 | 213 | 256 | 190 | 49 |
| PooledPromoterExpressionVaishnav2022Adapter | node | 1.02 | 1.1 | 25.9 | 0.98 | 110 | 165 | 108 | 0 |
| PooledPromoterExpressionVaishnav2022Adapter | edge | 0.70 | 1.2 | 30.3 | 0.99 | 48 | 59 | 108 | 0 |
| Hale2024Adapter | edge | 0.34 | 53.6 | 64.4 | 0.24 | 180 | 207 | 190 | 3 |
| Hale2024Adapter | node | 0.28 | 51.4 | 64.4 | 0.35 | 163 | 205 | 190 | 3 |
| DmiCostanzo2016Adapter | edge | 0.23 | 60.8 | 64.2 | 0.23 | 158 | 167 | 190 | 0 |
| EnvChemgenPiotrowski2017Adapter | node | 0.21 | 28.6 | 64.5 | 0.58 | 178 | 210 | 190 | 6 |
| DmfCostanzo2016Adapter | edge | 0.20 | 61.4 | 64.2 | 0.29 | 132 | 146 | 190 | 0 |
| DmiCostanzo2016Adapter | node | 0.19 | 60.0 | 64.3 | 0.32 | 161 | 167 | 190 | 0 |
| DmfCostanzo2016Adapter | node | 0.15 | 60.8 | 64.1 | 0.32 | 75 | 122 | 190 | 0 |
| CaudalPanTranscriptome2024Adapter | edge | 0.11 | 3.7 | 18.6 | 0.08 | 70 | 73 | 74 | 0 |
| ProteomeMuenzner2024Adapter | edge | 0.10 | 3.2 | 20.1 | 0.10 | 77 | 79 | 74 | 0 |
| CaudalPanTranscriptome2024Adapter | node | 0.10 | 3.4 | 6.3 | 0.13 | 66 | 72 | 74 | 0 |

What the table says about the two slow shapes:

- **Hoepfner 2014 is memory-throttled.** Its records resolve to 39 KB (Costanzo's to 22 KB), the 128 MiB chunk budget cuts its chunks to 3,446 records, and the container sat at a mean 213 G against the 256 G limit, above the 0.8 recycle threshold (205 G), so the pool was torn down and rebuilt every two minutes (139 recycles in the node pass) and the pass ran at a third of Costanzo's cores. The parent process used under one core, so it is not the bottleneck. Levers, untested: more container memory for the build, a per-adapter chunk budget and worker count for 39 KB-record adapters, and a profile of what in the record resolves to 39 KB.
- **Vaishnav 2022 runs on one core** (mean 1.1, no recycles): a per-record cost inside the adapter, not the pool.

The run's telemetry stays under `/db/database/biocypher-out/2026-10-14_03-53-33/telemetry/` and is archived with the CSVs to `/bulk/biocypher-out/` by the build's record stage.
