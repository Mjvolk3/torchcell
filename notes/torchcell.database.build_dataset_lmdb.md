---
id: sp2cvo47dklh1bxyalm4wy5
title: Build_dataset_lmdb
desc: ''
updated: 1791801780043
created: 1791801780043
---

## 2026.10.12 - `--list-stale --organisms`

`mapped_store_status(data_root, include_private, organisms)` walks `build_adapter_map(include_private, organisms)`, the map the build itself selects, and `--list-stale` takes `--organisms <comma-separated set>` (default every organism). A yeast-only rebuild therefore lists only the yeast stores it will read, and the live rebuild's preflight, which calls the same function with `KG_ORGANISMS`, does not block on a stale bacterial store ([[database.slurm.scripts.gilahyper_live_rebuild-slurm_docker]]).

The deviation from the request: the organism option is on `build_dataset_lmdb --list-stale`, not on `python -m torchcell.provenance.build_manifest`. That CLI has no `--list-stale`; it scans every built directory under `DATA_ROOT/data/torchcell/` whether or not it is mapped, and it sits below `torchcell.data` in the import graph (`experiment_dataset` imports it), so it cannot read a loader's organism without an import cycle. The map-walking list is this one, and it is the one the preflight runs.

Measured on the shared dev tree on 2026-10-12, with `--include-private`:

| `--organisms` | stores listed |
|---|---:|
| (absent) | 55 |
| `scerevisiae` | 3 (`Boocock2025Dataset`, `TfDeletionPseudobulkJackson2020Dataset`, `InhibitorBioscreenVolk2021Dataset`) |
| `ecoli` | 24 |
| `pputida` | 28 |

3 + 24 + 28 = 55, and the union of the three lists equals the unfiltered list name for name. Without `--include-private`, `--organisms scerevisiae` lists 2 (the private Bioscreen store drops out). The same unfiltered command run from the primary checkout on `main` (`eb07bd213`) lists the same 55 stores, so declaring `organism` on 154 loader classes made no store stale: the build manifest's closure is computed from `torchcell.datamodels` imports alone, and `Organism` lives in `torchcell.data`.
