---
id: 7z3p927hkzardqq5r01hfax
title: Kg_generation_array_conductor
desc: ''
updated: 1792055272817
created: 1792055272817
---

## 2026.10.15 - Start a per-adapter generation: directory, order, fence, one sbatch per tier

`database/scripts/kg_generation_array_conductor.sh` is the host-side start of the generation array ([[database.slurm.scripts.gilahyper_generate_adapter_array-slurm_docker]]). Given `OUT_NAME` (and `TORCHCELL_SRC`, `BUILD_COMMIT`, `KG_ORGANISMS`, `INCLUDE_PRIVATE`, `KG_CONFIG`, `GEN_OVERRIDES` as the live rebuild sets them) it:

1. creates `$SERVE_ROOT/biocypher-out/$OUT_NAME/adapters` through a container and hands it to the dev user (biocypher-out itself is uid 7474's, so the merge can later write `merged/`, the import call and the report there);
2. writes `adapter_order.json` (`generation_array order --write`) and `dev_fence.txt` (`generation_array fence`) into it;
3. runs `generation_array submit`, one `sbatch` per tier with the tier's sizes and `%N`, the tier's `GEN_CPUS` / `GEN_MEM_GB` and the multi-member values in the environment under `--export=ALL`;
4. prints `JOB_IDS=<a>,<b>,<c>` as its last line, which the live rebuild chains its follow-up on with `--dependency=afterany`.

`DRY_RUN=1` prints every command and submits nothing (the mkdir, the two writes and the sbatch lines, each prefixed `DRY-RUN`). `INDICES=<slurm array spelling>` restricts the submission to those positions of the order: the re-run of failed tasks (the live rebuild's `RESUME_FROM=array` passes the missing indices), or one adapter for a measurement run, in which case a missing directory is created and an existing one is reused with its order and fence.

Tests: the text pins in `tests/torchcell/database/test_generate_adapter_array_slurm.py`; the dry run was exercised by hand on 2026-10-15 and listed 170 adapters in three arrays (standard 157, large 12, hoepfner 1).
