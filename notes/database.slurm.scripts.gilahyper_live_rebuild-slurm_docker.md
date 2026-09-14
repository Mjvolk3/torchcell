---
id: rz4gzreqvak8f03yik9rsko
title: Gilahyper_live_rebuild Slurm_docker
desc: ''
updated: 1789369975374
created: 1789369975374
---

## 2026.09.14 - The live full rebuild: build beside the served store, then swap

`database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm` is the full-rebuild path
that keeps `tc-neo4j-readonly` serving while the new store is built. The in-place script
(`gilahyper_uncapped_build-slurm_docker.slurm`, job 1558) stops the serving container first,
so a full build meant a day of downtime with no way back; this one has minutes of downtime and
keeps the old store on disk.

### Stages

1. **Preflight.** The serving container must be running (its node and Dataset counts are
   recorded), `/db` must have at least `MIN_FREE_GB` (1,500 G) free, `$NEXT_ROOT/data`
   (`/db/database-next/data`) must be empty, and every dataset in `dataset_adapter_map` must
   have a dev-tree LMDB whose `preprocess/build_manifest.json` reads fresh against the
   schema surface of the checkout (`check_manifest`). The checkout named by `TORCHCELL_SRC`
   must match `BUILD_COMMIT` (default `origin/main`) under `torchcell/` and `biocypher/`,
   because the container installs torchcell from GitHub at that commit and the manifest
   bootstrap fingerprints the checkout. `directory_setup` refreshes the build tree's
   BioCypher config and container `.env`.
2. **Generate.** An ephemeral `--entrypoint bash` container (the incremental runner's
   pattern) runs `create_scerevisiae_kg_small --config-name kg_uncapped` with the DEV tree's
   `data/torchcell` mounted read-only over the build tree's copy, so the CSVs come from the
   stores L0-L4 verification ran on rather than from staged copies. Genome and graph caches
   (`data/sgd`, `go`, `string`, `tflink`) still come from the build tree. CSVs land in
   `$SERVE_ROOT/biocypher-out/<ts>` (7474-owned), the directory the serving container
   mounts. The generation log is `$BUILD_ROOT/database/slurm/output/<job>_generate.log`;
   the adapter count in its final "across N adapters" line must equal the map size.
3. **Import.** `tc-neo4j-build` (image entrypoint, no host ports) mounts a FRESH data root
   `$NEXT_ROOT/data`; the generated `neo4j-admin-import-call.sh` runs as uid neo4j, then
   `CREATE DATABASE torchcell`, a stop/start and one count query (the index-stats file the
   read-only container needs).
4. **Validate.** Before anything served is touched: `count(d:Dataset)` equals the map size,
   and the per-dataset `ExperimentMemberOf` counts in the store equal the rows in the CSVs
   (`<job>_expected.txt` vs `<job>_live.txt`, diffed into `<job>_validation.txt`).
5. **Swap.** Stop and remove both containers; `mv $SERVE_ROOT/data ->
   data.superseded.<ts>` and `mv $NEXT_ROOT/data -> $SERVE_ROOT/data` (same filesystem,
   two renames); relaunch `tc-neo4j-readonly` with the arguments of
   `scripts/migrate_storage_tiers.sh` (16 CPU, 200 G, `biocypher-out` mounted); wait for
   `online`; the served count must equal the validated count. Rollback is the reverse pair
   of `mv` and a relaunch; the superseded store stays until someone removes it.
6. **Record.** The production manifest moves to `kg_manifest.json.superseded.<ts>` and a
   new one is bootstrapped from the live store; the CSVs are archived to
   `/bulk/biocypher-out/<ts>`.

### Why generation is not in the neo4j container

The image entrypoint runs under `bash -eu` and does `chown -R neo4j:neo4j $NEO4J_HOME`
when started as root. A read-only bind mount anywhere under `/var/lib/neo4j` makes that
chown fail and the container restart-loops (the shadow-mount crash loop recorded in
[[torchcell.knowledge_graphs.incremental-admission]]). Mounting the dev tree read-write
instead would have the entrypoint chown the dev user's stores to uid 7474. So the dev tree
is only ever mounted into a container that bypasses the entrypoint. The same hazard applies
to the `torchcell-genomes:ro` mounts added to the in-place build scripts on 2026.09.14; they
have not been run since and are superseded by this script for full builds.

### Resources

32 CPUs, 200 G, 4 days. `MaxMemPerCPU=4100` makes 200 G a 50-CPU floor, so the job starts
beside the usual four 4-CPU/60 G GPU jobs, which 320 G would not. Job 1558 (36 datasets,
44.4 M records, 32 CPUs) took 21 h 22 min end to end.

### First run

Queued as job 1897 on 2026.09.14 behind the stale-store rebuild chain recorded in
[[user.Mjvolk3.torchcell.tasks.weekly.2026.38.live-rebuild]]; its outcome is recorded there.
