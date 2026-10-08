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

## 2026.09.17 - Memory envelope measured, and a resume mode after the restart bug

Three full-build attempts fix the resources paragraph above, which was written from the
41 G private-memory profile and is wrong: the generation container's anonymous memory
scales with the adapter worker count, which `SLURM_CPUS_PER_TASK` sets.

| job | CPUs / memory | outcome |
|---|---|---|
| 1996 | 32 / 200 G | OOM-killed 15 min in at the Costanzo dmf experiment chunk; memcg anonymous memory 206 G, one worker at 45 G (kernel log) |
| 2031 | 32 / 320 G | never started; pended on Priority behind queued 60 G GPU jobs, cancelled |
| 2032 | 24 / 256 G | generated all 51 datasets (472,298,282 node rows, 525,064,964 edge rows) in 28 h 40 min, container sampled between 84 and 155 GiB; imported 99,723,455 nodes and 361,895,230 relationships in 16 min 55 s; FAILED at the post-import restart |

The defaults are now 24 CPUs and 256 G. The 256 G request leaves the node room for the
GPU sweeps, which is why the user chose it over 320 G.

**The restart bug.** After `CREATE DATABASE` the script stopped and started neo4j to get
the index-stats file written, with `docker exec tc-neo4j-build neo4j stop` then
`neo4j start`. The image entrypoint runs neo4j as PID 1, so the stop ended the container,
and with `--restart=no` it stayed down; the `neo4j start` that followed failed under
`set -e` and the job exited 9 at 00:36 CDT. The in-place script survived the same pair
only because its container has `--restart=unless-stopped`, so docker relaunched it. The
fix restarts the container (`docker stop` then `docker start`) and fails loudly if the
database does not come back online. The imported store was intact: the debug log shows
`torchcell` STARTED at 05:36:25 UTC and checkpointed cleanly at shutdown.

**Resume mode.** `sbatch --export=ALL,RESUME_JOB=<old job>` skips generation and import.
It reads the generation commit, the dataset count, and the CSV directory from the old
job's `.out` and `_generate.log`, requires the `IMPORT DONE` line, the build container
`exited`, the generation container gone, and `$NEXT_ROOT/data/databases/torchcell`
present, then continues from the restart: count, validate, swap, record. The manifest is
bootstrapped at the generation commit (`kg_manifest` reads the schema surface and
adapters with `git show` at that ref), so the checkout may be ahead of the build commit;
the fresh-run preflight's HEAD-equals-commit check is not applied on a resume. Job 2032
is resumed this way rather than regenerated.

**Second resume failure, the CSV directory mode.** The first resume (job 2311) restarted
the build container, counted 99,723,455 nodes and 51 Dataset nodes, then failed reading
the CSVs on the host: `PermissionError` on `/db/database/biocypher-out/2026-09-16_00-44-53`.
The image entrypoint runs `chmod 700 $NEO4J_HOME` and `chmod -R 700` on every top-level
directory under it at each container start (lines 394 to 395 of
`/startup/docker-entrypoint.sh`), and `biocypher-out` is mounted there in both the build
and the serving container, so the dev user loses read access the moment either starts.
The preflight's `chmod 755` was undone by the build container's first start at 00:18 CDT.
The validate stage now reopens the directory (`chmod 755` on `biocypher-out`,
`chmod -R a+rX` on the CSV directory) from a root ephemeral container right before the
host-side read, and the resume preflight accepts a `running` build container as well as
an `exited` one. After the swap the serving container closes the directory again; the
archive step copies inside a container as uid 7474 and is unaffected.

**Third resume failure, the count query.** Job 2325 read the CSVs (51 expected rows) and
then failed on the live-count query: Neo4j 5.26 rejects
`RETURN d.id + '=' + toString(count(e))` as an aggregation expression that contains an
implicit grouping key. The query now returns `d.id` and `count(e)` as two columns and the
shell joins them; both sides are sorted in the C locale. Run by hand against the build
container before the fix landed, the rewritten query matched the CSV counts for all 51
datasets (`diff` empty), so the store is validated and the next resume proceeds to the swap.

**Fourth resume, the swap, and one more poll bug.** Resume job 2331 (4 CPUs, 16 G; a
resume needs no more, and 256 G pends on its 63-CPU floor behind 24-CPU jobs) validated
the store (live == CSV for all 51 datasets), swapped at 20:35 CDT, and relaunched
`tc-neo4j-readonly` on the new store. The job then died on the first status poll: under
`set -e` with `pipefail`, `STATUS=$(docker exec ... | tail | tr)` exits the script when
cypher-shell fails, and it fails until neo4j accepts connections. Serving came up on its
own within the minute. The loop now waits 60 s first and tolerates a failing poll. Stage 5
was run by hand with the script's own commands: the production manifest moved to
`kg_manifest.json.superseded.2026-09-17_20-35-08` and a new one bootstrapped from the live
store at commit 7715ee35 (51 datasets), and the CSVs copied to `/bulk/biocypher-out/`.

### Served after the swap (2026.09.17 20:35 CDT)

| | before | after |
|---|---|---|
| nodes | 83,066,931 | 99,723,455 |
| datasets | 36 | 51 |
| store | `data.superseded.2026-09-17_20-35-08` (535 G, rollback) | `/db/database/data` (682 G) |
| generation commit | job 1558 + Nadal increment | 7715ee35 |
| CSVs | `biocypher-out/2026-09-11_07-02-12` | `biocypher-out/2026-09-16_00-44-53` (634 G) |

## 2026.09.18 - Default database and the seeded Browser image

Two additions after the swap, both applied by hand to the served store first:

- **Default database.** After the imported `torchcell` is online in the build container,
  the script runs `STOP DATABASE neo4j; CALL dbms.setDefaultDatabase('torchcell'); START
  DATABASE neo4j` (guarded on `SHOW DEFAULT DATABASE`, so a resume of a store that already
  has it does nothing) and asserts the default both there and on the served store after
  the swap. The setting is in the system database under `data/`, so it travels with the
  store. On the served store it was set by hand on 2026.09.18 (`neo4j` has 0 nodes; the
  `neo4j` database is started again afterwards and stays online, read-only).
- **Image.** `IMAGE` defaults to `michaelvolk/tc-neo4j:5.26.28-browser.1`, the base image
  plus the Browser styling seed ([[database.docker.Dockerfile.tc-neo4j-browser]],
  [[torchcell.database.browser_style]]). After the swap the script fetches
  `http://localhost:7474/browser/` and fails if the page has no `torchcell-seed.js` tag,
  which would mean the serving container came up on an unseeded image. The serving
  container was relaunched on the new tag by hand (03:02:22 stop, online 03:03:03, 41 s).

## 2026.09.19 - Releases: content hashes, the KgRelease node, aliases, the /kg-releases mount

After validation the script streams every dataset's experiment ids through
`database/scripts/kg_content_hashes.sh` against the build container and writes
`<job>_content_hashes.json`. After the swap and the manifest bootstrap it stamps the
manifest (`releases stamp --kind full`: major bumped from the superseded manifest's
version, release id `<build date>-<commit>`, the hashes), opens the served database for
one write through `server.databases.writable`, writes the `KgRelease` node, closes it
again, creates the `latest` and `pinned` aliases, and prints the release table. The
serving container now also mounts `RELEASE_ROOT` (`/bulk/kg-releases`) at
`/kg-releases`, where `scripts/kg_release.sh backup` writes the online backup; on
2026-09-19 the container was relaunched by hand with that mount (37 s) after the first
backup, staged on /db, had to be killed for space. See [[torchcell.knowledge_graphs.releases]],
[[scripts.kg_release]], [[plan.kg-releases.2026.09.19]].

## 2026.09.21 - Hand the build-tree conf and biocypher dirs back before directory_setup

A full rebuild leaves `$BUILD_ROOT/database/conf` and `biocypher` owned by 7474 at mode 700, because the build container mounts them under `NEO4J_HOME` and the image entrypoint chowns and chmods them at start. The next job's host-side `directory_setup`, running as the dev user, then cannot rewrite `neo4j.conf`; increment job 2675 died at stage 3 that way after the 2026-09-17 rebuild. This script and [[database.slurm.scripts.gilahyper_increment_kg-slurm_docker]] now chown both directories back to the invoking user through a root container before the refresh (main `1fbd6f883`).

## 2026.10.02 - Biolink head ontology mirrored in the repo, the build is hermetic (issue #619)

Live rebuild job 3182 (2026-10-02 02:12) died 4 min in with `HTTP Error 504: Gateway Time-out` from `rdflib`: no BioCypher config under `biocypher/config/` set `head_ontology`, so every BioCypher construction fetched its packaged default, `https://github.com/biolink/biolink-model/raw/v3.2.1/biolink-model.owl.ttl`. Job 3198 ran once GitHub answered again.

Mirrored, not headless. Headless (`head_ontology: null`) would change the served graph: every node carries Biolink ancestor labels beside its class label (`torchcell.database.browser_style.ANCESTOR_LABELS`, read from `CALL db.labels()` on the 2026-09-18 store), the `:LABEL` column the Neo4j writer emits is BioCypher's `get_ancestors` list, and the schema's `is_a` targets (`phenotypic feature`, `information content entity`, `environmental exposure`, `nucleic acid entity`, `participates in`, `part of`, `mentions`) are Biolink classes. Under the mirror, `fitness phenotype` resolves to `fitness phenotype, phenotypic feature, disease or phenotypic feature, biological entity, named thing, entity`; four of those six are not schema classes and exist only through Biolink.

- File: `biocypher/ontology/biolink-model-v3.2.1.owl.ttl`, 451318 bytes, sha256 `a3e47f5a42e4e0e6b4c715f8f4dbbeef5e77683621fa9ad57d15b7d048707d1a`, fetched 2026-10-02T21:49:16Z with `curl -fsSL --retry 3 -o biocypher/ontology/biolink-model-v3.2.1.owl.ttl https://github.com/biolink/biolink-model/raw/v3.2.1/biolink-model.owl.ttl`.
- Provenance: `biocypher/ontology/biolink-model-v3.2.1.owl.ttl.provenance.json`, a `HeadOntologyMirror` (`torchcell/knowledge_graphs/head_ontology.py`) wrapping the shared `ArtifactRecord` + `RetrievalRecord` (retriever `torchcell.literature.retrieve.direct_url`). Its `last_check` re-ran that retriever at 2026-10-02T21:51:28Z and produced the same sha256. The record was written by `python -m torchcell.knowledge_graphs.head_ontology record`.
- Configs: `linux-amd` (the one `database.env` names), `biocypher_docker` and `costanzo_biocypher_docker` set `head_ontology.url: /var/lib/neo4j/biocypher/ontology/biolink-model-v3.2.1.owl.ttl`, the path inside the generation container: `directory_setup` copies the repo `biocypher/` (ontology included) to `$BUILD_ROOT/database/biocypher`, which this script mounts at `/var/lib/neo4j/biocypher`. The host configs `linux-arm` and `costanzo_biocypher` use the repo-relative `biocypher/ontology/...`, resolved against the repo root. All set `root_node: entity`, BioCypher's default.
- Check: `create_scerevisiae_kg_small.main` calls `verify_head_ontology(BIOCYPHER_CONFIG_PATH)` before constructing BioCypher. It raises when `head_ontology.url` is unset, has any `scheme://`, names a missing file, lacks the provenance record, has a different root node, or hashes to anything but the pinned sha256 and size.
- Equivalence, measured: `experiments/036-dataset-fixes-before-kg-build/scripts/biolink_head_ontology_equivalence.py` (biocypher 0.15.2, the version the container installs) built the BioCypher `Ontology` over `torchcell_schema_config.yaml` from the local file and from the URL: 55 schema classes each, 0 with a differing ancestor list; local and re-fetched sha256 identical. Result: `experiments/036-dataset-fixes-before-kg-build/results/biolink_head_ontology_equivalence.json`. Node labels and the `kg_manifest` graph-schema fingerprint (computed from `torchcell_schema_config.yaml` alone) do not change.
- Tests: `tests/torchcell/knowledge_graphs/test_head_ontology.py`; `test_create_scerevisiae_kg_small.py` now writes a config naming the mirror and pins the refusal of a config without one.

Open: `create_kg.py`, `create_scerevisiae_kg.py` and the other legacy build modules read the same configs (so they no longer fetch) but do not call the check; no slurm script runs them by default. The untracked, gitignored `PRODUCTION_linux-amd_biocypher_config.yaml` that `database/scripts/create_readonly_users.sh` names is not covered.

## 2026.10.06 - Pairing gate: the build commit must carry a package tag

Right after `BUILD_COMMIT` is fixed, the script requires `git describe --tags --exact-match --match 'v[0-9]*' $BUILD_COMMIT` to succeed and the fingerprint checkout's HEAD to be that commit, else it exits before touching anything. Reason: KG 2.0 and 3.0 were built from an untagged `main` (`1.6.1 (untagged)` in the status table) and no published package could read the store; the repair was `releases retag` plus a `DB(kg)` release cut after the fact. The recipe is now: land the work, push a `REL` (minor) or `DB`/`PATCH` (patch) commit, wait for semantic-release's `vX.Y.Z` bump commit on `main`, fast-forward the build checkout to it, submit. The snapshot commit after the build is `DB(kg): <release> snapshot ...`, never `RELEASE(kg)`, which is not an allowed tag.

## 2026.10.08 - Record the artifact pointer set before the stamp

Stage 5 runs `kg_manifest ... artifact-refs --data-root "$DEV_DATA_ROOT"` between the bootstrap and `releases stamp`, so the stamped manifest, the `KgRelease` node (`write-node`) and the committed snapshot carry each dataset's file-level `ArtifactRef` set, which `make ops` checks against tc-data. Only datasets whose closure names `ArtifactRef` are opened. Cost, estimated and not yet measured on a build: the walk is 0.26 ms per record, and on main at `1b180c7e8` thirteen loaders' closures name `ArtifactRef` (bloom2019, caudal2024, lian2019, mormino2022, smith2016, choe2025, cui2018, rapp2026, rousset2018, wang2018, carruthers2025, menasalvas2025, yunus2026), so the step walks all of their records once per release; Bloom 2019 alone (530,100 records) is about 2.3 min of walking plus the LMDB reads and `transform_item`. The step runs after the swap, under `set -euo pipefail`: a walk failure aborts before stamp, write-node, the aliases and the archive, with the new store already serving, so the recovery is to fix the dev tree and rerun from the stamp.
