---
id: s9lz971gfkklgyv2r7lryzc
title: Kg_release
desc: ''
updated: 1789840374851
created: 1789840374851
---

## 2026.09.19 - Release artifacts: online backup, archive, ship, keep, purge

The release artifact is an Enterprise online backup of the served database
(`neo4j-admin database backup`, no downtime), written straight to
`/bulk/kg-releases/<release>/`, which the serving container mounts at `/kg-releases`.
The mount sits outside NEO4J_HOME because the image entrypoint chmods every top-level
directory under NEO4J_HOME to 700 at each start (the biocypher-out lesson of job 2311),
which would hide the archive from the host. Backups cannot stage on /db: the client
receives the entire store (680 G) into a work directory before it compresses, and the
first attempt on 2026-09-19 (staged under biocypher-out) ran /db from 2.8 T free to
151 G while the 029 build had just landed 2.3 T there; it was killed at 373 G and its
work directory moved to `/bulk/deprecated/2026.09.19/` for deletion. The container was
relaunched with the /bulk mount (37 s of downtime) and the live rebuild script mounts it
from now on (`RELEASE_ROOT`).

`backup` starts the client detached inside the container as the neo4j user and names
the directory after the store's KgRelease node; `archive` adds `kg_manifest.json`,
`release.json` (the node's properties) and `SHA256SUMS`; `ship` rsyncs a release to
Taiga (`/mnt/zhao5/mjvolk3/projects/torchcell/kg-releases/`, through the
torchcell-database VM, 85 MB/s measured on the Parameter_Estimation transfer) and
verifies the checksums there; `keep <release> "<why>"` writes a `KEEP` file on both
copies; `purge [--dry-run]` removes untagged releases older than `KEEP_DAYS` (60) from
both, reading the age from the release id's date prefix. Serving several releases at
once (a loaded pinned beside latest) is the VM's story once it has a block volume; on
GilaHyper the archive is the version history and the DBMS holds one store.

## 2026.09.19 - First release artifact, measured

`kg_release.sh backup` on the served 1.0 store (680 G on /db, 99,723,456 nodes): the
client received the store files for 1 h 15 min and compressed them for 41 min, 1 h 57 min
in all, while serving continued. The artifact
`torchcell-2026-09-19T19-16-19.backup` is 14,545,906,536 bytes (14.5 GB), sha256
`bf613c08027b58c9142418989a30a8e6c0aaf213bcd6c4e2e9aef47fea65b91a`;
`neo4j-admin database backup --inspect-path` reads it as FULL, COMPRESSED, database id
`f55a3106-e65b-440a-ab75-9bedd06bc5c4`, transactions 1 to 15. The 47x ratio against the
store on disk is the block format's reserved space and the repeated `serialized_data`
text compressing; a restore test on a second DBMS is still owed before a release is
trusted as the sole copy. The directory carries `kg_manifest.json`, `release.json` and
`SHA256SUMS`; the extractor behind `release.json` reads `status --json` as the per-host
map it became (fixed here). The backup work directory is a full uncompressed copy of the
store: 680 G of scratch on /bulk for two hours, none on /db.

`kg_release.sh ship 2026.09.17-7715ee35` copied the 14.5 GB directory to
`rocky@141.142.216.218:/mnt/zhao5/mjvolk3/projects/torchcell/kg-releases/2026.09.17-7715ee35/`
in 2 min 5 s at 111 MB/s and `sha256sum -c` on Taiga reported all three files OK
(2026-09-19 15:10 CDT). The first pass ended with rsync exit 23 after a complete copy
because `-a` tried to chgrp files in Taiga's setgid group directory; `-rlpt` now.
`purge --dry-run` keeps the release (younger than 60 days).

## 2026.10.06 - deploy, restore-test and retag: the served version follows a release

The archive side of a release (`backup`, `archive`, `ship`) moved the 3.0 artifact to Taiga on 2026-10-06, and nothing then moved a served database: Radiant kept serving its 35-dataset store with no release node, and `make ops` reported DIVERGED with no step to act on. Three subcommands close that gap.

- `deploy <release> [--pin]` serves an archived release from the host it runs on. It pulls the release directory from Taiga when `$ARCHIVE_ROOT` lacks it, verifies `SHA256SUMS`, runs `neo4j-admin database restore` inside the serving container into a new database `kg-<release>` (dots become dashes: `kg-2026-10-06-4b293d34`) while the DBMS stays online, `CREATE DATABASE ... WAIT`, then compares the restored `KgRelease` node (release id, KG version, dataset count) and the `Dataset` node count with `release.json`. Only then does it retarget the `latest` alias (and `pinned` with `--pin`); the previous database stays for rollback (`ALTER ALIAS latest SET DATABASE <name>`). It ends with `ops.sh sync`, the one ops action with an exit code (1 on DIVERGED or on a host serving a store without a release node). A restore needs `MIN_FREE_GB` (300) free on the container's `/data`.
- `restore-test <release>` is the same restore and the same checks into `kgtest-<release>`, dropped at the end (`DROP DATABASE ... DESTROY DATA`), aliases untouched. While the throwaway database exists the DBMS holds two databases with the same `KgRelease` node, so a client naming the release id (not an alias) would see two candidates for those minutes; `latest` and `pinned` are unaffected.
- `retag <release> <tag>` repairs a release built from an untagged commit: `releases retag` rewrites the committed snapshot and the manifest (refused unless the schema surface at the tag reproduces every served closure), then the `KgRelease` node is rewritten from the manifest under the writable toggle. The snapshot then goes into a `DB(kg): ...` commit and the compatibility page is regenerated.

All three touch the DBMS, so on GilaHyper they run under slurm, `sbatch -c 4 --mem=16G --wrap "bash scripts/kg_release.sh <cmd> ..."`. On Radiant, `deploy` is what makes the served version follow a release, but only once its store sits on a block volume: Neo4j does not run a store on NFS, and the VM has no other disk, so the Taiga copy is read there as the archive, not served from.

**Restore test measured (2026-10-06, jobs 3338 and 3339).** `restore-test 2026.10.06-4b293d34` on the 11,175,064,969-byte artifact: checksums OK, `neo4j-admin database restore` unpacked the artifact into a 131 G temp directory beside it on /bulk (8 min 0 s, 21:25:32 to 21:33:29 UTC) and copied it into `/data/databases/kgtest-2026-10-06-4b293d34` (2 min 14 s; 151 G on /data), "completed successfully" at 21:35:43, 10 min 13 s in all; `CREATE DATABASE ... WAIT` came up `CaughtUp` while `torchcell` kept serving. The script's own check then crashed on a wrong key (`release.json` is the `KgRelease` model with a `datasets` map, not the node's flat `n_datasets`; fixed) before the drop, so job 3339 ran the checks by hand and dropped the database: the restored store carries `KgRelease` `2026.10.06-4b293d34`, version `3.0`, 51 datasets (node property and `Dataset` count agree), 100,127,516 nodes, and `torchcell_version 1.6.1 / torchcell_tag null`, the pre-retag node, as the artifact was taken before the pairing. `DROP DATABASE ... DESTROY DATA WAIT` returned /data to 3.4 T used. The 3.0 artifact is therefore proven restorable, and the deploy path (restore, create, check, alias) works against a live DBMS.

The restore temp directory is created by `neo4j-admin` beside the artifact (`<release>/torchcell-temp-extracted-artifacts-0`, the full uncompressed store, on /bulk) and removed when the restore finishes; a restore killed midway leaves it there, and `sha256sum -c` still passes because the directory is not listed.

## 2026.10.12 - Release 3.0 deployed to Radiant by hand, onto the NFS volume

The user asked for the 51-dataset graph on Radiant now, on the Taiga-backed volume the July store already ran from, without waiting for a block volume. `cmd_deploy` targets the GilaHyper container (`NEO4J_CONTAINER`, `/kg-releases` staging mount), so the same steps ran by hand against `tc-neo4j` on Radiant, all measured:

1. Staged the artifact: `rsync --rsync-path="sudo rsync"` of `/bulk/kg-releases/2026.10.06-4b293d34/` (11,175,064,969-byte backup plus `SHA256SUMS`, `release.json`, `kg_manifest.json`) into `/mnt/delta_bbub/mjvolk3/torchcell_openstack/neo4j-data/staging/2026.10.06-4b293d34` (`/data/staging/...` inside the container), 110 MB/s, under two minutes; `chown 67392:202` (the container's uid). The parent directory is not traversable by `rocky`, so the checksum ran inside the container: all three OK.
2. `docker exec tc-neo4j neo4j-admin database restore --from-path=/data/staging/<release>/<file>.backup kg-2026-10-06-4b293d34` with the DBMS online: 13:41:56 to 13:50:32 UTC, 8 min 36 s, 151 G under `/data/databases/kg-2026-10-06-4b293d34` (the restore extracts to a temp directory beside the artifact first, then moves it). `restore_metadata.cypher` was not run: it carries role grants and Radiant runs with `dbms.security.auth_enabled=false`.
3. `CREATE DATABASE \`kg-2026-10-06-4b293d34\` WAIT` (14 s). KgRelease node: `2026.10.06-4b293d34`, 3.0, 51 datasets, 100,127,515 nodes; `MATCH (d:Dataset)` 51. The backup predates the retag of 2026.10.06, so the node was set to what GilaHyper's carries: `torchcell_version = '1.6.2'`, `torchcell_tag = 'v1.6.2'`. Then `ALTER DATABASE ... SET ACCESS READ ONLY` (Radiant's `server.databases.default_to_read_only=false` and no auth would otherwise leave a writable public graph).
4. The name switch: Neo4j 5 has no rename, so `DROP DATABASE torchcell DUMP DATA WAIT` (the July store), then `CREATE ALIAS torchcell | latest | pinned FOR DATABASE kg-2026-10-06-4b293d34` and `CALL dbms.setDefaultDatabase('kg-2026-10-06-4b293d34')`. The drop reported `AccessDeniedException: /var/lib/neo4j/data/dumps` (the container cannot write the dumps directory, root-owned) and removed the database from the system catalog without the dump, yet the files stayed: `/data/databases/torchcell` and `/data/transactions/torchcell` are still on disk (49 G) and the volume usage did not move. Two July dumps also exist (`torchcell_openstack/dumps/torchcell.dump`, 2026-07-21, 2.48 GB, and `neo4j-data/dumps/torchcell.dump`, 2026-07-22, 2.77 GB), so the July graph is recoverable three ways; the orphan directories can go once nobody wants it.
5. `server.memory.pagecache.size` 512M to 24G and heap 8G/8G in `~/projects/torchcell/database/conf/neo4j.conf` (backup `neo4j.conf.bak-2026-10-12`), `docker restart tc-neo4j`, online again in 15 s, `debug.log` confirms the three values.

Result: `make ops` reads `radiant kg-2026-10-06-4b293d34 [default] 3.0 2026.10.06-4b293d34 v1.6.2 ... 51 ... 100,127,516 latest,pinned,torchcell online` and `sync: in sync (2026.10.06-4b293d34)`. A cold `MATCH (e:Experiment) RETURN e.id LIMIT 1` through the alias answered in 2.6 s, no I/O fault on this store so far (one read, not a soak). Volume: 70 G free of 500 G, so the next release needs the July directories gone or the block volume first.
