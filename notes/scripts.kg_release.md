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

## 2026.10.13 - Deploy from the archive mount: Radiant serves from Taiga zhao5

KG 4.0 (three organisms, about 165 datasets, Hoepfner alone ten times its 3.0 size) does not fit beside 3.0 on the 500 G export Radiant serves from today (70 G free). The owner, 2026-10-12: "We shouldn't need a TB block volume, the data is stored on Taiga" and "the db size is going to increase too, so we need a way to do direct transfer to the mount." `kg_release.sh` now restores an archived release in place on the Taiga mount the archive already sits on and serves the store from a directory on that mount.

### Radiant as measured (2026-10-13, 05:01 to 05:11 UTC, read-only inspection over ssh)

| mount | export | size | used | free |
|---|---|---|---|---|
| `/` | `/dev/vda5` (xfs) | 39 G | 36 G | 3.8 G |
| `/mnt/delta_bbub` | `taiga-nfs.ncsa.illinois.edu:/taiga/nsf/delta/bbub` (nfs4) | 500 G | 431 G | 70 G |
| `/mnt/zhao5` | `taiga-nfs.ncsa.illinois.edu:/taiga/illinois/eng/chbe/zhao5` (nfs4) | 50 T | 43 T | 8.0 T |

- **`/mnt/delta_bbub` is not a block volume.** It is a 500 G Taiga NFS export, so KG 3.0 has been served from NFS since 2026-10-12. Both mounts come from the same server (141.142.224.36) with identical options: `vers=4.2, rsize=wsize=1048576, hard, proto=tcp, timeo=600, local_lock=none`. Moving the store to `/mnt/zhao5` changes the quota it draws on, not its storage class. The 8.0 T free on zhao5 is the shared allocation of the zhao5 group.
- **The container.** `tc-neo4j`, image `michaelvolk/tc-neo4j:5.26.28`, user `67392:202`, `--restart unless-stopped`, ports 7473, 7474 and 7687, environment beyond the image's own: `NEO4J_ACCEPT_LICENSE_AGREEMENT=yes` only. It was created by a plain `docker run` on 2026-07-22 (no compose labels) and last started 2026-10-12 14:43 UTC. Bind mounts: `/data` <- `/mnt/delta_bbub/mjvolk3/torchcell_openstack/neo4j-data`, `/logs` <- `.../neo4j-logs`, `/var/lib/neo4j/conf` <- `/home/rocky/projects/torchcell/database/conf` (rw), `/var/lib/neo4j/certificates`, `/plugins`, `/.env` (ro) from the same `database/` directory. There is no `/kg-releases` mount. The conf: `dbms.security.auth_enabled=false`, `server.databases.default_to_read_only=false`, bolt TLS `REQUIRED`, page cache 24G, heap 8G, query log `VERBOSE` at threshold 0. VM: 16 CPUs, 62 G RAM (47 G available).
- **What delta_bbub holds** (`du -sh`): `databases/kg-2026-10-06-4b293d34` 151 G, the dropped July `databases/torchcell` 49 G (plus 258 M of transactions), `staging/2026.10.06-4b293d34` 11 G (the 2026-10-12 staging copy of the 3.0 backup), `neo4j-data/dumps` 2.6 G, `torchcell_openstack/dumps` 2.4 G, logs 30 M.
- **The archive on zhao5.** `kg-releases/2026.09.17-7715ee35/` (14,545,906,536-byte artifact) and `kg-releases/2026.10.06-4b293d34/` (11,175,064,969 bytes), owned `1000:555647` under setgid group directories (2770). `kg_release.sh ship` already writes here: rsync over ssh to the VM, whose NFS client writes the files straight onto Taiga, with nothing staged on the VM's disks. Ship is unchanged.
- **Permissions.** The container's uid 67392 (mjvolk3's NCSA uid, owner of `/mnt/zhao5/mjvolk3`) lists `kg-releases/` and passes `test -w` on `/mnt/zhao5/mjvolk3/projects/torchcell` with no supplementary group (`sudo setpriv --reuid=67392 --regid=202 --clear-groups`). Taiga resolves its groups server side, so the containers need no `--group-add`.
- **NFS read throughput**, `dd` with O_DIRECT, 16 MiB blocks, 4 GiB of the 3.0 artifact: zhao5 488 MB/s, delta_bbub 468 MB/s. Write throughput was not measured here (read-only inspection only). The recorded figure is the 2026-10-12 restore: 151 G written onto delta_bbub in 8 min 36 s, about 0.3 GB/s for extract plus move on one mount.
- **The NFS caveat.** Counting `Input/output error` lines in `debug.log` per day gives 205 on 2026-10-07 and 173 on 2026-10-08, the last day with any. Those faults hit the July store at a 512M page cache. Since the 3.0 deploy (restart 2026-10-12 14:43 UTC) there are none: `debug.log` has had no line since 14:43:37 that day, and none of the 2,360 `query.log` lines dated 2026-10-12 or 2026-10-13 contains "exception" or "error". That is 14.5 h of light serving, not a soak. The store runs exactly as the memory records: NFS, read only, auth off.

### What the script does now

- `deploy <release> --from-archive-mount [--pin] [--dry-run]` (run on GilaHyper; it drives the VM over ssh). It verifies `SHA256SUMS` where `ship` put the release, reads `release.json` from there, and requires the committed snapshot `database/releases/<release>.json`. It then restores in a detached helper container (`kg-restore-kg-<release>`, the serving image and uid, archive mounted read-only at `/kg-releases`, the serving conf read-only). `neo4j-admin database restore --temp-path=/data/restore-tmp-<db>` unpacks onto the same mount as `/data/databases`, so the finished store is moved into place, not written twice. Nothing crosses the ssh connection but commands and logs. If the serving container already runs on `$SERVE_DATA`, the database is then registered online: created, checked, paired, set READ ONLY, and `latest`, `torchcell` and the default database move to it (`pinned` too with `--pin`). Otherwise the run stops after the restore.
- `serve-swap --latest <r> [--pinned <r>] [--dry-run]` makes the one-time move. A staging DBMS (`tc-neo4j-staging`: no published port, 2G heap and 2G page cache beside the serving DBMS) starts on `$SERVE_DATA` with the serving conf mounted at `/conf`. The image entrypoint copies `/conf` before applying settings, so the host conf is never edited (md5 unchanged in the rehearsal). Staging registers each release, points the aliases and the default database, checks them, and is removed. Then `tc-neo4j` is stopped (`--time 300`), renamed `tc-neo4j-pre-<timestamp>` with `restart=no`, and relaunched from its own `docker inspect`: same image, user, ports, environment, conf, certificates, plugins and `.env`, with `/data` and `/logs` on zhao5 and the archive read-only at `/kg-releases`. Downtime is one stop and one start. Any unknown mount, a non-default network or a resource limit makes the relaunch refuse rather than drop something silently.
- `serve-point --latest <r> [--pinned <r>]` moves `latest`, `torchcell` and the default database (and `pinned`) between releases the archive-mount DBMS already serves. This is the alias-level rollback.
- `serve-rollback <tc-neo4j-pre-timestamp>` stops the archive-mount container, keeps it as `tc-neo4j-archive-mount-<timestamp>`, renames the kept container back and starts it on its untouched delta_bbub store.
- **The check** (`releases verify-restore`, now used by the GilaHyper deploy and `restore-test` too) runs one query. Its node total comes from the count store and each dataset's experiment count from the `ExperimentMemberOf` in-degree. It must equal `release.json` in release id, version, dataset set, every dataset's experiment count and the node total. A full build (`X.0`) stamps `n_nodes` before its release node exists and an admission after, so the expected total is `n_nodes + 1` or `n_nodes`. **Pairing:** the 3.0 artifact predates its retag (node 1.6.1, no tag), so `pairing-statement` sets the committed snapshot's `torchcell_version` and `torchcell_tag` before READ ONLY, the step the 2026-10-12 deploy did by hand. For 4.0 it is a no-op.
- `dbms.setDefaultDatabase` refuses while the old default runs ("The old default database neo4j is still running.", measured in the rehearsal), so the alias step stops the old default for that one call and starts it again. A release `pinned` names is unavailable for those seconds.
- `docker stop` uses `--time 300`: with Docker's 10 s default the toy DBMS was killed (exit 137) before it finished shutting down.

### Tests

- **Restore check against the real store.** Slurm job 4642, read-only, ran the restore check against GilaHyper's served `torchcell` (release 3.0) and compared it with `/bulk/kg-releases/2026.10.06-4b293d34/release.json`: passed, 51 datasets with release.json's experiment counts, 100,127,516 nodes, the query took 1.9 s. In the same job, `deploy 2026.10.06-4b293d34 --dry-run` (GilaHyper mode) verified the checksums and found the name free with 3,195 G free on `/data`.
- **Radiant dry runs, read-only.** `deploy 2026.10.06-4b293d34 --from-archive-mount --dry-run` took 45 s, most of it `sha256sum -c` of the 11 G artifact; all three files OK. It reported that `tc-neo4j` runs on delta_bbub (restore-only mode), that the two zhao5 directories would be created as `67392:555647` with mode 2770 (passwordless sudo present), and 8,121 G free. `serve-swap ... --dry-run` printed the staging command and the relaunch command shown above and refused on "NOT restored". A quoting round trip of a Cypher statement with backticks, quotes, `$`, `*` and `!` and of a Go template with tabs came back byte-identical through `rocky`'s zsh.
- **Rehearsal on GilaHyper** (toy containers, `SERVE_HOST=local`, scratch directories; no served store touched). The toy setup was an old DBMS serving a "3.0" (full build, node untagged) and three archived toy releases: a "3.0", a "4.0" and a "4.1" admission. Results:
  - Restores of the 3.0 and the 4.0 into `$SERVE_DATA` worked, with the temp directory removed afterwards.
  - `serve-swap --latest 4.0 --pinned 3.0` passed both checks, set the 3.0 node to the snapshot's 1.6.2/v1.6.2 and made both databases read-only. Downtime from stop to an online default was 34 s on the toy store.
  - An online `deploy` of the 4.1 into the swapped DBMS passed the admission node-count rule and kept the old default online.
  - `serve-point` back to the 4.0 and `serve-rollback` to the pre-swap container (clean exit 0) both worked.
  - Unit tests: seven new in `tests/torchcell/knowledge_graphs/test_releases.py`. Strict mypy is clean.

### Estimates (hypotheses, not measured)

- **4.0 restore time.** The 4.0 store size is the 2026-10-12 scratch estimate of 250 to 300 G (3.0 was 151 G for 52.7M experiments). Unverified: at the 3.0 rate of 0.3 GB/s the restore takes 15 to 20 min. The 3.0 restore onto zhao5 should take about 9 min, as on 2026-10-12.
- **Swap downtime.** Unverified: one Neo4j stop and one start with two large stores on NFS takes 1 to 3 min. The toy took 34 s.
- **Restore load on serving.** Unverified: the restore reads the archive at up to about 0.5 GB/s and writes the store on the same Taiga server the served store reads from, so cold queries may slow while it runs.

### Owner decisions

1. **Serving directory.** `SERVE_DATA=/mnt/zhao5/mjvolk3/projects/torchcell/neo4j-data` and `SERVE_LOGS=.../neo4j-logs` (the A0b names). The pair of stores (3.0 at 151 G plus 4.0 estimated at 250 to 300 G) draws on the shared zhao5 allocation (8.0 T free).
2. **3.0 as `pinned`.** By default 3.0 is restored from its archive into the new DBMS and stays online as `pinned` (the owner's plan of 2026-10-12). Without `--pinned`, 3.0 is reachable only through `serve-rollback`.
3. **When to swap.** Restoring 3.0 onto zhao5 has no downtime and can run before the build finishes; the swap is the only downtime.
4. **Cleanup on delta_bbub.** The 3.0 store there (151 G) is the container-level rollback copy until 4.0 is accepted. The July store (49 G), the staging copy (11 G) and the two dumps can go whenever the owner chooses.
5. **NCSA block volume (A0).** Capacity no longer needs it. It remains the fix if I/O faults return under load on NFS.
6. **Keep tags.** `kg_release.sh keep` both served releases so `purge` never removes their archives.

### Command sequence for KG 4.0

Run on GilaHyper from the primary checkout once this branch is on main, or from this worktree with `SNAPSHOT_ROOT=~/Documents/projects/torchcell` so the 4.0 snapshot the build writes is found. `R4` is the new release id (`make ops` shows it after the build), `R3=2026.10.06-4b293d34`. Use tmux: the restores run detached on Radiant either way.

```bash
cd ~/Documents/projects/torchcell
R3=2026.10.06-4b293d34
R4=<the 4.0 release id>

# 0. Any time before the build finishes: restore 3.0 onto zhao5 (no downtime, about 9 min)
bash scripts/kg_release.sh deploy $R3 --from-archive-mount --dry-run
bash scripts/kg_release.sh deploy $R3 --from-archive-mount

# 1. After the build swapped and stamped on GilaHyper: archive and ship 4.0 straight onto zhao5
bash scripts/kg_release.sh backup        # detached; wait for "exit 0" in /bulk/kg-releases/$R4/backup.log
bash scripts/kg_release.sh archive
bash scripts/kg_release.sh ship $R4
bash scripts/kg_release.sh keep $R4 "KG 4.0, served on Radiant"
bash scripts/kg_release.sh keep $R3 "KG 3.0, pinned on Radiant"

# 2. Restore 4.0 in place on zhao5 (no downtime)
bash scripts/kg_release.sh deploy $R4 --from-archive-mount --dry-run
bash scripts/kg_release.sh deploy $R4 --from-archive-mount

# 3. Swap Radiant onto zhao5 (staging registers both; downtime is one stop and one start)
bash scripts/kg_release.sh serve-swap --latest $R4 --pinned $R3 --dry-run
bash scripts/kg_release.sh serve-swap --latest $R4 --pinned $R3
#    prints the kept container: tc-neo4j-pre-<timestamp>

# 4. Verify
bash scripts/ops.sh sync                 # "in sync ($R4)": radiant [default] kg-<R4> with latest,torchcell; kg-<R3> with pinned
make ops                                 # full panel, artifacts line included
ssh rocky@141.142.216.218 docker inspect tc-neo4j --format '{{range .Mounts}}{{.Destination}} <- {{.Source}}{{println}}{{end}}'

# 5. Rollback, if needed
bash scripts/kg_release.sh serve-point --latest $R3 --pinned $R3     # stay on zhao5, serve 3.0 as latest
bash scripts/kg_release.sh serve-rollback tc-neo4j-pre-<timestamp>   # back to the delta_bbub container and store
```

Every step that changes something refuses on a failed precondition and prints what to do next. A failure partway through `serve-swap` before the swap leaves `tc-neo4j` serving untouched; a rerun resumes, because registration is idempotent.

## 2026.10.17 - KG 4.0 shipped and served on Radiant from the archive mount, measured

The 4.0 sequence of the 2026.10.13 section ran end to end on 2026-10-17, every step under slurm from the primary checkout (job ids in `/scratch/projects/torchcell/database/slurm/gen_job_7b56568db.txt`):

| step | measured |
|---|---|
| `backup` (online, detached in `tc-neo4j-readonly`) | 10:39 to 11:29 CDT, 50 min: 26 min receiving 252 G of store files, 24 min compressing; artifact `torchcell-2026-10-17T15-59-52.backup` 19.0 GB |
| `archive` + `ship 2026.10.17-4295f7ca` + `keep` x2 (job 5707) | 4 min 58 s, rsync to Taiga and SHA256SUMS verified there |
| `deploy 2026.10.17-4295f7ca --from-archive-mount --pin` (job 5709) | restore in place on zhao5 in 880 s (14 min 40 s), checked against `release.json` |
| `serve-swap --latest 2026.10.17-4295f7ca --pinned 2026.10.06-4b293d34` (job 5711) | stop to online 12:23:09 to 12:23:51, 42 s; kept `tc-neo4j-pre-20261017-122309` on delta_bbub |
| `ops.sh sync` | in sync (2026.10.17-4295f7ca): Radiant default `kg-2026-10-17-4295f7ca` with latest, torchcell; `kg-2026-10-06-4b293d34` with pinned |

The 3.0 restore onto zhao5 (step 0) had been done on 2026-10-13. The estimates of the 2026.10.13 section held: the restore ran at 0.3 GB/s of store and the swap downtime was under a minute. One thing the sequence note did not say: the `kg-ship` slurm job must run from a checkout that still exists when it starts (a job queued from a worktree that the merge queue then removed would have failed), so run it from the primary checkout after the snapshot lands, or set `SNAPSHOT_ROOT` to a tree that stays.

GilaHyper serves both versions too: `deploy 2026.10.06-4b293d34` (job 5713) restored 3.0 beside `torchcell` as `kg-2026-10-06-4b293d34` in 11 min and its check passed (51 datasets, 100,127,516 nodes), then the script's own alias step failed on syntax: Neo4j 5 wants `ALTER ALIAS ... SET DATABASE TARGET`, and the local deploy path had no `TARGET` (the archive-mount path did). The aliases were set by hand (`ALTER ALIAS pinned SET DATABASE TARGET \`kg-2026-10-06-4b293d34\``; `latest` stayed on `torchcell`, which is what was wanted), the script is fixed, and `ops.sh sync` reads both hosts in sync with 4.0 as latest and 3.0 as pinned. Note that the local deploy always points `latest` at what it restored, so a restore meant only as `pinned` needs `latest` re-pointed right after.
