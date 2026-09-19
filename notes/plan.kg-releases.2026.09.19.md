---
id: lji4m5lidahsdn23fv7jcsp
title: '19'
desc: ''
updated: 1789840113600
created: 1789840113600
---

## Context

The served knowledge graph moved hosts twice and is about to move again, to a Radiant
VM with Taiga behind it, and nothing so far names a served store: the manifest records
what was built, the Browser shows a database called `torchcell`, and a client hardcodes
both the host and that name. Three asks from 2026-09-19 fix that: code picks a version
(default latest), simple functions list what a version serves, and two versions can be
compared dataset by dataset for byte identity. Alongside, `make ops` reports every
service the way iBioFoundry's panel does, with date-plus-commit release ids.

## What landed (this branch)

| path | action | purpose |
|---|---|---|
| `torchcell/knowledge_graphs/releases.py` | NEW | release id, version bump, content hash, KgRelease node, resolver, list/datasets/diff/compat, CLI |
| `torchcell/knowledge_graphs/kg_manifest.py` | MODIFY | `version`, `release`, per-dataset `content_sha256` (optional fields) |
| `torchcell/database/connection.py` | MODIFY | `version` from `TORCHCELL_KG_VERSION`, default `latest` |
| `torchcell/data/neo4j_query_raw.py` | MODIFY | `version` attribute; database resolved at fetch time |
| `database/scripts/kg_content_hashes.sh` | NEW | per-dataset content hashes streamed through cypher-shell |
| `database/slurm/scripts/gilahyper_live_rebuild-slurm_docker.slurm` | MODIFY | hashes after validation; stamp, node, aliases after the swap; `/kg-releases` mount |
| `database/slurm/scripts/gilahyper_increment_kg-slurm_docker.slurm` | MODIFY | hashes of admitted datasets; minor bump; node rewrite |
| `scripts/ops.sh`, `Makefile` | NEW | `make ops`, `ops-health`, `ops-releases` |
| `scripts/kg_release.sh` | NEW | backup, archive, ship, keep, list, purge |
| `tests/torchcell/knowledge_graphs/test_releases.py` | NEW | 12 tests |

## Key design decisions

1. **Release id is `<build date>-<commit[:8]>`, version is `major.minor`.** The id is
   the immutable identity (what iBioFoundry calls BUILD); the version is the human
   counter: full rebuild bumps major, admission bumps minor. Both live in the store's
   `KgRelease` node and in the manifest.
2. **The store describes itself.** One node, scalar properties plus two JSON string
   properties (datasets, closures). A backup loaded on any host answers what it is.
3. **Byte identity is the sha256 of a dataset's sorted experiment ids.** Ids are
   already sha256 of the serialized record. Computed through cypher-shell on the build
   node, and from CSVs for a cross-check. Reference nodes are not part of it.
4. **Aliases, not renames.** `latest` and `pinned` are Neo4j database aliases. A client
   asks for a version; the resolver returns an alias or a physical database. On
   GilaHyper both aliases point at the one physical `torchcell`; a DBMS holding several
   releases (the VM) names physical databases by release and retargets the aliases,
   which is a seconds-long switchover with the old database still loaded.
5. **Compatibility is the admission gate turned around.** The node carries each
   dataset's closure fingerprints; `compatibility()` compares them with the local schema
   surface and names the datasets the local code would serialize differently. Clients
   are not blocked by default; `compat` in the CLI and `make ops` make it visible.
6. **Release artifacts are Enterprise online backups on /bulk.** No downtime. The
   serving container mounts `/bulk/kg-releases` at `/kg-releases`, outside NEO4J_HOME.
   Never stage on /db: the work directory holds the whole store before compressing.

## The move to the Radiant VM (blocked on a block volume)

Measured 2026-09-19: the VM has 16 CPUs, 62 G RAM, a 40 G root disk, no block volume,
and two Taiga NFS mounts; its current container serves the July store from NFS and
every store read, including the system database, raises `java.io.IOException`, with
"nfs: server not responding" in the kernel log. Neo4j does not support NFS for the
store. So:

1. Ask NCSA for a Radiant volume of about 2 T attached to the VM (one served store,
   one incoming, headroom). Until then GilaHyper serves and the VM's container should
   be stopped rather than answer errors.
2. Taiga holds releases only: `/mnt/zhao5/mjvolk3/projects/torchcell/kg-releases/<release>/`
   via `kg_release.sh ship` (85 MB/s measured, about 2.3 h for the store).
3. On the VM, `neo4j-admin database restore --from-path` into a physical database named
   by release, verify counts against `release.json`, `ALTER ALIAS latest SET DATABASE
   TARGET <db>`; `pinned` likewise, only while a paper needs it.
4. Retention: `kg_release.sh purge` on a 60-day window, `KEEP` files for tagged
   releases (a paper, an experiment series, every major boundary). Anything is
   rebuildable from source plus the archived CSVs.
5. Client side: collapse the dozen hardcoded hosts to `NEO4J_URI` before the hostname
   moves; the VM's TLS setup for `torchcell-database.ncsa.illinois.edu` is in place.

## Gotchas

- Importing `torchcell.knowledge_graphs` imports BioCypher: banner on stdout, a
  `biocypher-log/` directory in the cwd (gitignored). `ops.sh` filters the banner.
- `pkill -f "neo4j-admin database backup"` does not match the backup JVM (its command
  line is `... Neo4jAdminBoot database backup ...`); find `AdminTool` and kill by pid.
- The entrypoint chmods every top-level directory under NEO4J_HOME to 700 at start;
  mounts the host must read go outside it (`/kg-releases`, like `/logs`).
- The served database is read-only; the release node is written through
  `dbms.setConfigValue('server.databases.writable', ...)` for one statement, then reset.

## Verification

- `pytest tests/torchcell/knowledge_graphs/test_releases.py`: 12 passed.
- `make ops` on GilaHyper: table, verdict, health as recorded in [[scripts.ops]].
- Shell and Python content hashes agree: `CrisprMagicLian2019Dataset`, 266,304
  experiment ids, `8e2d5b91aa13...` from both `kg_content_hashes.sh` and
  `content_sha256` over a driver stream (2026-09-19).
