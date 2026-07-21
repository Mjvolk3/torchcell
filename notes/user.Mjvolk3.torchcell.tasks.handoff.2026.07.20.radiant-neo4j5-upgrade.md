---
id: n9uohfwvwvmtsdthjzmz2wx
title: radiant-neo4j5-upgrade
desc: ''
updated: 1784597565131
created: 1784597565131
---

# Radiant handoff — upgrade Neo4j 4.4.30 → 5.26.28, ready to load the torchcell dump

## 2026.07.20 - Handoff for the Radiant-side Claude Code session

**You are Claude Code running ON the Radiant VM** (`torchcell-database.ncsa.illinois.edu`,
OpenStack `.novalocal`, user `rocky`, docker-only, **no SLURM**). GilaHyper is a **separate**
lab machine where the KG build runs.

### Why (the invariant — do not skip)

`neo4j-admin database dump`/`load` only interoperate **within the same Neo4j major/store
version**. GilaHyper's build produces a **5.26.28** store. Radiant currently serves **4.4.30**.
So a 4.4 Radiant **cannot** load the 5.26.28 dump. Your job: bring Radiant to **5.26.28** and
stage it so tomorrow's `dump → transfer → load` just works. The command syntax below is 5.x
(it changed from 4.4: `neo4j-admin database dump/load` subcommands, `--to-path`/`--from-path`).

### Division of labor

- **GilaHyper (other session):** finishes build 977 (5.26.28) → verifies node/edge counts →
  `neo4j-admin database dump torchcell` → provides the **image** + the **`torchcell.dump`**.
- **Radiant (you), TODAY:** get the 5.26.28 image, migrate the conf to 5.x, preserve TLS,
  back up + clear the old 4.4 store, stage for load.
- **Radiant (you), TOMORROW:** receive `torchcell.dump` → `neo4j-admin database load` → start
  the 5.26.28 read-only container → verify.

### First: discover local facts (GilaHyper session CANNOT see your box)

Before changing anything, capture and report:

1. How the current 4.4 container runs — `docker inspect tc-neo4j` (or compose): exact **volume
   mounts**, **ports** (expect 7473/7474/7687), `NEO4J_AUTH`, read-only env, and the **image**.
   The 5.26.28 container must mirror these.
2. Exact VM paths for the neo4j **data** mount, **conf** dir, and **TLS certificates**
   (`certificates/https/...`).
3. **What else lives under the data mount.** On GilaHyper the dataset LMDBs share that tree; on
   Radiant it's likely ONLY the neo4j store — confirm, because step 4 deletes store dirs.
4. Free disk (need ~2× the DB size for `torchcell.dump` + the loaded store).
5. GilaHyper→Radiant transfer path: ssh/scp reachability + any Duo/2FA constraint.

### Step 1 — Get the 5.26.28 image (12.7 GB, `michaelvolk/tc-neo4j:5.26.28`, id 4fb1a33b8426)

It is **NOT on Docker Hub** (only 4.4 `latest`/`0.0.x` exist there). Options:

- **Stream from GilaHyper** (preferred if ssh works):
  `docker save michaelvolk/tc-neo4j:5.26.28 | gzip | ssh rocky@<radiant> 'gunzip | docker load'`
- **Docker Hub:** GilaHyper `docker login && docker push michaelvolk/tc-neo4j:5.26.28`, then
  Radiant `docker pull michaelvolk/tc-neo4j:5.26.28`.
- **Verify:** `docker run --rm --entrypoint neo4j michaelvolk/tc-neo4j:5.26.28 --version` → `5.26.28`.

### Step 2 — Migrate Radiant's neo4j.conf to 5.x (it is still 4.4 format)

The repo's `database/conf/neo4j.conf` (Radiant conf) is **4.4 format** (`dbms.*`, 0 `server.*`).
5.26 needs 5.x keys (`dbms.*` → `server.*`/`db.*`). Migrate, preserving Radiant specifics — do
NOT copy GilaHyper's `gh_neo4j.conf` (that has `gilahyper.zapto.org`):

```
neo4j-admin server migrate-configuration --from-path=<confdir> --to-path=<confdir>
```

Then **hand-verify** these survived (migration can drop/rename keys):

- `server.bolt.advertised_address=torchcell-database.ncsa.illinois.edu:7687`
- `server.https.enabled=true`, `server.https.listen_address=:7473`,
  `server.https.advertised_address=torchcell-database.ncsa.illinois.edu:7473`
- SSL policy → `certificates/https` base directory (`dbms.ssl.policy.*` → `server.ssl.policy.*`).

⚠️ 5.26 silently ignores unknown (4.4) keys — swap only the image and the server boots but
**drops HTTPS/TLS/advertised-address**, coming up as plaintext localhost. The migration is
mandatory and must be hand-verified.

### Step 3 — Preserve TLS certificates

Keep `certificates/https/*` in place, mounted to the 5.26.28 container at the path the conf's
ssl policy base_directory points at. Losing these breaks HTTPS serving.

### Step 4 — Back up, then clear the old 4.4 store (5.26 will NOT boot on a 4.4 store)

Neo4j 5.26 refuses to start on a 4.4 (`AF4.3.0`) store and dies on the **system** DB —
`--overwrite-destination` cannot save it. Since we load a fresh dump, the old store is replaced:

1. **BACK UP first** (the DB is live): stop the 4.4 container; `tar czf torchcell-4.4-store-backup.tgz`
   the current store dirs.
2. Remove ONLY the neo4j store dirs under the data mount: `data/databases`, `data/transactions`,
   `data/dbms`, `data/server_id` (root/7474-owned → via a root container:
   `docker run --rm --user 0:0 -v <datamount>:/data alpine rm -rf /data/databases ...`).
   **Delete nothing else** under the data mount.

### Step 5 — Stage for tomorrow

Have ready: the 5.26.28 image loaded, conf migrated + verified, certs in place, a directory to
receive `torchcell.dump`, and the mirrored `docker run` (from step-1 inspect) with the 5.26.28
image + read-only env.

### Tomorrow — load + serve

```
# after torchcell.dump arrives, with the DBMS stopped or torchcell DB absent:
neo4j-admin database load torchcell --from-path=<dumpdir> --overwrite-destination
# start the read-only 5.26.28 container (mirrors the old run cmd), then:
#   CREATE DATABASE torchcell   (if not auto-present)
#   MATCH (n) RETURN count(n)   (verify against the count GilaHyper reports)
```

### Rollback

Keep `michaelvolk/tc-neo4j:4.4.30-rollback` + the step-4 store backup. If the 5.x load fails,
restore the backup and run the 4.4 image to resume serving the Dec-2025 DB.

### Verified 5.26 syntax (from the image)

- dump: `neo4j-admin database dump <db> --to-path=<dir>` → `<db>.dump`
- load: `neo4j-admin database load <db> --from-path=<dir> --overwrite-destination`
- conf migrate: `neo4j-admin server migrate-configuration --from-path=<dir> --to-path=<dir>`
