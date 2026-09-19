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
