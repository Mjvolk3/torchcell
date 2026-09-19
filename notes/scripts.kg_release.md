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
