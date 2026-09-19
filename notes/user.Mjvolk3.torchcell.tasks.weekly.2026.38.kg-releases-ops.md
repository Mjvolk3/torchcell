---
id: aflc334cieph067sjarcfy3
title: kg-releases-ops
desc: ''
updated: 1789840629905
created: 1789840629905
---

## 2026.09.19

- [x] Knowledge-graph releases: `KgRelease` node + manifest version/release/content hashes, `TORCHCELL_KG_VERSION` (default `latest`) resolved through database aliases, `releases.datasets()/diff()/compat()`, `make ops` panel, `kg_release.sh` online backups on /bulk shipped to Taiga [[plan.kg-releases.2026.09.19]] [[torchcell.knowledge_graphs.releases]] [[scripts.ops]] [[scripts.kg_release]]
- [ ] Radiant VM: request a ~2 T block volume from NCSA; the NFS-backed store faults on every read (system database included) and Neo4j does not support NFS [[plan.kg-releases.2026.09.19]]
- [ ] Delete `/bulk/deprecated/2026.09.19/kg-releases-partial-backup-workdir` (373 G, the killed first backup) and `/db/deprecated/2026.09.19/kg-releases` if still present
