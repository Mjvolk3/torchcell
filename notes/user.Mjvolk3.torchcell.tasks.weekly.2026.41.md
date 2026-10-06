---
id: byzr4ppu8xdozf9fox128ed
title: '41'
desc: ''
updated: 1791256185172
created: 1791256185172
---

## 2026.10.05

- [x] `w037-isobutanol-scrnaseq` worktree: picked six of the twelve isobutanol strains on hand for the first single-cell RNA-seq run, as three pairs that each differ by one plasmid [[user.Mjvolk3.torchcell.tasks.weekly.2026.41.w037-mirror-cite]]
- [ ] The table note and the SI caption still say "pre-build ... not yet a versioned Neo4j DB build"; all 51 have been served as release 1.0 since 2026.09.17, so that framing needs the author's call [[torchcell.knowledge_graphs.releases]]
- [ ] Radiant VM: request a ~2 T block volume from NCSA; the NFS-backed store faults on every read (system database included) and Neo4j does not support NFS [[plan.kg-releases.2026.09.19]]
- [ ] Delete `/bulk/deprecated/2026.09.19/kg-releases-partial-backup-workdir` (373 G, the killed first backup) and `/db/deprecated/2026.09.19/kg-releases` if still present
