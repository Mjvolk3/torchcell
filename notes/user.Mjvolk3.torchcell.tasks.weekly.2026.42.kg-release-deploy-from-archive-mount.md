---
id: r9pjsse6lwks00p8dfg6fph
title: kg-release-deploy-from-archive-mount
desc: ''
updated: 1791870269384
created: 1791870269384
---

## 2026.10.13

- [x] `feat/kg-release-deploy-from-archive-mount` worktree: Radiant can serve KG 4.0 from the Taiga zhao5 mount the release archive sits on. `kg_release.sh deploy --from-archive-mount` restores in place, and `serve-swap`, `serve-point` and `serve-rollback` handle the move and its undo. The restore check is now per dataset ([[scripts.kg_release]], [[torchcell.knowledge_graphs.releases]]). Measured: `/mnt/delta_bbub` is a Taiga NFS export, not a block volume, so 3.0 already serves from NFS.
- [ ] Owner: restore 3.0 onto zhao5 before the build ends, then run the 4.0 sequence at the end of [[scripts.kg_release]]; decide the delta_bbub cleanup once 4.0 is accepted
