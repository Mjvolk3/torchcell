---
id: 36etta19ka5pz0m18o4784v
title: notes-tex-groups
desc: ''
updated: 1791353563448
created: 1791353563448
---

## 2026.10.07

- [x] Added a group layer to `notes-tex/` (`notes-tex/<group>/<slug>/`, six groups by research program) because the flat tree and its Zotero index had outgrown themselves at thirty-two collections; the Zotero path follows the directory, so the layer costs no configuration [[notes-tex.common.zotero_publish#20261007---the-group-layer-notes-texgroupslug-and-the-zotero-path-follows]]
- [x] Wrote the one-shot `zotero_regroup.py` and ran it: the twenty-nine existing collections now sit under their groups with rewritten Doc Keys, so no version history was split; the publisher now refuses a flat path [[notes-tex.common.zotero_regroup]]
- [ ] Worktree-only documents (019-*, 025-s3-closure, 026-*, 027-*, 028-*, 031-*, 032-*, 034-isobutanol-wetlab, figure-3-gate, perturbation-operator, metabolic-module-report, metabolism-figure-gate, metabolism-figures, cgt-metabolism-io) must `git mv` under their group when their branch rebases; the kinetics branch's rendered-note publisher must file under `notes-tex/metabolism/` [[notes-tex.common.zotero_publish]]
