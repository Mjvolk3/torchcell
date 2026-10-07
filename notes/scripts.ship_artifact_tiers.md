---
id: atf15dh3cydllboc9ngilpy
title: Ship_artifact_tiers
desc: ''
updated: 1791367131038
created: 1791367131038
---

## 2026.10.07 - Ship the file tiers to Taiga, verified by manifest

`bash scripts/ship_artifact_tiers.sh [--dry-run] [genomes|objects|raw ...]` copies the tiers tc-data serves from `$DATA_ROOT` on this host to `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/<tier>` on Taiga (through Radiant) and verifies them there. For each key directory it first writes `SHA256SUMS` from that key's `manifest.json`, so the check on arrival is against the manifest, the authority on what a key holds; the rsync is `-rlpt` (no owner or group, Taiga's setgid directories refuse chgrp; no `--delete`, the archive is a backstop); then `sha256sum -c` runs per key on Taiga and any failure fails the ship. A tier missing locally is skipped with a line, so the objects tier can be shipped before its first deposit. Phase 6 of [[plan.artifact-tier.2026.10.07]]: Radiant's tc-data reads the tiers from that mount, so a shipped tier is served at once and `torchcell.artifacts.resolve` reaches it over HTTP from any client.

`scripts/backup_mirrors_to_bulk.sh` now lists `torchcell-objects` beside the other three tiers for the Sunday copy to /bulk.

Measured 2026-10-07 (first ship, from GilaHyper): the genomes tier, 6 assembly sets, 56 files, 4,355,457,634 bytes, landed under `.../data/torchcell/torchcell-genomes` and every set passed `sha256sum -c` on Taiga (`ecoli_K12_BW25113_ASM75055v1`, `ecoli_K12_MG1655_ASM584v2`, `go_release_2026-08-05`, `peter2018_1011_assemblies`, `pputida_KT2440_ASM756v2`, `sgd_S288C_R64-4-1_20230830`); the run including verification finished at 06:04:43 CDT. The empty objects tier exposed a remote-shell detail: the far side's login shell is zsh, whose unmatched `*/` glob is an error, so the per-key loop now uses `find`. Radiant's tc-data still needs the two roots in its conf and a redeploy from the landed commit before `/genomes` answers there.
