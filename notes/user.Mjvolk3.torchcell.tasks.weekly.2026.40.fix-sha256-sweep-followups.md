---
id: ttowowe3chjbam9fw5hkyz4
title: fix-sha256-sweep-followups
desc: ''
updated: 1790819181991
created: 1790819181991
---

## 2026.09.30

- [x] PR: `FIX(loaders)` follow-ups to the sha256-on-build sweep (issue #561). Every pinned loader is now checked to verify its full `{file: pin}` mapping before any read, in [[tests.torchcell.datasets.scerevisiae.test_raw_pins]]. `slow`/`data` tests meet the real pin, per [[tests.torchcell.conftest]]. The seven manifest-pinned downloads now verify against the module constant and refuse a disagreeing manifest by name, in [[torchcell.data.experiment_dataset]].
