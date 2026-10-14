---
id: fu9iahxydw41bvxjds5lzw0
title: bacteria-datasets-table
desc: ''
updated: 1791956523299
created: 1791956523299
---

## 2026.10.14

- [x] `feat/bacteria-datasets-table` worktree: temporary bacteria Instances vs Signal table with the Signal controls of #1099 measured in the same pass (structure only, shuffled, digit noise, 1-bit in place, 1-bit packed), 100 datasets, all from the adapter map ([[experiments.database.scripts.build_bacteria_datasets_table]], generated view [[experiments.database.bacteria-datasets-table]]). Measured: shuffled 1.02 and digit noise 1.07 of Signal, structure 0.29, packed 1-bit 0.009.
- [ ] Yeast run of the same probes (`--organisms scerevisiae`, 377 GB) in flight; compare when it lands. Issues #1098 and #1099 stay open for the post-KG-4.0 table build.
