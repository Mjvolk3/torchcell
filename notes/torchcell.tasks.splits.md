---
id: pywr35ttpug2caerxlnoknu
title: Splits
desc: ''
updated: 1795561758216
created: 1795561758216
---

## 2026.11.24 - Split functions

Pure, seeded split functions that never divide a group: `assign_groups` (the dealer every split uses), `random_split`, `block_groups` (contiguous chromosome blocks), `pair_parts` and `pair_novelty` (gene-pair parts from gene parts). A group that crosses a boundary stays on the earlier side, so on a very small input a part can come out empty.
