---
id: 39e36fwxhgy8rp9i2na0okv
title: Sync
desc: ''
updated: 1791364529735
created: 1791364529735
---

## 2026.10.07 - The bacterial collections join the nightly defaults

`DEFAULT_COLLECTIONS` gains `Escherichia-coli` and `Pseudomonas-putida`. Both sit under `database/` in the group library, and `_collection_items` returns only a collection's direct members, so the `database` pass never reached their 50 papers. Their publisher supplementary files are fetched separately by [[torchcell.literature.capture_si]], since none of the 50 Zotero items carries an SI attachment.
