---
id: 39e36fwxhgy8rp9i2na0okv
title: Sync
desc: ''
updated: 1791364529735
created: 1791364529735
---

## 2026.10.07 - The bacterial collections join the nightly defaults

`DEFAULT_COLLECTIONS` gains `Escherichia-coli` and `Pseudomonas-putida`. Both sit under `database/` in the group library, and `_collection_items` returns only a collection's direct members, so the `database` pass never reached their 50 papers. Their publisher supplementary files are fetched separately by [[torchcell.literature.capture_si]], since none of the 50 Zotero items carries an SI attachment.

## 2026.10.11 - The yeast collection joins the nightly defaults

`DEFAULT_COLLECTIONS` gains `SCEREVISIAE_COLLECTION = "Saccharomyces-cerevisiae"` (group key `KCG3NWH7`, a direct child of `database/`, resolved read-only by `ZoteroLibrary.collection_key` on 2026-10-11 to the one collection of that name). It held 73 top-level items on 2026-10-11; a paper filed only there was invisible to the nightly `database` pass for the same reason the bacterial papers were. `capture_si.py` keeps no default collection list (its `--collection` is always explicit), so nothing there changes. Audit of the collection: [[database.yeast-zotero-collection-audit.2026.10.11]].
