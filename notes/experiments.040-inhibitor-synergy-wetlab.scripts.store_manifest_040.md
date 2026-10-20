---
id: 3ggwpi967ufoblcpezjumi2
title: Store_manifest_040
desc: ''
updated: 1792470022449
created: 1792470022449
---

## 2026.10.19 - The manifest and its verification

`StoreManifest` (pydantic) is what `build_040_store.py` writes beside the processed LMDB and
what `verify_store` reads back: the size and then the sha256 of `processed/lmdb/data.mdb`
must match, or the run stops before a record is read. The module imports nothing from
torchcell, so the build (torchcell at the release commit), the trainer (torchcell at the
experiment branch) and the launchers' preflight all load the same object. Why bytes and not a
file list: Delta's Lustre target 106 left about one file in a hundred unreadable with no error
on stat (2026-10-16), and a job that touched one hung to its 12 h limit.

`DatasetCount` carries four numbers per dataset and one table: `served` (the release's
count), `admitted` (what the block's clause admits, counted on the server with the same clause
text), `returned` (what the raw stage wrote), `dropped` (`served - returned`) and
`outside_gene_set` (served records per perturbed gene outside the gene set). The build asserts
`returned == admitted`, so `dropped` is the clause's exclusion and nothing else. In the full
store (5953) the three screens' `outside_gene_set` sums equal their drops exactly.
