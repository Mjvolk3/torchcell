---
id: jtdzfmcl27jlbfbyfkhpmts
title: Test_neo4j_cell_single_pass
desc: ''
updated: 1790899843379
created: 1790899843379
---

## 2026.10.01 - Raw handle released and store reopened in process

The byte-identity test failed under lmdb 2.3.0 (CI run 36941506091) with `The environment '.../old/raw/lmdb' is already open in this process.`: the source left the raw view's handle open while the aggregation opened the same path (lmdb 1.7.5 locally allows that, so it passed here). Fixed in `Neo4jCellDataset.process`. New test `test_build_releases_the_raw_handle_and_the_store_reopens_in_process` asserts the raw view's env is `None` when the aggregation starts, then constructs the raw store again in the same process and checks all 40 records read back equal to the queried experiments and references.
