---
id: a29v30c7cy6keppb6o6maor
title: Test_neo4j_query_raw
desc: ''
updated: 1790549803874
created: 1790549803874
---

## 2026.09.27 - Neo4jQueryRaw without a database

Thirteen tests. The constructor does not connect when `raw/lmdb/data.mdb` already exists (neo4j_query_raw.py lines 159 to 162), so a hand-written store under `tmp_path` gives the LMDB read path (`_init_lmdb`, `__getitem__`, `__len__`, `close_lmdb`, `__repr__`), the exact serialized JSON bytes of an experiment plus reference, the processed-file checks, the attrs slotted state as an exact dict with a pickle round trip, and `process()` with `fetch_data` replaced on the class (the attrs class is slotted). Module coverage from this file 90%; `fetch_data` itself needs a driver and `parallel_hash_computation`'s body runs in a forked child that coverage cannot see. Findings: `__len__` returns inside the transaction, so the trailing `close_lmdb()` is dead and the environment stays open (line 355); `compute_phenotype_label_index` reads `phenotype.label` where the schema field is `label_name`, so `phenotype_label_index` raises `AttributeError` whenever it has to compute rather than read the file (line 422); `parallel_hash_computation` reads the key `"reference"` while the sequential branch reads `"experiment_reference"`, so `compute_experiment_reference_index(records, num_workers=1)` raises `KeyError('reference')` on the very records the sequential path accepts (lines 40 and 90). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
