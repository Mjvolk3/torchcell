---
id: a29v30c7cy6keppb6o6maor
title: Test_neo4j_query_raw
desc: ''
updated: 1790549803874
created: 1790549803874
---

## 2026.09.27 - Neo4jQueryRaw without a database

Thirteen tests. The constructor does not connect when `raw/lmdb/data.mdb` already exists (neo4j_query_raw.py lines 159 to 162), so a hand-written store under `tmp_path` gives the LMDB read path (`_init_lmdb`, `__getitem__`, `__len__`, `close_lmdb`, `__repr__`), the exact serialized JSON bytes of an experiment plus reference, the processed-file checks, the attrs slotted state as an exact dict with a pickle round trip, and `process()` with `fetch_data` replaced on the class (the attrs class is slotted). Module coverage from this file 90%; `fetch_data` itself needs a driver and `parallel_hash_computation`'s body runs in a forked child that coverage cannot see. Findings: `__len__` returns inside the transaction, so the trailing `close_lmdb()` is dead and the environment stays open (line 355); `compute_phenotype_label_index` reads `phenotype.label` where the schema field is `label_name`, so `phenotype_label_index` raises `AttributeError` whenever it has to compute rather than read the file (line 422); `parallel_hash_computation` reads the key `"reference"` while the sequential branch reads `"experiment_reference"`, so `compute_experiment_reference_index(records, num_workers=1)` raises `KeyError('reference')` on the very records the sequential path accepts (lines 40 and 90). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].

## 2026.09.27 - Three findings fixed, their tests flipped

After the fix in [[torchcell.data.neo4j_query_raw]]: `len` closes the environment and a slice afterwards reopens it; `phenotype_label_index` computes `{"fitness": [0, 1, 2]}` and writes the JSON, and a file already on disk still wins; the sequential path, `num_workers=1` and the parallel helper return identical indices on the same records (`mp.cpu_count` patched to 1). The pickling test now checks the reopen through a record read, since `len` no longer leaves the store open.

## 2026.09.30 - Phase 17: fetch_data on a fake driver, hashing, the missing key

Twelve to nineteen tests, 94.5 to 100 percent. `fetch_data` on a fake driver with the real `resolve_database` alias pass-through (the version from the instance, else `TORCHCELL_KG_VERSION`; `fetch_size=1000`; `cypher_kwargs` reaching `run`; the driver closed once); a store built through the real `fetch_data`; `parallel_hash_computation` recomputed with hashlib; `_get_record` on a missing key refusing with `Record not found for key: data_9`; a cached reference index not rewritten after its JSON is deleted.

Findings: `driver.close()` is not in a `finally`, so a consumer that stops early leaves the driver open (line 194); a query that returns no records writes an empty store then fails in the gene-set setter, and the next construction would reuse that store and skip the query (494, 162).

## 2026.10.01 - Findings retired (issue #541)

Both Findings are retired. An early-closed generator now ends with `close`; an empty query raises `EmptyQueryResultError` with the exact message, leaves only the empty `raw/lmdb` directory, and a second construction re-runs the query (two full driver sequences) and stores two records.

## 2026.10.01 - Review: mid-query failure

A fetch that yields one record then raises leaves an empty `raw/lmdb` and no staging directory; a retry re-runs the query and stores all three records. A leftover `raw/lmdb.partial` is refused by name before any query.
