---
id: 348z9f7l6mnp9dk7qk4bk6a
title: Test_aggregate
desc: ''
updated: 1790534898826
created: 1790534898826
---

## 2026.09.27 - The Aggregator LMDB stage on a four-record raw store

`tests/torchcell/data/test_genotype_aggregate.py` pins `aggregate_check` on in-memory objects; this file pins the raw-JSON key, the stage and the readers, driven through `GenotypeAggregator` and `DeletionKeyedGenotypeAggregator`. Four records under `data_0` to `data_3`: fitness on YAL001C, fitness on YAL001C + YAL002W, gene interaction on the same pair, fitness on YAL001C again. The key is `sha256(str(sorted(set(gene names))))` and ignores the experiment type, so records 0 and 3 form group `0` and records 1 and 2 form group `1`; the two digests are written out in the tests and were computed with `hashlib` in a shell. A group is stored as the JSON array of its members' stored bytes joined by a comma, asserted byte for byte. Also `phenotype_info` and `_get_phenotype_info`, `create_aggregate_entry`, `ExperimentInfo`, and the environment lifecycle. Findings: `agg[[i, j]]` flattens the groups into one record list while `agg[i:j]` returns a list of groups (aggregate.py line 185); `process` resets `_experiment_info`, which nothing reads, instead of `_phenotype_info`, so `phenotype_info` stays stale after re-processing a store with a new phenotype family (line 166); `__repr__` is the literal base-class name (line 253). Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
