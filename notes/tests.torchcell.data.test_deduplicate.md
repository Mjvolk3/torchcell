---
id: 5jf2ppmjpuh47ssvuwfewxi
title: Test_deduplicate
desc: ''
updated: 1790534890254
created: 1790534890254
---

## 2026.09.27 - The Deduplicator LMDB stage on a four-record raw store

`tests/torchcell/data/test_mean_experiment_deduplicate.py` pins the merge arithmetic on in-memory records; this file pins the LMDB stage around it, driven through `MeanExperimentDeduplicator`. The raw store under `tmp_path` holds four records under the keys `Neo4jQueryRaw.process` writes (`data_0` to `data_3`): two fitness records on YAL001C (fitness 0.75 with std 1.0, fitness 0.25 with std 0.0), a fitness singleton on YAL002W, and a gene-interaction record on the pair. The duplicate key is `sha256("<experiment_type>:<sorted gene names>")`, so the two YAL001C fitness records form one group and the interaction record does not. `process` writes groups in first-occurrence order under `0`, `1`, `2`: the merged pair (fitness 0.5, RMS-pooled std sqrt(0.5), a `MeanDeletionPerturbation` with `num_duplicates` 2, dataset name `a+c`, reference fitness 1.0 with no std), then the two singletons passed through byte for byte. Readers: `__getitem__` on an int, a slice and a list, the index errors, `__len__`, `__bool__`, `__repr__`; `_init_lmdb` read-only versus writable; `close_lmdb`. Finding: `__repr__` is the literal base-class name `Deduplicator` for every subclass (deduplicate.py line 237). Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
