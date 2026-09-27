---
id: svqf6ykphhsygj6mxpxz4um
title: Test_conversion
desc: ''
updated: 1790534906643
created: 1790534906643
---

## 2026.09.27 - The Converter LMDB stage through the fitness converters

`Converter` only stores the `Neo4jQueryRaw` it is handed, so a sentinel `object()` stands in for it and no Neo4j is touched. `_compute_hash` on `{"b": 1, "a": [1, 2]}` hashes the sorted-key dump, digest written out in the test (`hashlib` in a shell). `ConversionEntry` and `ConversionMap` reject extra fields. `convert`: an essential-gene record becomes a `FitnessExperiment` with fitness 0.0 and no std, keeping dataset name, genotype and environment, its reference a `FitnessExperimentReference` with fitness 1.0; a record no entry matches passes through unchanged, extra keys included; the ValueError and TypeError paths. `process` writes under the SAME keys as the input, and the skip paths are asserted through exact `caplog.messages` and counts. `CompositeFitnessConverter` on both branches. Finding: a non-essential gene makes the conversion function return `None`, which `convert` refuses as a TypeError (conversion.py line 124); only the `except Exception` in `process` turns that into the documented "excluded from the converted dataset", so the exclusion happens through the error path. `__repr__` is the literal base-class name (line 314). Phase 6 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
