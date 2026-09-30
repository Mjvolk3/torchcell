---
id: 0dd0p29rdmdrgq64ya4qjqu
title: Test_nucleotide_transformer
desc: ''
updated: 1790780589051
created: 1790780589051
---

## 2026.09.30 - Phase 18: the wrapper on recorder fakes

New file, seven tests, 23 to 85 percent alone (`max_sequence_size` and `main()` stay). `AutoTokenizer` and `AutoModelForMaskedLM` are replaced by recorders returning fixed tensors, `__file__` is relocated into `tmp_path` for the cache check, and the offline variables are set. A cold cache downloads with `cache_dir` then loads in exact order; a warm cache skips the download and prints the exact notice; `embed` passes pad-length ids and the pad mask; per-token output is the detached last hidden state; mean pooling gives [[[1, 0], [2, 10]]]; a bare string becomes a batch of one.

Findings: `load_model(model_name)` ignores its argument (lines 55-65); the mean path returns `[1, batch, dim]` (102).
