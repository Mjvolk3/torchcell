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

## 2026.09.30 - Findings retired (issue #543)

Retired: the dead `model_name` argument and the `[1, batch, dim]` pooled shape. Now asserted: `load_model("some/other-model")` downloads that id into the cache and loads it, with the exact notice; the mean is `[[1, 0], [2, 10]]` of shape `[2, 2]` and a bare string gives `[[1.5, 0.0]]`.

## 2026.10.06 - CI addendum (real tokenizer path)

- Finding: `torchcell/models/nucleotide_transformer.py:78` calls `self.tokenizer.batch_encode_plus`, removed in transformers 5; CI runs 5.18.0 (unpinned), this machine 4.57.1, so on a fresh install the wrapper cannot embed.
- `test_real_tokenizer_path_depends_on_batch_encode_plus`: a real `EsmTokenizer` on a 9-token single-base vocabulary (unk 0, pad 1, mask 2, cls 3, eos 4, A 5, C 6, G 7, T 8) with `model_max_length = 6`; `__call__` pads `["AC", "ACGT"]` to `[[3, 5, 6, 4, 1, 1], [3, 5, 6, 7, 8, 4]]`. With `batch_encode_plus` on the class the wrapper sends those ids, the mask `ids != 1`, and returns the masked mean `[[6 / 4, 0], [15 / 6, 10]] = [[1.5, 0], [2.5, 10]]`; without it `embed` raises `AttributeError: EsmTokenizer has no attribute batch_encode_plus` and the model is not called.
- Pinned until the wrapper calls the tokenizer directly. See [[tests.torchcell.models.test_models_esm2]] for the shared CI addendum.
