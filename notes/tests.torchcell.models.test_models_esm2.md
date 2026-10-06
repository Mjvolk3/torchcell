---
id: kx2ex6lsrr304ogf4x876hg
title: Test_models_esm2
desc: ''
updated: 1791269239867
created: 1791269239867
---

## 2026.10.06 - Phase 21 tests

- Fixture: `AutoTokenizer` / `AutoModelForMaskedLM` replaced by recorders, module `__file__` in `tmp_path`, CUDA reported absent, hub offline.
- Fake tokenizer: `[0] + [5]*len + [2]`, pad 1. Fake model: `hidden_states[-1][b, t] = [t, 10b]`.
- Masked mean for `["AC", "ACGT"]`: 6/4 = 1.5 and 15/6 = 2.5, output `[[[1.5, 0], [2.5, 10]]]` (leading axis kept; `Esm2Dataset` embeds one sequence per call and flattens it with `.reshape(1, -1)`, datasets/esm2.py:166).
- Finding (esm2.py:46-47, 63-64): the "already downloaded" check tests `<cache>/facebook/<ckpt>`, but the Hub writes `models--facebook--<ckpt>`; the loads pass no `cache_dir`, so the fetched copy is never read.
- Finding (esm2.py:88): `max_length=1022` counts CLS and EOS, so the real `EsmTokenizer` keeps 1020 residues of a 1024-residue protein; ESM-2 accepts 1022.
- Reach (audit 1, not measured): the truncation finding reaches stored `Esm2Dataset` vectors for proteins longer than 1020 aa; esm2_t33 embeddings are named in smf-dmf-tmf-001 (72 configs), 019, 020-023 and 002/003. Effect size not measured. The cache-layout finding is latent (it costs a second download, not a different vector).

## 2026.10.06 - CI addendum

- CI installs transformers 5.18.0 (unpinned in `pyproject.toml`); this machine has 4.57.1. Transformers 5 removed `batch_encode_plus` from tokenizers.
- Source lines: `torchcell/models/esm2.py:83` (`self.tokenizer.batch_encode_plus(...)`) and `torchcell/models/nucleotide_transformer.py:78` (same call). On a fresh install neither wrapper can embed anything; the faked-tokenizer tests pass only because their fakes define `batch_encode_plus`.
- `test_real_esm_tokenizer_truncation_keeps_1020_residues` now proves the truncation with the real `EsmTokenizer.__call__` (ids `[0] + [5] * 1020 + [2]`, mask all ones, 1022 tokens), then branches on `hasattr(type(tokenizer), "batch_encode_plus")`: present, the wrapper sends exactly those ids and that mask to the model; absent, `embed` raises `AttributeError` with the full message `EsmTokenizer has no attribute batch_encode_plus` and the model is never called. Neither branch is skipped or xfailed.
- Both branches were executed here: the absent branch by a scratch runner that shadows `batch_encode_plus` on the `EsmTokenizer` class for lookups from outside transformers (coverage of the test file shows the if-body missed in that run and the else-body missed in the plain run).
- Pinned until both wrappers call the tokenizer directly.
