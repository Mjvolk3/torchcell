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
