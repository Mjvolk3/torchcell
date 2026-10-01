---
id: ombvru1jryxzxovx9ju5l46
title: fix-verifier-filelock-bibstore
desc: ''
updated: 1790872943352
created: 1790872943352
---

## 2026.10.01

- [x] PR: FIX(verification,utils,literature) issue #529 bullets 1-3: eager and streaming environment-response verifiers now produce equal reports (redundant-record duplicate count, record-indexed bad values with reasons, enum values); `FileLockHelper` keeps `timeout=0`, forwards `retry_delay` as `poll_interval`, stages through `<name>.tmp`; the bib store addresses collections by declared key, cleans its `.part` files on failure, moves undeclared `.bib` files to `_bib/_retired/`, and reads inline Makefile comments. Notes: [[torchcell.verification.environment_response]], [[torchcell.utils.file_lock]], [[torchcell.literature.bib_store]], [[torchcell.literature.bib]]; tests in [[tests.torchcell.verification.test_environment_response]], [[tests.torchcell.utils.test_file_lock]], [[tests.torchcell.literature.test_bib_store]], [[tests.torchcell.literature.test_bib_more]].
