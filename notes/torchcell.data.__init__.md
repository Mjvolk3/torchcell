---
id: o7wyt2ee7asyzd4brez6r04
title: __init__
desc: ''
updated: 1790814219519
created: 1790814219519
---

## 2026.09.30 - Raw sha256 helpers exported

`torchcell.data` now re-exports the raw-file pin helpers from [[torchcell.data.experiment_dataset]]: `RawSha256MismatchError`, `file_sha256`, `verify_sha256`, `verify_raw_files`, `copy_verified`, `write_verified`, `link_verified`. Every pinned loader imports them from here (issues #518, #524, #528, #537).
