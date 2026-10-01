---
id: xdnss3rnsr7aaxuwjq8x3zx
title: Test_notes_tex_zotero_paging
desc: ''
updated: 1790875427785
created: 1790875427786
---

## 2026.10.01 - Children past the first page (issue #563)

Imports `notes-tex/common/zotero_publish.py` and `zotero_comments.py` from their directory and drives them with the read-only paging `FakeZot` (no write methods). One document with 150 versions, attachments `A000` to `A149`, filenames ending in the 8-hex index. Asserts that `existing_hashes` returns all 150 sha8 entries after one `follow`, and that `zotero_comments.fetch` picks `A149` (`..._00000095.pdf`) as the latest with its one annotation flattened exactly, and `A100` for `--version 101`. Both tests fail on the previous first-page reads.
