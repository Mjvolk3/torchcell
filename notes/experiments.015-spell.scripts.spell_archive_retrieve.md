---
id: jeobvg1rmlbwuht1m7vpqq7
title: Spell_archive_retrieve
desc: ''
updated: 1791152040528
created: 1791152040529
---

## 2026.10.04 - Retrieval of SGD's expression download area, with a provenance manifest

SGD serves downloads from the public S3 bucket `sgd-archive.yeastgenome.org` (us-west-2). `downloads.yeastgenome.org/expression/microarray/` is a JavaScript browser over that bucket, so the listing comes from the S3 list API (`?list-type=2&prefix=expression/microarray/`), not from the page.

The script lists the area, fetches every file that is not inside a per-study folder, and writes `$DATA_ROOT/data/sgd/spell/sgd_download/manifest.json` with one pydantic record per file: `source_url`, `retrieval_method`, `retrieval_command`, `retrieved_at`, `sha256`, bytes, and the upstream `Last-Modified` and ETag.

Run of 2026.10.04: 2,050 upstream objects; 603 study folders (same names as the local extraction, not fetched because they repeat `all_spell_datasets.tar.gz`); 56 files recorded, 1,445,979,856 bytes.

- `all_spell_datasets.tar.gz` (301,411,734 bytes, sha256 `d021ba29...`, upstream dated 2024-04-25) was downloaded by hand on 2025-12-21 with no record. It is entered as `manual_browser` with `retrieved_at` from the file's birth time and its size checked against upstream. The bytes were not re-fetched.
- New: `dual_channel_arrays.tar.gz`, `single_channel_arrays.tar.gz`, `all_spell_readmes.tar.gz`, `all_readmes.tar.gz`, `all_spell_exprconn_pcl.tar.gz` (2019), `expression/README.html`, and `archive/` (dated archives of 2015-02-17 and 2017-05-12, plus 45 Expression Connection files).
- `archive/all_spell_datasets_20170512.tar.gz` and `archive/all_spell_exprconn_pcl.20170512.tar.gz` have the same sha256 (`f8bccf98...`).
- SGD's README says only that the directory holds the files that populate SPELL. It does not describe the processing.

Not yet done: the new tarballs have not been unpacked or compared with the current archive.
