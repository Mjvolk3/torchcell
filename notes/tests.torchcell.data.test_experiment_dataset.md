---
id: dgu7ncrpk9c0wb0wowsc9o6
title: Test_experiment_dataset
desc: ''
updated: 1790769084408
created: 1790769084408
---

## 2026.09.30 - Phase 15: the base class on a toy loader, the tc-data path with a fake client

New file, sixteen tests; with the two sibling files the module goes from 49 to 97 percent (the misses are the two abstract bodies and three minor branches). A toy loader writes three records through the real interning writer. Pinned: build order (publisher download, then `process`, once; a second build runs neither), the `$ref` pointers for environment and references with the 66-byte publication inline, `get` on an index list or mask, the sorted gene set and the reference index A to [0, 1], B to [2], the build manifest fields, a stale reference index failing the coverage check with its exact message, `io_workers=2` matching the sequential build, `gene_set.json` winning over the LMDB and an empty set refused, `transform_item` round trips, and the tc-data path with a fake client that never opens a socket: a compatible artifact is unpacked in place of download and `process`; no compatible artifact raises the full message naming the slug and version with no publisher fallback; an archive without a build manifest raises the exact `FileNotFoundError`.

Findings: `transform_item` builds the reference twice and discards the first (lines 639-640); `serialize_for_hashing` sorts only the top-level keys of a reference model, so a model and its dump hash differently, latent because the index always hashes the dict (67-72).

## 2026.09.30 - Raw sha256 pin enforced at build time

Issues #518, #524, #528, #537. Five tests for the shared raw-pin helpers: `file_sha256` (chunk size and symlink), `verify_sha256` / `verify_raw_files` (exact message and attributes, first mismatch in mapping order), `write_verified` (nothing written on a refusal), `copy_verified` (source hashed first, destination untouched on a refusal), `link_verified` (no link on a refusal, an existing link kept). Digests use the FIPS `abc` vector.

## 2026.09.30 - Findings Retired (Issue #532)

Both findings are retired. `transform_item` constructs the reference once (count 1). A reference model and its dump serialize to the same string and hash; the fixture's three dict reference ids equal the literal pre-fix hashes, pinning that the index path is unchanged.
