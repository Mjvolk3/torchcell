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

## 2026.10.06 - Phase 21: abstract bodies, interned cache, splice and harvest

- `ExperimentDataset` refuses instantiation naming its seven abstract members; through the base class, `download` and `process` raise `NotImplementedError` and the other abstract bodies return `None`.
- A second dataset on one root re-attaches the same interned table and validated-instance dict (identity); `_load_interned` is idempotent; `get_single_item` reopens a closed env.
- `__getstate__` nulls `_interned` and `_experiment_reference_index` and empties `_validated_interned` in copies and pickles only; it keeps the LMDB handle, so pickling with the env open raises `TypeError: cannot pickle 'Environment' object`.
- `_splice_validated` is copy-on-write through dicts and lists; `_harvest_validated` caches list members the first time only; `_build` returns a cached constant without calling the model class.
- `check_manifest_pin` message and attributes.

Finding: `_INTERNED_BY_DIR` (experiment_dataset.py line 236, read at 505, filled at 525) is never invalidated, so deleting and rebuilding a store at the same root in one process with a new constant fails with `KeyError` on the new constant's digest. Audit 1 rates it latent and loud: keys are content hashes, so it can never return a wrong constant, and no production path deletes and rebuilds in one process (REPL or test only).
