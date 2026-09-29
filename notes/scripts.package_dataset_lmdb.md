---
id: 6m41g0i336jjn7pkmfcpg7h
title: Package_dataset_lmdb
desc: ''
updated: 1790716968911
created: 1790716968911
---

## 2026.09.29 - Packaging a built LMDB into the tc-data store

`scripts/package_dataset_lmdb.py --dataset-dir $DATA_ROOT/data/torchcell/<slug> --store $TC_DATA_ROOT [--kg-release R --kg-version V --status supported|deprecated --no-content-hash]` tars `preprocess/` and `processed/` (never `raw/`) into `<store>/<slug>/<slug>-<version>-<sha8>.tar.xz`, upserts the `DatasetArtifact` row in `<store>/index.json` and rewrites `<store>/SHA256SUMS` (the `kg_release.sh` precedent). `package_dataset()` is importable for tests and other scripts.

Determinism: members in sorted order, `mtime` fixed to the manifest's `built_at`, uid/gid 0, empty user names, mode 0644, GNU format, LMDB `lock.mdb` excluded (a runtime lock file that changes with every reader), xz preset 6. Packaging the same build twice yields the same bytes, the same sha256 and the same file name, so the index row is replaced rather than duplicated.

Refusals (`PackagingRefused`, exit 1, none overridable): no `preprocess/build_manifest.json`; no `processed/lmdb`; a manifest `dataset_name` that differs from the directory name; a manifest that is stale against the local schema surface (`check_manifest` drift), because the artifact would claim the packager's `torchcell.__version__` for an LMDB the local schema would serialize differently. Rebuild with `torchcell.database.build_dataset_lmdb` first.

`content_sha256` is computed from the records without re-validating them: each record stores `experiment.model_dump()` verbatim with constant sub-objects interned by `$ref`, so hashing `json.dumps` of the resolved dict reproduces `CellAdapter._experiment_node`'s id. `--no-content-hash` skips the pass and stores null (for a store whose only purpose is a smoke test).

Measured on the dev `smf_costanzo2016` build (20,484 records, `data.mdb` 76,894,208 bytes, packaged into the session scratch store, not `$DATA_ROOT`): archive 1,053,744 bytes (`xz -l`: 75.1 MiB uncompressed, ratio 0.013; the pickled records are highly repetitive), 5.1 s inside the packager and 11.6 s wall including the torchcell imports (`/usr/bin/time`: user 22.8 s, 206% CPU, 1.27 GB RSS). Checks on that archive: all eight members byte-identical to the source files, the unpacked LMDB reports 20,484 entries, `sha256sum -c SHA256SUMS` passes, and the raw-dict experiment ids equal the `FitnessExperiment(**d).model_dump()` ids on 20,484/20,484 records, giving `content_sha256` `634eaf5e...ec9f` both ways. The dev build carries `torchcell_commit f77f6c15` and no KG release yet; the row records both as they are.
