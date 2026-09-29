---
id: jt9z5d56cvp4ejiwig75bu2
title: Artifact
desc: ''
updated: 1790716936654
created: 1790716936654
---

## 2026.09.29 - DatasetArtifact and ArtifactIndex

`torchcell/datasets/artifact.py` is the record layer for the `tc-data` download API (Decision 6 of [[plan.data-release-program.2026.09.29]], issue #469). A `DatasetArtifact` is one packaged archive of a built dataset directory: `slug`, `dataset_class`, `torchcell_version` and `torchcell_commit` of the build, `kg_release` and `kg_version` it was admitted to (null until an admission names them), `content_sha256`, `n_experiments`, `archive`, `archive_sha256`, `archive_bytes`, `built_at`, `packaged_at`, `status` (`supported | deprecated`) and the verbatim `BuildManifest`. An `ArtifactIndex` (`schema_version` 1, `generated_at`, `artifacts`) is the store's `index.json`.

Decisions:

- The archive name embeds the slug, the package version and the first eight hex digits of the archive sha256 (`smf_costanzo2016-1.2.1-d8a0f06c.tar.xz`), so two packagings with different bytes never share a name, and `upsert` replaces a row only when both slug and archive name match.
- `content_sha256` is the same digest as `torchcell.knowledge_graphs.releases.content_sha256` (sorted, newline-joined experiment ids), so an artifact row is comparable with a KG release snapshot. The ids are the adapter's `sha256(json.dumps(experiment.model_dump()))`; the packager computes them from the stored record dicts, which was checked equal to the re-validated path on every one of the 20,484 `smf_costanzo2016` records (see [[scripts.package_dataset_lmdb]]).
- `select(slug, version)` returns the newest `supported` row whose `torchcell_version` shares `major.minor` with `version` (highest version, then latest `packaged_at`), or None. A `deprecated` row stays in the index for provenance and is never selected.
- `save` writes the rows sorted by (slug, version tuple, archive sha256) with a two-space indent and a trailing newline, so `save(load(x))` reproduces `x` byte for byte; `sha256sums()` emits `sha256sum -c` lines relative to the store root.
- The archive suffix is `.tar.xz`, not `.tar.zst`: `zstandard` is not importable in the torchcell environment, and stdlib `tarfile` + `lzma` needs no dependency in the slim server image. `ARCHIVE_SUFFIX` is the one place to change if zstd is added.
