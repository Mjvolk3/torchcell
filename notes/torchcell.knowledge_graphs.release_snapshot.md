---
id: 0j9r0u87h5skfnbblw9bkuy
title: Release_snapshot
desc: ''
updated: 1790716446380
created: 1790716446380
---

## 2026.09.29 - Committed release snapshots

Plan: [[plan.data-release-program.2026.09.29]], Decisions 1 and 2 (T4, issue #467). The
build manifest lives only on GilaHyper (`/scratch/projects/torchcell/database/kg_manifest.json`),
so CI cannot compute which package versions read a release. The snapshot is the committed
subset that can.

- `KgReleaseSnapshot` (pydantic): `release`, `version`, `torchcell_commit` (the full
  build's, as `KgRelease`), `torchcell_version`, `torchcell_tag`, `built_at`,
  `neo4j_version`, `biocypher_version`, `store_host`, `n_nodes`, `datasets`
  (`SnapshotDataset`: class, `n_experiments`, `content_sha256`, `import_mode`,
  `admitted_at`, sorted by name), `graph_schema` (the manifest's `GraphSchemaEntry` map),
  `events` (`SnapshotEvent`: kind, at, commit, datasets, note) and `composite_sha256`,
  the sha256 over the sorted `content_sha256` values newline-joined with a trailing
  newline (`releases.content_sha256` applied to hashes instead of ids).
- Files: `database/releases/<release>.json` and `<release>.closures.json` (the per-dataset
  closure fingerprints, about 155 KB for 51 datasets). Both are written with sorted keys,
  `indent=2` and one trailing newline; a second write of the same input is byte-identical
  (tested).
- `snapshot_from_manifest(manifest, n_nodes=None, built_at=None)` refuses an unstamped
  manifest and a dataset without a hash or a record count; `built_at` defaults to the
  manifest's last event time and the slurm scripts pass the stamp's own timestamp.
  `bootstrap_package_version` fills a pre-spine manifest's version and tag and says so in
  the last event note; it is refused when the manifest already records a version.
- `load_snapshots(repo_root)` sorts by `built_at`; `load_closures(repo_root, release)` is
  the companion. Consumer: [[scripts.kg_compat_page]]. CLI: `python -m
  torchcell.knowledge_graphs.releases snapshot` ([[torchcell.knowledge_graphs.releases]]).
- First snapshot: `2026.09.21-ab6d8c5d` (torchcell 1.2.0, untagged, 99,724,909 nodes,
  composite `39e26472ceb731c22f882e80fbd9c10a941637e57823317795283fcfbf36d2dc`).
