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

## 2026.10.06 - `pair_package_tag`: pairing a snapshot after the build

`pair_package_tag(snapshot, closures, tag, surface)` is the repair for a release built from an untagged commit (`torchcell_tag` None). It accepts only a `vX.Y.Z` tag and only when `surface`, the schema surface at that tag (`kg_manifest.surface_at_ref`), reproduces every served closure (`ReleaseCompatibility.paired`: none drifted, none unverified); it then sets `torchcell_version` from the tag and `torchcell_tag` to it and appends to the last event's note what the build checkout had reported. The same tag again is a no-op (byte-stable snapshot), a different tag on an already paired snapshot is refused: a release has one paired package. `releases retag` is the CLI; `scripts/kg_release.sh retag` chains it with the manifest and the store's node.

Applied 2026-10-06 to the three committed snapshots: `2026.09.21-ab6d8c5d` paired with `v1.2.1` (tag of 2026-09-27, the first that reads it; built from 1.2.0 untagged), `2026.10.02-833970cd` with `v1.6.1` (tag of 2026-10-02 02:41 UTC, before the 10:55 UTC build; built from 1.6.1 untagged), `2026.10.06-4b293d34` with `v1.6.2` (the `DB(kg)` release cut after the build, PR #679).

## 2026.10.08 - The snapshot carries the pointer set

`KgReleaseSnapshot.artifact_refs: dict[str, list[ArtifactRef]] | None`, filled by `snapshot_from_manifest` through `kg_manifest.manifest_artifact_refs` (None when any entry is unrecorded). The committed snapshots predate it and load with None; a snapshot written from now on carries the key (as `null` until the build records pointers), so its first key is `artifact_refs`. Test: `test_snapshot_carries_the_pointer_set_through_write_and_load`.
