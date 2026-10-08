---
id: yw9q0vqh6p42zzi2gvmn33b
title: Kg_manifest
desc: ''
updated: 1790204534064
created: 1790204534064
---

## 2026.09.21 - Admit a served dataset as a proven superset

The admission gate exists so a dataset can join the served graph without a full rebuild, but it blocked any dataset already served. That turned a loader fix that only adds records into a full rebuild: the Kuzmin 2018/2020 dmf query-strain fitnesses (+172 and +201, every existing record byte-identical, main `4c4a4f950`). A served dataset is now re-admissible when it provably only grows.

- The check reads the live store's experiment ids under the dataset's `Dataset` node over bolt through `ExperimentMemberOf` (no property index needed), then walks the dev LMDB through the adapter's own id path (`transform_item`, then the sha256 of the json-dumped `model_dump` that `_experiment_node` uses; a test pins the two).
- Admissible only if every served id is still produced and at least one new id exists. A served id the LMDB no longer produces blocks, because that is a changed record and the full-rebuild case; an identical re-admission blocks as nothing to add.
- Manifest: `KgDatasetEntry.superset_of` records the previous entry and `n_added`; the event kind is `superset_admission`. CLI: `admit --neo4j-uri --database`.
- Measured runs and the resulting served counts (410,571 and 632,998): [[torchcell.knowledge_graphs.incremental-admission]]. Slurm side: [[database.slurm.scripts.gilahyper_increment_kg-slurm_docker]].

## 2026.09.29 - The manifest records the package version

Plan: [[plan.data-release-program.2026.09.29]], Decision 1 (T4, issue #467). The schema
version stays 1: every new field is optional, so the served manifest (stamped 2026-09-21,
before this) still loads with `torchcell_version` None.

- `KgBuildManifest.torchcell_version` and `torchcell_tag`: the package version and tag the
  release names, written by `releases stamp` from the stamping checkout
  (`checkout_package_version`: `torchcell/__version__.py` of the checkout, `git describe
  --tags --exact-match HEAD`). `bootstrap` fills them from the build commit instead
  (`package_version_at_ref`, `package_tag_at_ref`: `git show <commit>:torchcell/__version__.py`
  and `git tag --points-at <commit>`; two tags on one commit raise).
- `KgEvent.torchcell_version`: the package version of the checkout that produced the
  event. `bootstrap` writes it from the commit; `admit` puts the working tree's version
  in the `AdmissionReport` (`torchcell_version`, next to `torchcell_commit`) and `record`
  copies it into the event, so a batch carries one value for all members.
- `manifest.torchcell_commit` stays the FULL build's commit while `release` names the
  latest event's commit; the snapshot's `torchcell_commit` follows `KgRelease` (the full
  build), and the compatibility page prints it as `commit`.
- Tests: `tests/torchcell/knowledge_graphs/test_kg_manifest.py` (round trip, the
  pre-spine manifest, the git helpers on a throwaway repo).

## 2026.10.07 - `drift`: the shared-surface checks with no dataset named

`admit` needs a dataset, and every mapped dataset is already served, so judging a change
to a surface all datasets share (a new graph class, a new `CellAdapter` method) meant
either a live-store superset proof or an ad hoc script. `drift` runs checks 2 to 4 of
`check_admission` alone and prints a `ServedSurfaceDrift`:

- `graph_schema_changed` / `graph_schema_added`, from `graph_schema_drift` (now also what
  `check_admission` calls for its step 2);
- `graph_schema_widened`: served edge classes that gained source or target labels, with
  the labels gained (additive under `served_nodes_unchanged_by`, named so a reviewer sees
  which served edge types new classes join);
- adapter drift touching served datasets and the ADDED methods (`adapter_drift_against`);
- the value surface.

Exit 1 when any served surface changed (`changes_served`), 0 when everything is additive;
`--report` writes the JSON. Per-dataset schema closures stay with
`torchcell.provenance.schema_impact`. First use, the bacterial graph classes measured as
ADDED-only: [[torchcell.adapters.bacterial-graph-classes]]. Tests:
`test_cli_drift_names_added_classes_widened_edges_and_new_methods`,
`test_an_edge_that_loses_a_source_is_changed_not_widened`
(`tests/torchcell/knowledge_graphs/test_kg_manifest_admission.py`).

## 2026.10.08 - `artifact-refs`: the release records the files its records point at

Records can carry `ArtifactRef` pointers to files kept off the graph (inside `Experiment.serialized_data` for Caudal's `sequence_ref` and Bloom's `assembly_ref`, flattened on `CrisprConstruct` as `effector_plasmid_ref` + `effector_plasmid_sha256`). The served store holds about 52 M experiments, so no reporter can scan it for them; the release records its pointer set instead.

- `KgDatasetEntry.artifact_refs: list[ArtifactRef] | None`. None means never recorded (every manifest written before this change, and an entry admitted since the last recording); `[]` means recorded, none. Entries are FILE-level: `member`, `bytes` and `media_type` are stripped (`file_level_refs`), deduplicated and sorted by `ref_key` `(tier, key, path, sha256)`, so the many members of one tarball are one entry.
- `record_artifact_refs(manifest, data_root, *, only=None) -> {dataset: n_refs}`. The gate is the closure: only an entry whose `closure` has the key `"ArtifactRef"` (`ARTIFACT_REF_SYMBOL`) is walked; every other entry gets `[]` without its dataset being opened. A walked dataset is instantiated read-only the way `dev_experiment_ids` does (`cls(root=data_root / default_root)`, `transform_item(dataset[i])` per record, `close_lmdb()`), and the pydantic values of each item (experiment, reference) go through `torchcell.artifacts.walk.distinct_refs` (`dataset_artifact_refs`). `only` names a subset; a name the manifest does not serve raises before anything opens.
- `manifest_artifact_refs(manifest)` is the release-level map (dataset -> refs) that `releases.release_from_manifest` and `release_snapshot.snapshot_from_manifest` carry, and it is None when ANY entry is None, so a half-recorded manifest never reads as complete.
- CLI: `kg_manifest --manifest M artifact-refs --data-root ROOT [--dataset NAME ...]` saves the manifest and prints `<name>: <n> pointer(s)` per walked dataset or `<name>: no ref-bearing class in its closure`. Both slurm scripts run it right before `releases stamp`.

The refs come from the DEV tree, not from the store: the full build reads those same LMDBs, so there they are what was served; in an increment, a served dataset's dev LMDB may have moved since its admission (the closure gate stops schema changes, not loader-data changes). Importing `ArtifactRef` from `torchcell.artifacts.ref` runs the artifacts package `__init__` (resolver, tiers, genome registry): measured 1.0 s wall for `import torchcell.artifacts.ref` against 0.18 s for `import torchcell.knowledge_graphs.releases` before the change, one run each.

Tests (`tests/torchcell/knowledge_graphs/test_kg_manifest.py`): `test_record_artifact_refs_walks_only_ref_bearing_closures`, `test_record_artifact_refs_only_records_the_named_subset`, `test_cli_artifact_refs_saves_the_manifest_and_prints_one_line_per_dataset`, `test_manifest_entry_without_artifact_refs_loads_as_unrecorded`.
