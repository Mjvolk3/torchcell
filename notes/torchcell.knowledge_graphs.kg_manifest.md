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
