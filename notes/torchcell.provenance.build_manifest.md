---
id: t0iqox0eeewcefc2tdzhwby
title: Build_manifest
desc: ''
updated: 1792055097906
created: 1792055097906
---

## 2026.10.15 - `perturbation_leaves`, manifest schema version 2

Stage 1 of the ontology unification, guard (b1) of [[scratch.2026.10.15.ontology-audit-C_serving]] section 4, placed in `write_build_manifest` rather than the optional verifier per [[scratch.2026.10.15.ontology-review-3-serving]] finding 7 (the `--verify` path returned `True` with "NO PER-DATASET VERIFIER" for 38 of the mapped stores, so a walk inside the verifier would never have run for them).

`walk_perturbation_leaves(root)` streams every record of `<root>/processed/lmdb` once, the same read path as the store's own (`pickle.loads`, then `resolve_interned` on the genotype and environment dicts, since a loader may intern either), reads each perturbation dict's `perturbation_type` literal and names its class through `leaf_class_by_perturbation_type()`, a bijection over the 43 genotype and 5 environment union members (measured today; the walk raises if the literal is shared or unknown). No record is validated, so the pass costs what `compute_gene_set_sequential` costs. A `SegregantGenotype` carries no perturbation list and contributes nothing. The result is `BuildManifest.perturbation_leaves: PerturbationLeaves | None` with sorted class names under `genotype` and `environment`.

`MANIFEST_SCHEMA_VERSION` is 2. A version 1 manifest (no field) reads `stale`: `check_manifest` reports it in `StaleResult.missing_fields`, `check_store` lists `perturbation_leaves` in the store's `drift`, and the CLI prints `[STALE] <name> -> rebuild; manifest lacks: perturbation_leaves`. Consequence, intended: after this lands every one of the 170 dev stores reads stale under `--list-stale` until the sweep (work plan stage 8) rewrites its manifest, which is why the field lands before the sweep and not after it. The preflight's lane check (`mapped_store_lanes`, [[torchcell.database.build_dataset_lmdb]]) reads the field without opening the store.

Tests added to `tests/torchcell/provenance/test_build_manifest.py`: the bijection, the walk over three records with interned genotype and environment parts and a segregant record, the empty-store and unknown-literal refusals, the version 1 manifest reading stale through `check_manifest`, `check_store` and the CLI, and the round trip.

## 2026.10.15 - content_id_collisions, measured in the same pass (lane V)

`walk_store(root)` is the one streaming pass over `<root>/processed/lmdb`, returning a
`StoreWalk` with the `PerturbationLeaves` the leaf walk already measured and a new
`ContentIdCollisions`: `n_records`, `n_distinct`, `n_groups`, `n_records_lost` and up to
five `example_groups` of LMDB keys. `walk_perturbation_leaves` is kept as the leaf half
of it, because the rebuild preflight and the lane arithmetic are written in terms of
leaves. `BuildManifest.content_id_collisions` records it, `MANIFEST_SCHEMA_VERSION` is
3, and a manifest missing the field reads `stale` exactly as a manifest missing
`perturbation_leaves` does.

Why the count belongs in the manifest. The adapter's `Experiment` node id is
`sha256(json.dumps(experiment.model_dump()))` and `FastCsvSink` deduplicates node ids, so
two records of one store with byte-identical dumps become ONE node and the second
measurement is in the LMDB and absent from the graph, with no error anywhere. Four served
datasets and 65 records were in that state when lane Q first asked
([[experiments.database.scripts.content_id_collision_scan]]), and the number was
computable from the served store the whole time. A property that only a special scan can
see is a property nobody looks at.

The digest is taken over the FULLY RESOLVED experiment dict (`resolve_interned` on the
whole experiment, not only on the genotype and the environment), which is what
`CellAdapter._experiment_node` hashes: a loader pickles `experiment.model_dump()`, so the
stored dict IS that dump and the two agree without validating a record here. A walk that
hashed the `{"$ref": ...}` form would compare pointers instead of records, which a test
pins.

Memory. The digests go into one contiguous `bytearray`, 32 bytes per record, and the
grouping is a stable `numpy` argsort rather than a dict of 30 million `bytes` objects:
970 MB for Hoepfner's 30,308,574 records against about 4.5 GB in the dict form, exact
either way (whole digests, no truncation). The sort is stable, so a group's keys come out
in store-iteration order.

It RECORDS and never refuses. The count is a measured property of a store that already
exists, and a build that ended with no manifest would read `no_manifest`, be rebuilt, and
land in the same place. The refusal lives in the rebuild preflight
(`mapped_store_collisions`, [[torchcell.database.build_dataset_lmdb]]), which is the
point where a store is about to become a graph.

Consequence, intended and already in the plan: version 3 stales every manifest, so the
sweep that stage 8 of the unification plan already schedules is what rewrites them. The
field lands before that sweep for the same reason `perturbation_leaves` did.

Tests added to `tests/torchcell/provenance/test_build_manifest.py`: the five-record store
with two groups and three records lost (with the example groups in iteration order), the
all-distinct store reporting zero, the interned-pointer equality, and the version-1
manifest now naming both missing fields through `check_manifest`, `check_store` and the
CLI.
