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
