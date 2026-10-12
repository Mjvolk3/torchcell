---
id: 9yfrnoy1nyqhdx6jqele6cq
title: Schema_change
desc: ''
updated: 1791803411667
created: 1791803411667
---

## 2026.10.12 - Schema-change records, the tripwire and the enforcement check

`torchcell/provenance/schema_change.py` implements [[torchcell.datamodels.schema-change-policy]]: the `SchemaChangeRecord` model (one JSON file per change under `database/schema_changes/`), the kind rules that check each claimed kind against `schema_impact`'s reason strings in both directions (`admissible_kinds`, `kind_problems`), the grandfathered baseline (`GrandfatheredBaseline`, `database/schema_changes/grandfathered.json`, 284 symbols at `eb07bd213`), the chain check the enforcement test runs (`symbol_chain_problems`, `enforcement_problems`), the index-vs-merge-base tripwire (`branch_problems`), and the `schema-change` CLI (`template`, `new`, `check`, `list`, `tripwire`, `baseline`).

It reuses rather than re-derives: surfaces from `kg_manifest.surface_at_ref` / `surface_in_worktree`, the diff and its reasons from `schema_impact.diff_surfaces`, the loader map from `schema_impact.map_impacts`, the served set from the newest `release_snapshot.load_snapshots` entry, and `SourcedValue` for quotes. It imports nothing from `schema.py`, so it adds no symbol to the fingerprinted surface.

An unknown `schema_impact` reason raises in `admissible_kinds` rather than passing as any kind, so a new reason there fails the record tests until the table is extended.

Tests: [[tests.torchcell.provenance.test_schema_change]] (rules, models, chains, the tripwire and the CLI in a throwaway git repository), [[tests.torchcell.datamodels.test_schema_change_records]] (the real records against the real schema).
