---
id: 5j830foe5jjzlj9giskfdja
title: Levels
desc: ''
updated: 1783563791335
created: 1783563791335
---

## 2026.07.08 - L0-L4 levels: escalating tiers that gate a record from raw to trustworthy

This module exists so "verified" means something concrete and universal rather than per-dataset folklore. It defines five ordered tiers -- L0 structural (does the record instantiate its schema), L1 completeness (count / known-keys vs an oracle), L2 value fidelity (finite / in-range / cross-method agreement), L3 semantic convention (units hold, e.g. `log2(sample/ref)`), L4 cross-source (overlapping entities agree across independent datasets) -- as small, composable check functions each returning a `LevelResult`. Every per-datatype verifier (WS5-WS10) assembles the relevant checks from this one vocabulary, so the meaning of a level is shared across the whole ontology.

- Deliberately tiny building blocks: a verifier picks which checks apply, rather than each family reinventing "what counts as verified."
- The tiers are a GATE, not a set -- a record is only as trustworthy as the highest level it clears; a report that skips levels is not silently "passed."
- Phase A scope: operates purely on pydantic/LMDB records, no graph -- keeps the harness cheap enough to run on every build.
- Failures are returned as DATA (indices, offending values), never raised or swallowed -- surfacing what broke is the harness's whole job. See [[torchcell.verification.report]] for the result/report models.
