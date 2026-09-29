---
id: d25nrcic9casxrycziu7siq
title: Cypher_deps
desc: ''
updated: 1790725761979
created: 1790725761979
---

## 2026.09.29 - Regex dependency extractor

Plan: [[plan.data-release-program.2026.09.29]], Decision 3; issue #468. A regex extractor
over normalized text, not a Cypher parser: the queries in this repo are written in one
block shape (the 025 solid-growth blocks), and a parser dependency would pull a grammar
into the import-light CI job for no gain on that shape.

Normalization: `//` and `/* */` comments become spaces; for every rule that must not see
literal text, the characters inside `'...'` and `"..."` are also blanked. Both forms keep
the original offsets, so a `UNION` boundary found in the blanked text slices the
comment-stripped text identically. Variables are scoped to a `UNION` block.

| field | rule |
|---|---|
| `node_labels` | `:Label` inside `(:Label)`, `(x:Label)`, `(x:A:B)` |
| `relationship_types` | `[:T]`, `[r:T]`, `[r:A\|B]` |
| `properties` | `x.prop` for `x` bound by a node or relationship pattern, `x IN`, or `AS x`; not `x.f(` |
| `label_properties` | `Label.prop` for every label of a node variable `x` in the block |
| `dataset_ids` | `x.id = '<lit>'` / `x.id IN [...]` for `x` labeled `Dataset` |
| `graph_levels` | `x.graph_level = '<lit>'` / `IN [...]`, any `x` |
| `media_names` | `x.name = '<lit>'` / `IN [...]` for `x` labeled `Media` |
| `parameters` | `$name` outside literals |

Documented limits (the extractor under-reports, it never invents): `WHERE x:Label`
predicates, backticked names, label expressions, a literal on the left of `=`, dataset ids
passed as parameters, `CALL {}` scoping, and property reads through relationship
variables (recorded as `properties` only; the graph schema lists no edge properties).

Extracted from the real 025 query (`experiments/025-solid-growth/queries/001_all_solid_growth.cql`,
15 blocks), asserted exactly in `test_cypher_deps.py`:

- labels: `Dataset`, `Environment`, `Experiment`, `ExperimentReference`, `Genotype`,
  `Media`, `PhenotypicFeature`
- relationship types: `EnvironmentMemberOf`, `ExperimentMemberOf`, `ExperimentReferenceOf`,
  `GenotypeMemberOf`, `MediaMemberOf`, `PerturbationMemberOf`, `PhenotypeMemberOf`
- properties: `graph_level`, `id`, `serialized_data`, `state`, `systematic_gene_name`
- label properties: `Dataset.id`, `Experiment.id`, `Experiment.serialized_data`,
  `ExperimentReference.serialized_data`, `Media.state`, `PhenotypicFeature.graph_level`
- graph levels: `edge`, `global`, `hyperedge`, `node`; media names: none (the query filters
  `m.state`); parameters: `gene_set`; 15 dataset ids.
