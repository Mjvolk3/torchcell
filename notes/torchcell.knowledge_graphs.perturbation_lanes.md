---
id: pfeujzgi0l48mpwjxz9quij
title: Perturbation_lanes
desc: ''
updated: 1792055080854
created: 1792055080854
---

## 2026.10.15 - The lane arithmetic, stated once

Stage 1 of the ontology unification (Path U, [[scratch.2026.10.15.ontology-unification-decision]] work plan row 1; constitution principle 12, [[database.constitution]]). The adapter's edge methods walk every perturbation of a genotype and of an environment, while its node methods partition the leaves by class, so a loader that builds a leaf under a conf enabling the edge and no node lane holding that leaf writes a `perturbation member of` row whose start node does not exist. The full import drops it (`skip_bad_relationships: true`), which is how the Carruthers `GeneAdditionPerturbation` records lost their rows ([[scratch.2026.10.15.ontology-audit-C_serving]] section 1.3).

`torchcell/knowledge_graphs/perturbation_lanes.py` is the one statement of which leaves each node method emits, read off the adapter's own tuples:

| node method | leaves |
|---|---|
| `perturbation (chunked)` | every `GenePerturbationType` member (43) |
| `bacterial perturbation (chunked)` | `BACTERIAL_PERTURBATION_LEAVES` (11) |
| `bacterial sequence variant perturbation (chunked)` | `BACTERIAL_VARIANT_PERTURBATION_LEAVES` (3) |
| `environment perturbation (chunked)` | every `EnvironmentPerturbationType` member except `PhagePerturbation` |
| `phage perturbation (chunked)` | `PhagePerturbation` |

`unserved_leaves(genotype_leaves, environment_leaves, methods)` returns the leaves an enabled edge method walks and no enabled node method emits. Gating on the edge method is what makes the seven natural-isolate confs (which omit `perturbation to genotype (chunked)` by design) need no allowlist. `loader_leaf_imports(loader_path, surface)` intersects a loader's DIRECT schema imports (`loader_schema_deps`, not the closure) with the two unions.

Three readers:

- `tests/torchcell/adapters/test_loader_lane_exhaustiveness.py`, one parametrized case per mapped dataset (170 with the private map). Measured today: 3 xfail(strict) (`PhageRbTnseqMutalik2020Dataset`, `IsoprenolTiterCarruthers2025Dataset`, `ProteomeCarruthers2025Dataset`), 174 passed. The literal `SIBLING_IMPORT_ALLOWLIST` names the three multi-class entries whose import serves a sibling class in the same module (Menasalvas selection for the proteome's variant leaves; Carruthers Campaign and FoldChange for the titer and proteome classes' gene addition), each checked to still be imported and to name a sibling of the same module that is not itself allowlisted. `test_the_known_gaps_are_exactly_the_three_the_unification_closes` pins the xfail set, so a fourth gap is a failure and not a mark. When lane H lands (one `_perturbation_node` with the label chosen by leaf), the three XPASS strict and the marks come off.
- `torchcell.database.build_dataset_lmdb.mapped_store_lanes`, the rebuild preflight's second walk, reading `perturbation_leaves` off every mapped store's build manifest ([[torchcell.provenance.build_manifest]]).
- The closure rule the audit first proposed was run and rejected: 55 violations, nearly all base classes reached through subclassing ([[scratch.2026.10.15.ontology-review-3-serving]] finding 6).

## 2026.10.15 - One lane per axis (ontology unification G)

`GENOTYPE_NODE_METHODS` is `("perturbation (chunked)",)` and `ENVIRONMENT_NODE_METHODS` is `("environment perturbation (chunked)",)`: each method serves its whole union and chooses the label by leaf (`sequence variant perturbation` for a composed call, `phage perturbation` for a phage), so `genotype_lanes()` and `environment_lanes()` no longer read class tuples off the adapter. The only way a leaf can dangle now is a conf that enables the edge method without the node method of that axis, which `unserved_leaves` still reports and `mapped_store_lanes` still blocks on. `test_loader_lane_exhaustiveness.py` lost its three `xfail(strict=True)` marks and the `KNOWN_GAPS` table; `test_the_known_gaps_are_exactly_the_three_the_unification_closes` asserts `[]`.
