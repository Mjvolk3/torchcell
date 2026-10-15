---
id: 4lyw6e013cztgpr7brtkccs
title: interned-id-collision
desc: ''
updated: 1792072615620
created: 1792072615620
---

## 2026.10.15

- [x] `interned-id-collision` worktree (lane N of the ontology unification): reproduced lane H's finding on the job 4901 partial from the CSVs alone, read-only ([[experiments.database.scripts.interned_id_collision_counts]]): 218,258 whole genotypes and 135 Hale backgrounds exist only as `InternedConstant` nodes; 1,948,098 `GenotypeMemberOf`, 121,180 `PerturbationMemberOf` and 709,716 `SegregantBackgroundOf` rows point at them; every blob pointer resolves.
- [x] Asked the served KG 3.0 store the same question through bolt, read-only ([[experiments.database.scripts.interned_id_collision_served]]): 14,573 genotypes (Bloom2019 13,630, Caudal 943) and 518,883 `GenotypeMemberOf` relationships are affected today. Release-note text in [[database.kg-4-pre-build-verification]].
- [x] Fix: `constant_id(kind, payload)` namespaced by label and kind, `verified_constant` and `fetch_constants` verify by the node's own kind ([[torchcell.datamodels.interned_constant]]); the sink refuses an id under a second label ([[torchcell.fast_csv]]); the CSV endpoint gate checks endpoint labels against the schema yaml through the row's ancestry and node ids against a second label ([[torchcell.knowledge_graphs.csv_endpoint_gate]]).
- [x] Regression tests fail on `main` and pass here: `tests/torchcell/adapters/test_interned_id_collision.py`, `tests/torchcell/test_fast_csv.py`, `tests/torchcell/knowledge_graphs/test_csv_endpoint_gate.py`. No schema-change record (no `schema.py` change, Rule 8).
