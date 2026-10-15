---
id: oqd2sqi9zrp6t1r3u6exm8n
title: Interned_id_collision_served
desc: ''
updated: 1792072591089
created: 1792072591089
---

## 2026.10.15 - The interned-id collision on the served KG 3.0 store

`experiments/database/scripts/interned_id_collision_served.py` asks the live store (`tc-neo4j-readonly`, `bolt://localhost:7687`, database `torchcell`, read transactions only) the same question as the partial's count: every query is anchored on the 21,020 `InternedConstant` nodes, so nothing scans `Experiment` or `Genotype`; 8 s. Result: `experiments/database/results/interned_id_collision_served.json` (tracked).

| measure | value |
|---|---|
| nodes | Genotype 22,657,896; SegregantGenotype 320; PerturbedSegregantGenotype 0; InternedConstant 21,020 (genotype 14,573; environment 6,447); Perturbation 173,766; Experiment 52,668,931 |
| GenotypeMemberOf | 52,668,931 relationships; 518,883 start at an InternedConstant (14,573 distinct) |
| PerturbationMemberOf | 45,297,656 relationships; 0 end at an InternedConstant |
| SegregantBackgroundOf | 0 (Hale 2024 is not in 3.0) |
| per dataset | Bloom2019Dataset 13,630 genotypes only as constants (517,940 rows), CaudalPanTranscriptome2024Dataset 943 (943 rows) |
| sampled blob pointers | 441 + 500, all resolve |

So the data served today has 14,573 genotypes that `MATCH (g:Genotype)` cannot reach and whose `GenotypeMemberOf` rows lead to an information-content node. The query loader is unaffected (every pointer resolves). The 4.0 rebuild with the namespaced id ([[torchcell.datamodels.interned_constant]]) corrects it; the 4.0 release note states the 3.0 defect.
