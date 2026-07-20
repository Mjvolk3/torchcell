---
id: t1plt4o67xd6vzi11vurvc5
title: Constructed_10_dmf_reference
desc: ''
updated: 1784566132316
created: 1784566132316
---

## 2026.07.20 - Published DMF Baseline So Constructed Doubles Can Be Judged

Exists to answer, at the bench, "is this double mutant's fitness what the literature
says it should be?" When the wet-lab builds the 8 set-cover doubles there is otherwise no
reference to compare a measured fitness against. This pulls published double-mutant
fitness ± SD (plus digenic interaction ε and its p-value) for **every** pair among the 10
properly-constructed genes, and flags which pairs are the 8 construction targets
(`is_optimized_double`), so the comparison is a lookup rather than a re-query.

Coverage: all 45 pairs = C(10,2); Costanzo2016 has DMF for **44/45**, Kuzmin2018 for 4,
Kuzmin2020 for 2 — the Kuzmin sparsity is structural (their SMF/DMF values are byproducts
of trigenic arrays, so only query/array strains appear). The 8 construction doubles sit
~0.97–1.12, i.e. near-neutral, which is what makes a measured departure interpretable;
every low-fitness double contains CBF1/YJR060W, whose single defect dominates its pairs.
Data is filtered from the already-queried panel-12 doubles table, so no LMDB re-query.

Forest figure + table: [[experiments.010-kuzmin-tmi.scripts.topk_triples_from_constructed_10]].
