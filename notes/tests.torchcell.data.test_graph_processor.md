---
id: paqad729rrvlgcck4rzj7l8
title: Test_graph_processor
desc: ''
updated: 1790415116697
created: 1790415116697
---

## 2026.09.26 - The Perturbation processor on a three-gene cell graph

Exact tensors on a cell graph that carries one physical edge (four after self loops), so the assertion that the processor copies no edge type is falsifiable: the processor stores the UNION of perturbed genes for the whole batch (`perturbation_indices` [0, 1, 2] for records perturbing {YAL001C} and {YAL002W, YAL003W}), phenotypes in COO form (values [0.9, 0.4], type index 0, sample indices [0, 1], `phenotype_types == ["fitness"]`), the statistic tensors empty but `fitness_se` declared, and one filled statistic when a record carries `fitness_se`. An empty batch raises. The subgraph processors stay behind `--data` in `test_graph_processor_equivalence.py`. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].
