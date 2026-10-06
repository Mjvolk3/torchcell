---
id: paqad729rrvlgcck4rzj7l8
title: Test_graph_processor
desc: ''
updated: 1790415116697
created: 1790415116697
---

## 2026.09.26 - The Perturbation processor on a three-gene cell graph

Exact tensors on a cell graph that carries one physical edge (four after self loops), so the assertion that the processor copies no edge type is falsifiable: the processor stores the UNION of perturbed genes for the whole batch (`perturbation_indices` [0, 1, 2] for records perturbing {YAL001C} and {YAL002W, YAL003W}), phenotypes in COO form (values [0.9, 0.4], type index 0, sample indices [0, 1], `phenotype_types == ["fitness"]`), the statistic tensors empty but `fitness_se` declared, and one filled statistic when a record carries `fitness_se`. An empty batch raises. The subgraph processors stay behind `--data` in `test_graph_processor_equivalence.py`. Phase 3 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.27 - Two-class phenotype info and dict-valued labels

Two more `Perturbation` tests: with two phenotype classes in `phenotype_info` the type and statistic indices follow the class order and the reference phenotypes are excluded; a dict-valued CalMorph label with its CV is flattened in sorted-key order. The subgraph processors now run hermetically in [[tests.torchcell.data.test_graph_processor_subgraph]] and [[tests.torchcell.data.test_graph_processor_unperturbed_dcell]]. Phase 6 of [[plan.test-suite-buildout.2026.09.25]].

## 2026.09.30 - Phase 14: the subgraph family on a three-gene graph, DCell on the conftest graph

Six to twenty-eight tests (35 cases); this file alone 14 to 79 percent, the four `test_graph_processor*.py` files together 84 to 96 percent. What remains is dead or unobservable (the `if not reaction_info` returns, the `device is None` checks, a subsystem-derived `w_growth` computed and discarded, the neighbor `process` tail behind the list/`.tolist()` finding on record). Pinned: each processor's exact output on a hand-built perturbation (removed and masked nodes, the edge index after subsetting, the mask tensors, per-graph attributes), the incidence handling, the error messages.

Findings: Subgraph and Incidence take `w_growth` from the stored tensor while Lazy takes it from the subsystem labels, [0.5, 0, 1] versus [1, 0, 1] on the same graph (lines 367, 1725); a tensor `subsystem` can never mark Growth (1604); the incidence cache is never tied to the graph it was built from, so processing a second graph reuses the first graph's cache and emits an edge with node index -1 (664, 1350); `Unperturbed` raises `TypeError` on a phenotype with no statistic name (1926-1929); DCell omits `perturbation_indices_batch` when no perturbed gene is a node (2275-2279); the neighbor processor writes no phenotype placeholder when there is no value.

## 2026.09.30 - Issue #527 findings retired

All six Phase 14 findings in this file are fixed in [[torchcell.data.graph_processor]] and now asserted as contracts: the three masking processors return the stored `w_growth` [0.5, 0, 1] exactly; a tensor `subsystem` leaves the stored [1, 0, 1] untouched; one incidence processor fed a three-gene then a four-gene graph returns exact edges [[0, 1], [1, 1]], pert_mask [F, F, T, T], regulatory [[2], [1]], per-gene incidence lengths [2, 2, 2, 1], then [1, 2, 2] on return to the first graph, and a same-size graph with permuted edges rebuilds (gene 0's list [2]) while an equal-content copy returns the zero report; `Unperturbed` writes `{visual_score}` for VisualScorePhenotype and `{fitness, fitness_se}` for FitnessPhenotype; DCell writes an empty long `perturbation_indices_batch` (superseded below, it does not collate); the neighbor processor writes the four placeholder keys (values [NaN], type [0], sample [0], types ["gene_interaction"]). 61 to 65 passing cases across the four `test_graph_processor*.py` files.

## 2026.09.30 - Review corrections (issue #527)

DCell writes no per-sample `perturbation_indices_batch`; the outside-gene test now collates [inside, outside] with `follow_batch=["perturbation_indices"]` and asserts `perturbation_indices` [1], batch [0]. The tensor-subsystem test stores `w_growth` [0, 1, 0], the opposite of a naive decode of `tensor([1, 0, 1])`, so a reintroduced decoder fails it.

## 2026.10.06 - Phase 21: the one-line guards

- `SubgraphRepresentation`, `IncidenceSubgraphRepresentation` and `LazySubgraphRepresentation`: `_add_reaction_data` with an empty reaction_info and `_process_metabolic_network` on a graph without reactions write nothing.
- The `device is None` guard of `_add_phenotype_data` (dead in practice: every `__init__` sets the CPU) restores the CPU and writes the same phenotype tensors, for the three processors and `DCellGraphProcessor`.
- `GraphProcessor` cannot be instantiated; its abstract `process` body returns None.

Left uncovered: `NeighborSubgraphRepresentation.process` after line 2696, unreachable because of the pinned list-versus-tensor finding.

## 2026.10.06 - Phase 21 audit 2

The base-class test keeps only the instantiation refusal; the call of the abstract `pass` body was removed as padding.
