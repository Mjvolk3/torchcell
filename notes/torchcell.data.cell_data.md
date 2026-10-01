---
id: al1q4hvt263upphk7uiqn8i
title: Cell_data
desc: ''
updated: 1790816094280
created: 1790816094280
---

## 2026.09.30 - GO feature counts indexed genes; cycle fallback orders descendants (issue #538)

- The `gene_ontology` feature `x` was `len(gene_set)`, counting annotated genes absent from the base graph, while `term_gene_counts` counted only indexed genes. `x` now equals `term_gene_counts` for every term with a gene set.
- `compute_strata`'s cycle fallback peeled nodes with no outgoing edge, but every node Kahn's pass leaves has an unassigned parent, so the loop never ran and the cycle and all its descendants shared one stratum. The fallback now condenses the remainder into strongly connected components and assigns them parent-first by topological generation: a cycle shares one stratum and each descendant comes after it. Stratum 0 is the roots and DCell processes strata in descending order (the docstring said "from leaves to root"; corrected). The GO is_a graph is acyclic in practice, so real strata are unchanged.
- Evidence: `tests/torchcell/data/test_cell_data.py` (`test_gene_ontology_feature_counts_only_genes_in_the_base_graph`, `test_a_descendant_of_a_cycle_gets_a_later_stratum`), `tests/torchcell/data/test_cell_data_synthetic.py::test_compute_strata_places_a_cyclic_component_in_one_stratum_after_the_dag`.
