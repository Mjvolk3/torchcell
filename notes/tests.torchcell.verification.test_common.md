---
id: oxvc5gggq0h07atepub1vu8
title: Test_common
desc: ''
updated: 1790777355770
created: 1790777355770
---

## 2026.09.30 - Phase 17: every shared rule with its exact message

New file, twenty-four tests, 0 to 100 percent of `torchcell/verification/common.py` (the module holds the shared rules; the report writer and level helpers live in `report.py` and `levels.py`). The carrier index maps a stored dump back to its class by key set; `provenance_gaps` with the exact census (top-5 ordering by (-count, name), by reason, by field, worklist, silent fields, classes, and the clause left out when nothing is silent); `canonical_gene_names` in eight cases with exact messages; `uncertainty_sanity` (zero SE not counted, the `fitness_se` fallback, examples stopping at 20 of 21); `compound_identity` and `media_compound_identity` over every compound context; `media_membership` (`library:YPD` versus `derived:YPD`, free-text and missing media); L4 containment 0.667 with off-genome records, the verdict flipping between `min_containment` 0.6 and 0.9.

Finding: with `sgd_genes` given and no gene perturbations, overlap is 0.0, so `gene_containment_sgd` fails ("0.000 of 0") while `current_genome_genes` passes on the same empty set (line 748).

## 2026.10.01 - Findings retired (issue #541)

The empty-gene-set Finding is retired: `test_gene_containment_passes_a_dataset_with_no_genes_vacuously` asserts both L4 results pass at `min_containment=1.0`, the exact empty-set message and details (overlap 1.0). The uncertainty messages now assert "labeled".
