---
id: 6n9p2a53i3zhm2jluln1om8
title: tc_ontology
desc: ''
updated: 1695170558223
created: 1695170380293
---
## Tracking Media in the Ontology

Need to keep track of Media maybe by making a note for each publication. This would really be best if it is linked to the the specific publication notes that contain the necessary information on any given publication.

Here is an example of why we should do this. YPD a YEPD are the same. Yeast Extract Peptone Dextrose with their corresponding concentrations. YEPD + G418 for DMA (Deletion Mutant Array) Growth. Need to pay careful attention to this, may not matter if it has already been proven within reason that the addition of G418 creates a small enough deviation.

## 2026.09.30 - Counted Headers and Joined Endpoints (Issue #532)

The compact `print_schema_mappings` headers were the literals `NODES (16 total)` and `EDGES (11 total)`; they now print `len(nodes)` and `len(edges)` (26 and 13 for the committed schema). List-valued edge endpoints printed as Python list reprs (`['genotype', 'segregant genotype']`); `_endpoint` now joins them with ` | ` in YAML order (`genotype | segregant genotype`). Tests: `test_compact_headers_count_the_schema`, `test_real_schema_expanded_joins_list_endpoints_with_a_bar`.
