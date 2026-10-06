---
id: 05h2iudwvtya116sw7gjnn6
title: fix-main-green-after-640
desc: ''
updated: 1790983346676
created: 1790983346676
---

## 2026.10.02

- [x] PR: main is green again after #640. [[torchcell.paper.ontology_graph]] does not draw `HashableProvenanceGapMixin` as the parent of the #640 perturbation leaves and reads own annotations from `__dict__`; [[tests.torchcell.datamodels.test_ontology_coherence]] reads own annotations from the class namespace; [[tests.torchcell.data.test_neo4j_query_raw_single_pass]] pins the exact declared environment-class map on top of the cache fix in 8533adeb5. Test: [[tests.torchcell.paper.test_ontology_graph]].
- [x] PR follow-up: [[tests.torchcell.adapters.test_ohya2005_adapter]] exact node and edge lists for the sourced YPD medium of ca0734254 (#622); own annotations via `inspect.get_annotations` in [[torchcell.paper.ontology_graph]] and [[tests.torchcell.datamodels.test_ontology_coherence]].
