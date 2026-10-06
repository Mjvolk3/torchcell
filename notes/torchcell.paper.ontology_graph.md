---
id: zweqpkhf3mx8b5wqnsifkxf
title: Ontology_graph
desc: ''
updated: 1790983321608
created: 1790983321608
---

## 2026.10.05 - Behavior mixins are not drawn as the parent (#640)

`build_ontology_graph` took a class's first schema model base as its inheritance parent. PR #640 lists `HashableProvenanceGapMixin` FIRST in the bases of `BarcodedKanMxDeletionPerturbation`, `HeterozygousDeletionPerturbation` and `ConditionalAllelePerturbation`, because pydantic keeps the first `__hash__` it finds, so the figure filed all three under the mixin in the provenance lane and `descendants_of("GenePerturbation")` lost them (CI failure `test_real_schema_genotype_backbone_edge_is_never_drawn_bold`). `BEHAVIOR_MIXINS = {"HashableProvenanceGapMixin"}` and `_inheritance_parent` now skip such a mixin when the class has another schema model base; `ConstructedOrf`, whose only schema base is the mixin, keeps it. The three committed SVGs under `notes/assets/images/schema-ontology/` were regenerated with `paper/nature-biotech/scripts/generate_ontology_diagram.py` (the `ontology-drift` job byte-compares them). Test: `tests/torchcell/paper/test_ontology_graph.py::test_behavior_mixin_listed_first_is_not_drawn_as_the_parent`.

The per-class "declared" field set now reads `obj.__dict__.get("__annotations__", {})` instead of `getattr(obj, "__annotations__", {})`, for the reason recorded in [[tests.torchcell.datamodels.test_ontology_coherence]]: in an environment where `ABCMeta.__annotations__` exists, `getattr` returns an ancestor's annotations for a class with none, which would mark inherited fields as declared. The committed SVGs are byte-identical under either read in the local env.
