---
id: y25pwyyj07mtuhrxbth1oix
title: Test_ontology_coherence
desc: ''
updated: 1791256692358
created: 1791256692358
---

## 2026.10.05 - Own annotations read from the class namespace (#640)

CI on main failed `test_gap_mixin_rule_is_never_weakened_by_a_subclass[HashableProvenanceGapMixin]` although `HashableProvenanceGapMixin` declares no field. Reproduced in an Ubuntu 24.04 container with the CI interpreter (actions/python-versions 3.13.0), `env/requirements*.txt` and pydantic 2.13.5: at collection end `ABCMeta.__dict__` held a lazily created `{}` for `__annotations__`, which sits ahead of `type`'s descriptor in `ModelMetaclass`'s MRO, so `getattr(cls, "__annotations__")` fell through to the class MRO and returned `ProvenanceGapMixin`'s dict. The local conda env never creates that entry and passes. The test now reads `cls.__dict__.get("__annotations__", {})` (the class's own namespace, which is what "redeclares" means) and additionally asserts that the resolved `provenance_gaps` field is `list[ProvenanceGap]` with `default_factory=list` and that the resolved `validate_provenance_gaps` is the mixin's own function. Which import creates the `ABCMeta` entry was not identified.

## 2026.10.06 - inspect.get_annotations

The own-annotation read is now `inspect.get_annotations(cls)`, since `cls.__dict__` is empty under Python 3.14's lazy annotations. It passes in the CI-equivalent container (Python 3.13.0, pydantic 2.13.5).
