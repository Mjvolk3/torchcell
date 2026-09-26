---
id: vu8km39l9w4ke1posmoktop
title: Test_datamodels_roundtrip
desc: ''
updated: 1790412600609
created: 1790412600609
---

## 2026.09.26 - Round trip over every datamodels model

One parametrized case per pydantic model found by walking `torchcell.datamodels` (109 on 2026.09.26; `compound_identity_curate` is skipped for its optional dependency). A generic builder fills required fields from annotations plus a few field-name conventions the validators enforce (`systematic_gene_name`, `*so_id`, `*sha256`, `pubmed_id`); eight classes carry hand-written `EXAMPLES` because a validator wants a specific shape (`Concentration`, `EnvironmentResponsePhenotype`, `ExpressionModulationPerturbation`, `Media`, `PresenceAbsencePerturbation`, `Publication`, `VisualScorePhenotype`, `CalMorphPhenotype` with a real CalMorph label), and a companion test proves each example is needed. Five classes are strict xfails in `UNCONSTRUCTIBLE`: the abstract bases `Phenotype`, `Experiment`, `ExperimentReference` (`label_name` must be a class attribute of a concrete subclass) and the two conversion registries (callables, not JSON-serializable by design). Pinned per class: dict and JSON round trips are equal, `model_dump` is deterministic, `model_json_schema` is JSON-serializable and stable. Phase 1 of [[plan.test-suite-buildout.2026.09.25]].
