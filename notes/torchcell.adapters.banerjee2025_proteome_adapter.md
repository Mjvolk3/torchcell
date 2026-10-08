---
id: tt1gxs9ktw1jsj6vr6n1zsx
title: Banerjee2025_proteome_adapter
desc: ''
updated: 1791452201228
created: 1791452201228
---

## 2026.10.08 - Banerjee 2025 proteome adapter

`torchcell/adapters/banerjee2025_proteome_adapter.py`, conf
`torchcell/adapters/conf/proteome_banerjee2025_adapter.yaml`, serving
`ProteomeBanerjee2025Dataset` ([[torchcell.datasets.pputida.banerjee2025]]).

Six records. Four carry one `bacterial perturbation` node each: a
`PromoterReplacementPerturbation` on PP_0897 for the two promoter variants and a
`BacterialDeletionPerturbation` on PP_0897 for the two deletion samples. The two
parental records carry none, and all four strains share the `D1b_gf`
`BacterialStrainBackground` on one `AssemblyReferenceGenome`. The environment carries
`EnvironmentPhysicalPerturbation` carbon-source edits, so the environment-perturbation
pair is enabled.

No new graph class. `PromoterReplacementPerturbation` is already in
`cell_adapter.BACTERIAL_PERTURBATION_LEAVES`, and `protein abundance phenotype` is
already declared in `biocypher/config/torchcell_schema_config.yaml`.

**The temperature methods are enabled and emit nothing.** Every record's
`Environment.temperature` is `None` with a typed `ProvenanceGap`, because the
shotgun-proteomics Methods state no incubation temperature. `_temperature_node` and
`_temperature_to_environment_edge` both return an empty list for a gapped temperature,
so the canonical enable-list for this record shape is kept rather than the conf
diverging from every sibling proteome conf. The adapter-construction harness builds the
expected conf FROM the record shape, so omitting the methods fails
`assert_construction`; that is what settled it.

Four pin places, all derived rather than incremented:
`dataset_adapter_map` (`ProteomeBanerjee2025Dataset -> ProteomeBanerjee2025Adapter`), the
re-export in `torchcell/adapters/__init__.py`, the first P. putida row of
`tests/torchcell/adapters/_bacterial_adapter_cases.py`, and the matching first P. putida
entry of `torchcell/knowledge_graphs/conf/kg_bacteria.yaml` (the gate asserts positional
equality). The three counts in `tests/torchcell/adapters/test_bacterial_adapters.py` go
44 -> 45, `len(dataset_adapter_map)` 95 -> 96, and `BACTERIAL_DATASETS` in
`tests/torchcell/knowledge_graphs/test_build_time_projection.py` gains
`ProteomeBanerjee2025Dataset`. The verification runner is registered in
`BACTERIAL_PROTEIN_ABUNDANCE_DATASETS`, which is not one of the eight registries the
root-count assert covers, so that count is unchanged.
