---
id: z3soqzrxbb42u4kkmbxlagu
title: Cooper2010_adapter
desc: ''
updated: 1789463997105
created: 1789463997105
---

## 2026.09.15 - Adapter for the Cooper 2010 amino-acid metabolome

`torchcell/adapters/cooper2010_adapter.py`, class `AminoAcidCooper2010Adapter`, a clone of
[[torchcell.adapters.mulleder2016_adapter]] pointing at `conf/amino_acid_cooper2010_adapter.yaml`,
which is a byte-for-byte copy of `amino_acid_mulleder2016_adapter.yaml` (asserted by
`test_conf_is_the_mulleder_metabolite_enable_list`). The conf is the whole surface: genome,
experiment/reference, genotype + KanMX perturbation, environment + media + temperature (the
temperature node is absent per record since `temperature=None`), metabolite phenotype +
reference, dataset, publication; no environment-perturbation methods because the environment
carries none. Registered in `torchcell/adapters/__init__.py` (`proteome_metabolome_adapters`) and
`torchcell/knowledge_graphs/dataset_adapter_map.py`.

The metabolite phenotype node id is `sha256(json.dumps(phenotype.model_dump()))`, so the ragged
key set and the measurement type `ce_lif_peak_area_ratio_to_plate_mean` are part of the identity
(`test_metabolite_phenotype_node_is_keyed_on_the_serialized_phenotype`). Dataset note:
[[torchcell.datasets.scerevisiae.cooper2010]].

Tests: `tests/torchcell/adapters/test_cooper2010_adapter.py` (5 passed);
`tests/torchcell/adapters tests/torchcell/knowledge_graphs` 109 passed on 2026-09-15.
