---
id: 476axzexz7kauit9kp4qtr3
title: Puddu2019_adapter
desc: ''
updated: 1791732611921
created: 1791732611921
---

## 2026.10.11 - Puddu 2019 adapter

`torchcell/adapters/puddu2019_adapter.py` (`GenomeArchitecturePuddu2019Adapter`), conf `torchcell/adapters/conf/genome_architecture_puddu2019_adapter.yaml`, map entry in `torchcell/knowledge_graphs/dataset_adapter_map.py`.

15 node methods and 13 edge methods. Each record is one sequenced colony: the designed kanMX deletion with its DETECTED barcodes is a `perturbation` on a `genotype` (the full form, with `perturbation (chunked)` and `perturbation to genotype (chunked)`, because a colony has exactly one perturbation), the 1.8 ml liquid YPAD culture is an `environment` with its media and temperature as first-class nodes, and the readout is the new `genome architecture phenotype` node class.

That node class (`biocypher/config/torchcell_schema_config.yaml`, `is_a: phenotypic feature`, so `browser_style.LANE_OF_LABEL` needs no edit) carries the 28 features as one JSON string beside `assay`, with `genome_architecture_sd`, `uncertainty_type`, `n_samples`, `sample_unit`, `sample_id` and `sample_accession` typed. `assay` has to travel with the JSON: a key's unit basis is declared by the assay, never by the key's spelling. The sample id and its archive accession stay typed because they are what makes one stored profile traceable to the reads it was measured from.

The screen is one medium at one temperature with no environment perturbation, so the environment-perturbation node and edge pair is left off.

Dataset, counts and verification: [[torchcell.datasets.scerevisiae.puddu2019]].
