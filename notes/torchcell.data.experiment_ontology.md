---
id: be48fzdm8khyudrzi651ves
title: Experiment_ontology
desc: ''
updated: 1792696947559
created: 1792696947559
---

## 2026.10.22 - The experiment's annotation to the schema DAG, and the encoder over it

Built for [[plan.030-ontology-conditioning.2026.10.22]] (D1, D2 revised). A stored record is an instance of the schema, so it is already an annotation to the ontology; this module reads it into a fixed vector over a closed node vocabulary and propagates it to the `Experiment` root.

Nodes (147 on the S3Q store, depth 4): the schema classes of the experiment, phenotype and environment lanes as `build_ontology_graph` reads them from the live pydantic models (inheritance parent per class, composition edges per field), value nodes for the closed vocabularies those classes carry (a media component as `MediaComponent:<compound>:<role>`, a dropout as `Media.dropout:<compound>`, the medium's state, base medium and `is_synthetic`, the temperature unit, the aerobicity), one numeric node `Temperature.value` (celsius / 100), and the reference's experiment and phenotype classes under `ref:`. `dataset_name` and every provenance field are excluded by construction. The value vocabulary is the media library plus the records scanned at build time (`scan_store_records`, 25 per source); a value outside it raises.

Encoder: DCell's per-term layer over the concatenated children, written as a sum of per-child projections so a variable number of children batches: `s_v = tanh(sum_{c in children(v)} P_c s_c + Q_v [x_v ; present_v] + b_v)`, leaves first, one depth at a time; a node neither annotated nor above an annotated node is held at zero; the root state is projected to the token. A plain mean over all children diluted one differing leaf (the uracil dropout among fifteen media children) to nothing at the root, which is why the aggregation is a sum. At initialization the Costanzo-vs-Kuzmin token difference is about 1e-3 and the fitness-vs-interaction difference larger; both are learnable.

Measured on the S3Q store (`smoke_ontology_annotation_030.py`, PASS): the two Kuzmin screens of every measurement type annotate identically; Costanzo rows differ from Kuzmin rows by `Media.dropout:uracil` and the temperature value only (Costanzo carries two profiles, 26 C and 30 C); fitness and interaction rows differ in their experiment, phenotype and reference classes.
