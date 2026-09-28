---
id: wb4af4lyxa144h1ubflc6qh
title: Physical_axis_coverage
desc: ''
updated: 1790487321655
created: 1790487321655
---

## 2026.09.27 - Do the physical environment fields vary inside any dataset

Reads the flattened served records of the five chemogenomic datasets and, for every physical
and medium field the record carries (medium base, temperature, aerobicity, duration in hours
and in generations, pH as a physical perturbation, solvent, measurement type, assay type),
reports the distinct stated values and the share of records stating one. Writes
`results/physical_axis_summary.csv` (one row per dataset) and `results/physical_axis_fields.csv`
(one row per field and value, with the datasets that carry it).

**Finding.** No physical field varies inside a dataset in a way a model could learn from.

| field | values across the five | in more than one dataset |
|---|---|---|
| temperature | 20, 23, 25, 30, 37 C | 23, 30, 37; stated for 1 to 2% of Hillenmeyer records, 30 everywhere else |
| aerobicity | aerobic, anaerobic | aerobic; only Vanacloig is anaerobic |
| duration, hours | 16, 18, 48 | none |
| duration, generations | 0, 5, 6.5, 10, 15, 20 | 0, 5, 10, 15, 20 (the Hillenmeyer generation arms) |
| pH | 5.0, 7.5, 8.0 | none; stated only by Vanacloig and 2% of Hillenmeyer HOM |
| solvent | DMSO | DMSO (Hoepfner, Wildenhain); unstated elsewhere |
| medium base | SynBase, SC, SD, YP, YPD | SC, SD, YP, YPD |

A temperature, aerobicity or pH channel would therefore take one constant per dataset, which is
the dataset identity written a second way. The unified representation
([[experiments.031-env-chemgen-inhibitor-tolerance.mermaid.unified-input]]) gives these fields no
channel and lets the dataset token carry them; the fields stay in the record so the channel is a
one-line addition the day a served dataset varies one of them.

The same table shows the output disagreement the decoder has to absorb: log2 ratio (Vanacloig,
Hillenmeyer HET), z score (Hillenmeyer HOM, Wildenhain), sensitivity score (Hoepfner); pooled
barcode competition in four datasets and liquid OD in Wildenhain.

Rendered as table t8 of `notes-tex/031-unified-representation` by
`notes_tex_representation_tables.py`.
