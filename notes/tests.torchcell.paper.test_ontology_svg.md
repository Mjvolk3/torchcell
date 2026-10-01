---
id: 7k6sa8cvspvz84x2prd6x87
title: Test_ontology_svg
desc: ''
updated: 1790777423960
created: 1790777423960
---

## 2026.09.30 - Phase 17: exact geometry on an eight-class schema, the real schema in memory

New file, twenty-three functions (26 cases), 50 to 99 percent. On a hand-built eight-class schema: exact lane and card geometry from the docstring arithmetic in full and compact form; tree wrapping at the lane height target; exact card, compact-card and edge markup; the lane palette hex values, banner shrink sizes, legend rows and heights; the legend ending at vb_h - 56; the document parsing as XML with element counts; schematic block positions and arrow paths. The real schema rendered in memory: one mark per class, parent link and edge. The explorer HTML lives in the script, not this module.

Findings: the ("Genotype", "GenePerturbation") backbone edge matches nothing in the real schema because `Genotype.perturbations` references the twenty concrete subclasses, so that spine is drawn as twenty hairlines and never bold (line 63); when the parent sits right of the child (ProvenanceGapMixin to Phenotype) the inheritance line runs through both cards (435-441); each elbow arrow's last leg runs 2.9 units backward under the arrowhead (767-781).

## 2026.10.01 - Findings retired (issue #541)

The right-side parent elbow (`M1316.0 116.1 H1713.0 V103.5 H1719.0`, head `l-7`) and the schematic elbow route (radius 2.3, last leg ends exactly at the tip) are asserted with exact coordinates. The `Genotype -> GenePerturbation` backbone Finding stays pinned as left open.
