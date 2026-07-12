---
id: re5hs9oed1a8poek9pxx1u5
title: Plasmid
desc: ''
updated: 1783563784291
created: 1783563784291
---

## 2026.07.08 - Plasmids as first-class total-genomic-content

This module exists so a plasmid that is physically PRESENT in a cell during phenotype collection becomes a first-class part of the genotype, not an untracked footnote. Under torchcell's stance that a perturbation is an edit to the TOTAL genomic content in the cell (deletions, integrations, alleles, present plasmids, heterologous cassettes), owning plasmid sequence + annotation as pydantic objects is what makes both sequence-level genotype fidelity and future inverse strain design representable.

- Owns an SBOL/Sequence-Ontology-aligned SUPERSET (`Component`/`Feature`/`Location`/`SORole`), so we interoperate with the standards while capturing more than they mandate.
- Ingests GenBank first (BioPython) with sha256-pinned provenance; GenBank/GFF3/SBOL writers are deliberately deferred future work.
- Present-content vs construction-source is a property of how a perturbation REFERENCES a Component, not of the sequence: a plasmid used only to build a strain becomes a chromosomal `GeneAddition`, not stored cell content.
- `feature_sequence` (flank + circular wrap) and `subcomponent` (design composition) are the extraction/design operations the store is built to serve.
