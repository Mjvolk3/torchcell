---
id: krckj1xurqnpg3afthxk1mm
title: '28'
desc: ''
updated: 1783476309116
created: 1783476309116
---

## 2026.07.07

- [x] Promoted the three scratch CGT Fig 1 working notes into permanent paper notes, correcting the readout description (mean-pool + MLP, not PMA) and re-rendering the mermaid figure assets [[paper.nature-biotech.fig1.block-diagrams]] [[paper.nature-biotech.fig1.perturbation-operator]] [[paper.nature-biotech.fig1.gat-cgt-equivalence]]
- [x] Built a reusable paper-table generator that renders one registry to markdown + LaTeX and computes a gzip "signal" (Kolmogorov proxy) straight from each built LMDB, filling every dataset row including the 20.7M-record Costanzo tables [[torchcell.paper.tables]] [[paper.nature-biotech.scripts.generate_datasets_table]]
- [x] Wrote two SI Supplementary Notes with house-format proofs anchoring the CGT design claims -- Note 1 shows the perturbation operator is functionally equivalent to (and contains) the amortized rebuild, Note 2 proves graph-regularized attention recovers hard-masked graph attention as lambda->infinity and strictly extends it at finite lambda [[paper.nature-biotech.fig1.perturbation-operator]] [[paper.nature-biotech.fig1.gat-cgt-equivalence]] [[paper.proof-writing-standard]]
- [x] Added Fig 5 (drug exposure / robustness) and Fig 6 (metabolism / metabolic engineering) with re-exported figure PDFs and drawio SVG sources, plus supporting methods/results/introduction/discussion, preamble, references.bib, and Makefile/figure-gate updates [[paper.nature-biotech.figures]]

## 2026.07.08

- [x] revitalized supported-datasets table with verified class/KG-adapter status, verbatim DOIs/URLs, and new ingest targets for the Zotero-backed rebuild [[torchcell.datasets.supported-datasets-table-revitalized]]

## 2026.07.13

- [x] Natural-isolate genomic diversity vs KO-expression variability -- bit accounting for CGT inputs (`#66`): per-ORF divergence, core/accessory, codon usage, coding-vs-regulatory π from the population VCF, and a single-KO-vs-natural-isolate DE comparison under Kemmeren's own sourced criterion [[experiments.018-natural-isolate-genomics]]
- [x] Corrected three overclaims from that first pass (Signal composition, gzip order-effect, a phantom isolate drop) and, in the process, confirmed two real loader defects -- see the Corrections section of [[experiments.018-natural-isolate-genomics]]
- [x] `#71` caudal2024 silently omits ~133 gene-absence edits per isolate (`s288c_mask` computed but never used; both loops guard on `core_mask`) -- **genome-fidelity bug, needs a Caudal LMDB rebuild**
- [x] `#72` sameith2015: 70/287 arrays (24%) enter sign-flipped -- GSE42536 is a dye-swap design and GEO declares BOTH ratio directions; the global sign is CORRECT (do not flip), the fix is to recompute per array from `Signal Norm_Cy5`/`Cy3` as kemmeren2014 already does
