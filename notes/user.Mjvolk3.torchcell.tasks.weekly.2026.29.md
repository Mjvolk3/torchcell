---
id: 34o942tb7way1avtfhxt10m
title: '29'
desc: ''
updated: 1783986627743
created: 1783986627743
---

## 2026.07.13

- [x] Fixed the SI classical-ML MSE table running off the page; the overflow traced to two diverged SVR fits rather than styling, and all three benchmark tables now fit the real journal column width [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [x] Added a Pearson SI table so the correlations quoted in Supplementary Note 6 can be checked against the data [[experiments.smf-dmf-tmf-001.traditional_ml-summary_table]]
- [x] Measured the persistent-entity corpora (UniProt, NCBI nt/RefSeq, PubChem, wwPDB) straight from the public archives so Fig 1c's bit counts are reproducible rather than asserted, and pinned them to a dated snapshot that can be cited if a referee challenges them [[experiments.016-information-accounting.scripts.persistent_entity_corpus_sizes]]
- [x] Wrote the information-accounting Supplementary Note behind Fig 1c, which says exactly what the bit counts do and do not mean and keeps compressed storage separate from usable supervision, so the new numbers do not quietly break the identifiability argument [[paper.information-accounting]]
- [x] Corrected the supplementary-citation style across the whole manuscript after finding that "Fig. S1" is a form Nature explicitly rejects; SI floats now number from 1 and are cited only through cross-reference macros, so inserting a Note can no longer silently invalidate the references to it [[paper.proof-writing-standard]]
- [x] Documented the true-size SVG export that stops draw.io silently shrinking every matplotlib panel to 72 percent of its authored width, which had been quietly breaking the panel-width contract every figure depends on [[torchcell.utils.utils]]
- [x] Built the Fig 7 classical-ML dataset panels showing each benchmark's perturbation-order makeup and its 80/10/10 stratified split, so the performance panels beside them are read on equal footing [[experiments.002-dmi-tmi.scripts.dataset_composition_palette]]
- [x] Established repo-wide figure standards - strict panel-width templates, loose height, boxed plots, and one ordered palette - as `torchcell.utils` constants so every plot pulls from a single source [[torchcell.utils]]
- [x] Consolidated the figure colors into one ordered green-free draw.io line-and-fill reference matching Fig 1 and repainted the classical-ML bar and progression panels to line-color faces with solid black hatching and tenth gridlines [[notes.assets.scripts.generate_color_palette]]

## 2026.07.15

## 2026.07.17

- [x] Computed the minimum doubles set-cover for the panel-12 triples so only 11 doubles (not all 66) need constructing to build every one of the 52 constructible triples, an 83% build reduction [[experiments.010-kuzmin-tmi.scripts.optimized_doubles_setcover]]
- [x] Built an SMF reference table (Costanzo/Kuzmin fitness ± s.d.) for the droplet validation panel, recovering the SPH1/YLR313C and LCL1/YPL056C fitness values the inference panel never queried [[experiments.010-kuzmin-tmi.scripts.validation_panel_smf_reference]]
- [x] Added a shared-axis ridgeline of the 12 Costanzo SMF Gaussians so each gene's fitness spread is visually comparable across the panel [[experiments.010-kuzmin-tmi.scripts.panel12_smf_gaussian_ridgeline]]
