---
id: krckj1xurqnpg3afthxk1mm
title: '28'
desc: ''
updated: 1783476309116
created: 1783476309116
---

## 2026.07.07

- [x] Built a reusable paper-table generator that renders one registry to markdown + LaTeX and computes a gzip "signal" (Kolmogorov proxy) straight from each built LMDB, filling every dataset row including the 20.7M-record Costanzo tables [[torchcell.paper.tables]] [[paper.nature-biotech.scripts.generate_datasets_table]]

## 2026.07.08

- [ ] Reconcile stale curated genotype counts the generator surfaced (Kuzmin 2020 dmf 632,797 vs 256,862; Kemmeren 1,450; Zelezniak metabolome built) into [[paper.supported-datasets-and-databases]]
- [ ] revitalized supported-datasets table with verified class/KG-adapter status, verbatim DOIs/URLs, and new ingest targets for the Zotero-backed rebuild [[torchcell.datasets.supported-datasets-table-revitalized]]
- [ ] Implement CI/quality foundation -- ruff (E,F,I,UP,D), runnable mypy (defer 7,601-error cleanup), pytest/coverage config, repair broken `src/`-path CI; supersedes `feat/literature-zotero-ocr` tooling [[plan.ci-foundation-ruff-mypy-pytest.2026.06.18]]
- [ ] Make the pytest CI job blocking (`#16`): harden 4 CI-fragile tests (test_s288c module-level DATA_ROOT guard, wall-clock benchmark `@pytest.mark.gpu`, targeted DATA_ROOT skipif, filelock cleanup rewrite) + CPU-only wheel install (torch/scatter from CPU indexes) + remove `continue-on-error`, then add `pytest-coverage` to main's required checks post-merge [[plan.pytest-ci-blocking.2026.07.01]]

## 2026.07.13

- [x] Natural-isolate genomic diversity vs KO-expression variability -- bit accounting for CGT inputs (`#66`): per-ORF divergence, core/accessory, codon usage, coding-vs-regulatory π from the population VCF, and a single-KO-vs-natural-isolate DE comparison under Kemmeren's own sourced criterion [[experiments.018-natural-isolate-genomics]]
- [x] Corrected three overclaims from that first pass (Signal composition, gzip order-effect, a phantom isolate drop) and, in the process, confirmed two real loader defects -- see the Corrections section of [[experiments.018-natural-isolate-genomics]]
- [ ] `#71` caudal2024 silently omits ~133 gene-absence edits per isolate (`s288c_mask` computed but never used; both loops guard on `core_mask`) -- **genome-fidelity bug, needs a Caudal LMDB rebuild**
- [ ] `#72` sameith2015: 70/287 arrays (24%) enter sign-flipped -- GSE42536 is a dye-swap design and GEO declares BOTH ratio directions; the global sign is CORRECT (do not flip), the fix is to recompute per array from `Signal Norm_Cy5`/`Cy3` as kemmeren2014 already does
- [ ] `#73` caudal2024 `SACE_`-prefixed FASTA headers -- silent `continue` would drop an isolate's entire genotype; latent today (none of the 93 fall in the built 943)
