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
