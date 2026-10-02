---
id: w3k8q1zt5h9d2m7cx4bn6va
title: website-benchmark-scaffold
desc: ''
updated: 1790895906254
created: 1790895906254
---

## 2026.10.01

- [x] Scaffold of the public website and the benchmark service, plan and runbook in [[plan.website-benchmark-platform.2026.10.01]]: the `tc-bench` FastAPI service [[torchcell.benchmark.app]] (accounts with email confirmation, prediction uploads validated by the pydantic contract in [[torchcell.benchmark.submission]], grader [[torchcell.benchmark.grading]], quota of 3 per 24 h and 1 h apart in [[torchcell.benchmark.ratelimit]], integrity flags, provisional or verified board, zip archive per scored submission), `Dockerfile.tc-bench` + `docker-compose.tc-bench.yml` (PostgreSQL with no published port, API on loopback), and the Docusaurus site under `website/` with nine tabs; 143 hermetic tests. Nothing deployed.
- [ ] Choose the three starting datasets, export record ids and pinned splits into bundles, run the kNN and linear-regression baselines.
- [ ] Decide hosting (GitHub Pages subpath or the Radiant proxy) and the account policy, then follow the runbook.

## 2026.10.02

- [x] Ontology explorer rework in [[paper.nature-biotech.scripts.generate_ontology_diagram]]: a class lists the served datasets that use it, a dataset view lights its classes and links to its dataset page and loader API page, from the release closures through the new [[torchcell.paper.ontology_usage]] (9 tests); click hit-testing, glide, resize re-fit, pinch zoom. Paper SVGs byte-identical.
- [x] Website: cell-only favicon written by `docs/make_logo.py`, a navbar page-width toggle (default reading width, or the whole window), the Ontology tab embeds the configurable explorer, dataset cards link to their classes in it. Checked in headless Chromium: 18 of 18 site checks, including the leaderboard charts with mock data.
- [x] Radiant VM: `trash` graveyard and heavy scratch default to Taiga under `torchcell/tmp/`; tc-bench data roots default beside the tc-data store.
