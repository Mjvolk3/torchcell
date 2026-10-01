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
