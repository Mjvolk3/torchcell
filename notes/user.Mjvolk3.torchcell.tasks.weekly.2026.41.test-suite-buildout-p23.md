---
id: kq6owkhol0fwrhfhe34jljx
title: test-suite-buildout-p23
desc: ''
updated: 1791330893380
created: 1791330893380
---

## 2026.10.06

- [x] PR-23 of [[plan.test-suite-buildout.2026.09.25]]: the demo mains leave the library: 15 `main()` demos (hydra training loops of the 005 and 006 era, the five `neo4j_cell` explorations, the toy forward passes) moved verbatim into `torchcell/scratch/<module>_demo.py` with a pointer left in each module ([[torchcell.scratch.dango_demo]], [[torchcell.scratch.hetero_cell_bipartite_dango_gi_lazy_demo]], [[torchcell.scratch.neo4j_cell_demo]] and twelve more, each with its origin note); no behavior change, two independent Opus 5.5 audits (15 of 15 verbatim, one mypy gate fix), TOTAL 92.4% to 97.7% on 48,599 statements; record in [[test-campaign.2026.09.25]]
