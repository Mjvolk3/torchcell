---
id: ebacxf7p0mpzgcpvijzxb61
title: test-suite-buildout-p24
desc: ''
updated: 1791359076285
created: 1791359076285
---

## 2026.10.07

- [x] PR-24 of [[plan.test-suite-buildout.2026.09.25]]: the scattered non-main lines: 160 new hermetic test cases over 54 test files in four lanes (CLIs and small modules, models and losses, dataset loaders with the new [[tests.torchcell.datasets.test_loader_mains]], image processing and the KG builder), each pinning an exact value or message; four independent Opus 5.5 audits (92 accept, 18 notes applied, 1 rewrite, 3 rejects removed); coverage config now omits the `torchcell/scratch` carve-out and excludes abstract-method and Protocol stub bodies; ten source findings recorded (dead helpers and branches, a constant-label crash in the weighted distribution loss, DCell and DCellOpt disagreeing on an input-less term, a float32 overflow in the gated combination); TOTAL 97.7% to 98.8% line+branch (99.4% line) on 48,338 statements; record in [[test-campaign.2026.09.25]]
