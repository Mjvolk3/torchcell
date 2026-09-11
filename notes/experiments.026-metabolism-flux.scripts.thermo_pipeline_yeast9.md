---
id: nj4xaildrlrqxot65hqekmc
title: Thermo_pipeline_yeast9
desc: ''
updated: 1789101193157
created: 1789101193157
---

## 2026.09.11 - What this script is

The yeast-specific driver for [[torchcell.metabolism.thermo_pipeline]]. Builds the resolver
stages out of the 026 artifacts (the eQuilibrator recomputation, the re-priced external-pKa
targets, the written-structure lipids, the composition surrogates, the shipped metabolite
column, and a diagnosis-only stage that turns the audit's blocker into a reason enum), runs
the pipeline, and writes the coverage ceiling and one exclusion record per unpriced
metabolite.

Coverage numbers: [[experiments.026-metabolism-flux.pka-method-selection]].
