---
id: xdnpxppiuegueh9o0nw0l8s
title: Test_constraints
desc: ''
updated: 1790549296458
created: 1790549296458
---

## 2026.09.27 - Constraints on a two-reaction cobra toy

Thirteen tests on the smallest cobra model built inside the test (two reactions, two metabolites, gene-protein-reaction rules with AND and OR terms): exact bounds, exact constraint matrices and exact error messages for the previously uncovered branches. Coverage of the module from this file 99%; the one missed line (constraints.py 452, an OR term whose genes are all absent from `model.genes`) cannot be produced through a cobra model because cobra registers every gene named in a rule. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
