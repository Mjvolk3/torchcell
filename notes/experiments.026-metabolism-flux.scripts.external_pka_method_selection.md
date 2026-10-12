---
id: hwvjpvhsz8j92klcevzy0cu
title: External_pka_method_selection
desc: ''
updated: 1789101170786
created: 1789101170786
---

## 2026.09.11 - What this script is

Chooses the pKa method the yeast-GEM formation-energy fill-in uses, on 2,400 eQuilibrator
cache compounds stratified by pKa count, functional group, net charge at pH 7, molecular
weight and a lipid-like axis, with the earlier seed-17 300 pinned in as a fixed subset.
Scores every tool and five ensembles at pH 7.2 and at each working compartment pH, then
applies a stated selection rule and writes the decision as a pydantic record.

Full method, tables, figure and decision:
[[experiments.026-metabolism-flux.pka-method-selection]].
