---
id: rljsgu9k3ycbfwh88no2949
title: Graph_reg_khop_density
desc: ''
updated: 1790495575047
created: 1790495575047
---

## 2026.09.27 - How far the nine graphs reach

Measured on the 6,607 genes and the nine gene-gene graphs the 025 trainer builds, with the model's own `khop_reach`: ordered pairs reachable within 1, 2, 3 steps, as stored and with edges made undirected. Writes `results/graph_reg_khop_density.json` and `notes-tex/025-graph-reg-sweep/tables/t5-khop-density.tex`.

| graph | edges | 1 hop | 2 hops | 3 hops |
|---|---|---|---|---|
| physical | undirected | 0.6% | 61% | 75% |
| regulatory | directed | 0.09% | 0.4% | 1.0% |
| regulatory | undirected | 0.2% | 46% | 82% |
| tflink | directed | 0.5% | 2.6% | 3.2% |
| tflink | undirected | 0.9% | 59% | (see json) |
| string neighborhood | undirected | 0.7% | 8% | 11% |
| string fusion | undirected | 0.05% | 0.6% | 3.4% |
| string cooccurence | undirected | 0.05% | 0.1% | 0.3% |
| string coexpression | undirected | 4.6% | 69% | 94% |
| string experimental | undirected | 3.8% | 67% | 82% |
| string database | undirected | 0.3% | 3.7% | 16% |

Consequence for round 2: a symmetric two-hop mask is close to no mask on physical, coexpression, experimental (and on regulatory and TFLink once symmetrized), so the reach arms use the directed k-hop support for the mask as the prior always has, with the directed one-hop mask as their anchor. The plan and Section 6 of `notes-tex/025-graph-reg-sweep` carry this ([[experiments.025-solid-growth.scripts.graph_reg_round2_plan]]).
