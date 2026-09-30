---
id: 71e9yjm1vwafvtfgyekv538
title: Test_cell_data
desc: ''
updated: 1790773240650
created: 1790773240650
---

## 2026.09.30 - Phase 16: gene-ontology conversion, strata, the hypergraph cases

Three (all data-gated) to twelve tests, 3 to 85 percent alone (97 with the synthetic file). The import-time `load_dotenv()` is gone; each data test is marked individually and loads it only when it runs. Gene-ontology conversion with exact tensors, the strata printouts, the cycle fallback, hypergraph and bipartite cases with no GPR edge.

Findings: the GO feature `x` counts genes missing from the base graph (line 324); the sink-peeling loop can never run, so a child of a cycle gets the cycle's stratum (260-280).
