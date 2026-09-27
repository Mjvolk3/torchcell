---
id: nxxqpnigvlmhpxwryv0ye9f
title: Graph_reg_round2_plan
desc: ''
updated: 1790490618340
created: 1790490618340
---

## 2026.09.27 - Round 2 of the graph-regularization study, planned

The single source of the round-2 design: a pydantic `Arm` list (question, name, override on `ctrl_013`, code change needed, seeds, epochs, planned panel), the budget at the measured 27 min/epoch on four A40s, `tables/t4-round2.tex` for the document, `results/graph_reg_round2_plan.json` for the launcher to read, and a wireframe of the three planned figures (`graph_reg_round2_mockup.svg`, no data). Nothing is measured; nothing is submitted until approved.

Arms (46 runs, about 2,970 GPU-h): random graphs at KL 1 and 0.1 (biology vs conditioning where the effect is); KL 10 and 100 (turnover); symmetric-target KL 1 and a directed mask (direction: the KL target is directed, the mask is symmetrized); two-hop mask (reach); mask on layers 1-2, 1-4, all 8 and KL 1 on layers 1-2, 1-4 (placement, the encoder has 8 layers and the graphs enter layer 1 only); 60-epoch none / mask / KL 1 (budget, 48 h clock); KL 1e-5 seed 2 (completion). Three small model changes: `graph_regularization.symmetrize`, `attention_mask.symmetric=false`, `attention_mask.hops=2`.

![](./assets/images/025-solid-growth/graph_reg_round2_mockup.svg)

Document: `notes-tex/025-graph-reg-sweep/sections/6-round2.tex`. Readings of round 1: [[experiments.025-solid-growth.graph-reg-sweep]].
