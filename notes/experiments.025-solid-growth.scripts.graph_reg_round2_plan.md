---
id: nxxqpnigvlmhpxwryv0ye9f
title: Graph_reg_round2_plan
desc: ''
updated: 1790490618340
created: 1790490618340
---

## 2026.09.27 - Round 2 of the graph-regularization study, planned

The single source of the round-2 design: a pydantic `Arm` list (question, name, override on `ctrl_013`, code change needed, seeds, epochs, planned panel), the budget at the measured 27 min/epoch on four A40s, `tables/t4-round2.tex` for the document, `results/graph_reg_round2_plan.json` for the launcher to read, and a wireframe of the three planned figures (`graph_reg_round2_mockup.svg`, no data). Nothing is measured; nothing is submitted until approved.

Revised the same night with a `round` column (1b finishes the current figure; 2 the mechanism figure; 3 budget and representation), the shifted placement [S S S M M S S S] in place of the all-8 mask, and the composite-embedding pair. Arms (55 runs, about 3,456 GPU-h; 1b 13 runs, 2 27, 3 15): random graphs at KL 1 and 0.1 (biology vs conditioning where the effect is); KL 10 and 100 (turnover); symmetric-target KL 1 and a directed mask (direction: the KL target is directed, the mask is symmetrized); two-hop mask (reach); mask and KL 1 on layers 1-2, 3-4 and 1-4 (placement, the encoder has 8 layers and the graphs enter layer 1 only); 60-epoch none / mask / KL 1 (budget, 48 h clock); composite embedding with and without KL 1 (representation); KL 1e-5 seed 2 (completion). The round-1 random arm ran the full 30 epochs on both complete seeds, so f is at matched budget. Three small model changes: `graph_regularization.symmetrize`, `attention_mask.symmetric=false`, `attention_mask.hops=2`.

![](./assets/images/025-solid-growth/graph_reg_round2_mockup.svg)

Document: `notes-tex/025-graph-reg-sweep/sections/6-round2.tex`. Readings of round 1: [[experiments.025-solid-growth.graph-reg-sweep]].
