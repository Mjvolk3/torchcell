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

Reordered 2026-09-27 after review around the study's two questions: round 1b (17 runs) = random graphs at KL 1 and 0.1 with five seeds, KL 10 and 100, the missing 1e-5 seed; round 2 (18 runs) = reach for both mechanisms, 2-hop and 3-hop mask and 2-hop and 3-hop KL targets on identical supports, plus the direction pair; round 3 (45 runs) = placement (optional), 60-epoch budget, composite pair, and hidden width 360 on table and composite with and without KL 1. 80 runs, about 4,800 GPU-h. Pre-launch check for reach: tabulate the support density of every graph at 1, 2 and 3 hops (the STRING channels are dense).

Implemented 2026-09-27 (same worktree): `khop_reach` plus four flags in the model (`graph_regularization.hops`, `.symmetrize`, `attention_mask.hops`, `.symmetric`; defaults reproduce every earlier run; 7 tests in `tests/torchcell/models/test_cgt_graph_entry.py`), configs `cgt_s0_r_kl_emb_040` (composite, no fitness head) and `cgt_s0_r_kl_emb_w360_041` (composite at 360, preprocessor h 644), and the plan now carries `config`, `overrides`, `hours` per arm and writes `scripts/delta_submit_round2.sh` (80 sbatch lines, six `after:+30` chains, round 1b first in every chain; `ROUNDS=` and `DRY=1`). All 26 distinct config-override sets compose under Hydra. Reach masks are directed after the density table ([[experiments.025-solid-growth.scripts.graph_reg_khop_density]]); real-size CPU smoke: [[experiments.025-solid-growth.scripts.graph_reg_flags_smoke]].
