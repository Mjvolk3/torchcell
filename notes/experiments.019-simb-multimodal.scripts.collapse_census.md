---
id: qbbysxncvok82f6898t0n1j
title: Collapse_census
desc: ''
updated: 1791362350602
created: 1791362350602
---

## 2026.10.07 - Collapse census of v21 and v22

Reads every run of W&B projects `torchcell_019_expr_v21` and `torchcell_019_expr_v22`, finds the launch epoch (validation prediction spread ratio reaching 0.05), every dead stretch after launch (consecutive logged epochs with the ratio below 0.01), and marks a run collapsed at a stretch of 50 epochs or more, the rule of [[experiments.019-simb-multimodal.scripts.v22_readout]]. Per stretch it stores whether it was still open at the last logged epoch, whether the ratio later returned to 0.05, and the per-epoch gradient norm at the onset against the 50 epochs before. Writes `results/collapse_census.json` and `notes-tex/figure-3-gate/tables/collapse_census.tex` (Section 9 of the Figure 3 gate document).

First run, 2026.10.07 01:50 (v21: 108 pre-swap runs plus the 104 relaunched at epochs 491 to 663; v22: 27 runs): 352 dead stretches, 54 of 50 epochs or more, of which 10 later recovered; 3 of the 29 at 100 or more. Onsets concentrate at epochs 150 to 350. The gradient norm at the onset is a median 1.14 times the preceding 50-epoch mean (quartiles 0.96 to 1.48; 5 of 54 at 2 or more), which corrects the three-run reading of 2026.10.06 that a gradient spike precedes collapse. Clipping (threshold 10) never fired. On the relaunched v21 round: Hadamard 8 of 12 collapsed, no-dropout 5, sink 5, stack 4, basis-64 3, reference 1 of 8; mask, prop2 and proteome 0.
