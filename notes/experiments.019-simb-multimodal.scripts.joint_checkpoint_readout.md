---
id: 8awsfam0oipx8atbyfzskyp
title: Joint_checkpoint_readout
desc: ''
updated: 1791151999413
created: 1791151999413
---

## 2026.10.04 - Checkpoint readout of the joint and conditioned rounds

`experiments/019-simb-multimodal/scripts/joint_checkpoint_readout.py` scores v19 by its
registered per-head windows (proteome epochs 200 to 400, expression 1,000 to 1,200), paired
over the partitions whose three arms finished, reports v20 as partial against the v19
controls at the same epochs, and collects the single-deletion triangle from the two
covariation result files. It writes `results/joint_checkpoint_readout.json` and the four
tables of section 8 of the expression document
(`notes-tex/019-simb-multimodal-expression/sections/6-checkpoint.tex`).

At 11 of 12 partitions: expression superiority $-0.001$ (SE 0.009, 5 of 11 positive, one-sided
$p$ 0.55); proteome non-inferiority at 0.015, $-0.017$ (SE 0.006, 1 of 11 positive, $p$
0.64). Neither rejects. Partition 11 was still training. The four runs of canary 2423179
sit in the v20 project with no history and are excluded by id.
