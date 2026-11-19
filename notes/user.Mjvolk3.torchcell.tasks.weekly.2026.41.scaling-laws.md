---
id: yh6cx76wdju95kojdhcdk9a
title: scaling-laws
desc: ''
updated: 1791533063921
created: 1791533063921
---


## 2026.10.09

- [x] Started the scaling-laws explainer as a notes-tex document in a new `modeling/` group, with the Kaplan and Chinchilla forms, counting rules for `N`, `D`, `C` on strain data, the minimal ablation, and a plan for the trigenic build [[scaling-laws]]
- [x] Wrote the 3 x 3 figure of both forms, nine GIF sweeps, one per panel, typeset by LaTeX, and the generated constants table [[experiments.041-scaling-laws.scripts.scaling_law_forms]]
- [x] Kaplan 2020, Hoffmann 2022 and Besiroglu 2024 added to the library by hand and mirrored by `lit_sync` (2026.10.11); the script note opens with a LaTeX introduction to the equations read from them [[experiments.041-scaling-laws.scripts.scaling_law_forms]]
- [ ] Yang 2022 (muP) and Muennighoff 2023 remain flagged external
- [x] Added "How to run the sweeps" to the script note: the data, parameter and IsoFLOP sweeps on the 025 disjoint split, the replicate floor on `E`, hold-out and bootstrap fitting, and the three-panel figure the sweeps produce; none run yet [[experiments.041-scaling-laws.scripts.scaling_law_forms]]
- [x] Bootstrap GIF now runs 100 resamples on panel i's own draws with an inset histogram of the resampled exponent; a panel-to-GIF coherence table and a "what counts as more data" section (multi-label, multi-input, pooled datasets, repetition) added to the script note [[experiments.041-scaling-laws.scripts.scaling_law_forms]]
