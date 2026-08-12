---
id: zg8aai5iol9x29r3usn5e0p
title: Create_ffa_species_specific_overlays
desc: ''
updated: 1786496505526
created: 1786496505526
---

## 2026.08.11 - Make the Per-Species Overlays Runnable Off the Original Mac

Draws the per-FFA-species interaction overlays (C14:0, C16:0, C16:1, C18:0, C18:1, Total
Titer) for both the multiplicative and additive models. Its results directory was a
hardcoded `/Users/michaelvolk/...` path from the original Mac authoring, so the script
could not run on GilaHyper at all. It now resolves from `EXPERIMENT_ROOT`.

Part of making the whole 008 script set machine-portable so the FFA analysis can continue
on GilaHyper -- see [[experiments.008-xue-ffa.scripts.ffa_epistatic_path_panels]].
