---
id: avtsltb29kdyvv1al9929a8
title: Morphology_noise_ceiling
desc: ''
updated: 1787795258538
created: 1787795258538
---

## 2026.08.26 - morphology_noise_ceiling

Per-feature reliability of the Ohya CalMorph target from 122 wild-type replicates against 4,718 mutants. Mean ceiling over the 278 modeled features is 0.611, with 201 above 0.5. Now also dumps results/morphology_noise_ceiling.json, a per-feature CSV, and a SCALAR-TARGET SHORTLIST ranked on ceiling times robust CV, so a single-feature warm-up picks a feature that is both reliable and actually moved by deletions.

Run from repo root:

```bash
PYTHONPATH=. python experiments/019-simb-multimodal/scripts/morphology_noise_ceiling.py
```

Context: [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
