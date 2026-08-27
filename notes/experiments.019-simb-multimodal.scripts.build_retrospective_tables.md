---
id: pkhw3cvjcmtvzht295x9sq5
title: Build_retrospective_tables
desc: ''
updated: 1787795236593
created: 1787795236593
---

## 2026.08.26 - build_retrospective_tables

Emits the two generated tables of notes-tex/019-simb-multimodal: the strand summary (ceiling, best, fraction realized, epoch budget) and the paired betaxanthin/metabolome arm table. The paired table flags cells whose two arms ran unequal epoch counts and cells whose control never learned the task, and excludes both from the headline mean.

Run from repo root:

```bash
PYTHONPATH=. python experiments/019-simb-multimodal/scripts/build_retrospective_tables.py
```

Context: [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
