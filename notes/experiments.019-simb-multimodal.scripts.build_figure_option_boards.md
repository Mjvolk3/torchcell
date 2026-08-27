---
id: qbvr936zz1ew1umwlwl06ad
title: Build_figure_option_boards
desc: ''
updated: 1787795251217
created: 1787795251217
---

## 2026.08.26 - build_figure_option_boards

Generates notes/assets/drawio/Fig3-options.drawio and Fig6-options.drawio. Page 1 of each is a PANEL BANK of every candidate panel at true print size; the remaining pages are alternative compositions. Panels are embedded as base64 SVG data URIs, and geometry is read from each SVG's own root, so a panel authored 179 mm wide lands 179 mm wide (draw.io's unit is 1/100 inch, which is what savefig_true_size_svg writes).

Run from repo root:

```bash
PYTHONPATH=. python experiments/019-simb-multimodal/scripts/build_figure_option_boards.py
```

Context: [[experiments.019-simb-multimodal.phenotype-strand-retrospective]]
