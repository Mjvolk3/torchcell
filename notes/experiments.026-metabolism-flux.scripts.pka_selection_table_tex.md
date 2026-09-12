---
id: brwfyaygmaalfr96l7dcyn1
title: Pka_selection_table_tex
desc: ''
updated: 1789186932785
created: 1789186932785
---

## 2026.09.11 - Generated pKa selection tables

Emits `pka_method_selection.tex` (one row per candidate method with its failure rate,
worst per-pH p90, pH 7.2 statistics, ladder-length agreement and the rule's verdict) and
`pka_p90_by_ph.tex` (the same p90 at each of the eight working compartment pHs).

Reads `results/pka_method_selection.json` and
`results/pka_method_selection_figure.json`. A method with no display name in the script
raises rather than printing a slug.
