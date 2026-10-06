---
id: gv3ydrulmxfdwbnwhy8e0sg
title: Spell_publication_tables
desc: ''
updated: 1791138604544
created: 1791138604544
---

## 2026.10.04 - LaTeX tables for notes-tex/spell-publications

Reads `experiments/015-spell/results/spell_publications.csv` and writes `counts.tex` (macros for every number in the prose), `t1-summary.tex`, `t2-no-geo.tex` and `t3-all.tex` under `notes-tex/spell-publications/tables/`.

PubMed text is Unicode and the documents are typeset in T1 Latin Modern, so each non-ASCII character maps to its LaTeX form and an unmapped one raises. Authors, title and journal are wrapped in `\sourcetext{}`, which `notes-tex/common/check_doc.py` skips, so source spelling ("Mech Ageing Dev", the surname Storey) is left as published.

Used by [[experiments.015-spell.publications]].
