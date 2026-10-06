---
id: d9vb1po52fcr9ucjgncepe9
title: Yeastract_publication_tables
desc: ''
updated: 1791153243014
created: 1791153243014
---

## 2026.10.04 - LaTeX tables for notes-tex/yeastract-publications

Reads `experiments/037-yeastract/results/yeastract_publications.csv` and writes `counts.tex` (macros for every number in the prose), `t1-summary.tex` (by relation to SPELL), `t2-sizes.tex` (by rows per paper), `t3-leverage.tex` (papers outside SPELL to 95% of the rows outside SPELL) and `t4-all.tex` (every paper) under `notes-tex/yeastract-publications/tables/`.

Same conventions as [[experiments.015-spell.scripts.spell_publication_tables]]: each non-ASCII PubMed character maps to its LaTeX form and an unmapped one raises, and authors, title, journal and the YEASTRACT assay string are wrapped in `\sourcetext{}` so `notes-tex/common/check_doc.py` leaves source spelling alone. SICI-form DOIs carry angle brackets, which are percent-encoded in the link.

Used by [[experiments.037-yeastract.publications]].
