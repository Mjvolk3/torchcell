---
id: llib2j2t009oxze37jdonzo
title: Spell_knockout_coverage
desc: ''
updated: 1791148570404
created: 1791148570404
---

## 2026.10.04 - Deletion coverage estimated from PCL column headers

A SPELL condition is described only by its column header, so deletion coverage is an estimate from text. A header is a deletion condition when a deletion marker (`-del`, a delta sign, `delta`, `deletion`, `knockout`, `null`, `KO`, or a lowercase gene name plus `D`) is attached to a token that resolves to a gene in the SGD R64-4-1 GFF (sha256 `64f61e3153083a8ef6d853721c9e83e4469cdc120883ec281e51a0df4ba390fa`): systematic name, standard name, or an alias of exactly one gene. Headers that are only gene names are counted separately as `gene_named` and never as deletions.

Measured on the 2026.10.04 run (n = 603 studies, 16,160 headers):

- 107 studies with a deletion call, 2,349 deletion conditions, 685 distinct deleted genes.
- 53 of the 685 are absent from the 1,484 mutants of Kemmeren 2014 Table S1 (sha256 `885b96ce...`), spread over 36 studies.
- 9 studies delete ten or more genes. Largest: Hu 2007 (269), Lenstra 2011 (162), van Wageningen 2010 (133), Apweiler 2012 (91), Sameith 2015 (82), Chua 2006 (51).
- 7 studies have gene-named conditions (522 conditions, 496 genes), almost all in Hughes 2000 (276 genes) and Mnaimneh 2004 (215 genes).

Known errors, none quantified: misses marker-less deletions (`ptc1 vs wt`); counts marked non-deletion alleles (`SIC1Δ3P`, `TAF1-(delta)TAND`, `rpb1-del-plasm-...`). No call was checked against its paper. Every call is in `experiments/015-spell/results/spell_knockout_conditions.csv`.

Used by [[experiments.015-spell.scripts.spell_publication_tables]] and [[experiments.015-spell.publications]].
