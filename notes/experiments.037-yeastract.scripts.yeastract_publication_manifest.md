---
id: 2il8ztyaipncd7q41lvxvb4
title: Yeastract_publication_manifest
desc: ''
updated: 1791153239443
created: 1791153239443
---

## 2026.10.04 - One row per PMID in the YEASTRACT+ flat file

Reads the *S. cerevisiae* member straight out of `YeastractPlus_2022_all_regs.zip` and refuses to run if either the zip or the decompressed member does not match its pinned sha256. Groups the rows by PMID (column 5) and counts rows, factors, targets, pairs, evidence labels, signs, assays and conditions per paper.

Citation fields come from NCBI `esummary`, with the raw responses stored in `yeastract_pubmed_esummary.json`. A PMID with no PubMed record comes back as an `error` record; it is kept and reported (`pubmed_found = False`), not dropped.

SPELL membership is a PMID match against `experiments/015-spell/results/spell_publications.csv`. `REANALYSES` holds the papers known to reprocess another paper's data, each with the verbatim sentence that says so. It holds one entry and is not a survey of all papers.

Used by [[experiments.037-yeastract.publications]].
