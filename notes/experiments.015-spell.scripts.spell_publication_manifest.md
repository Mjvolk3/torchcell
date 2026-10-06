---
id: 0oyovjuei6hizprs82tgu0k
title: Spell_publication_manifest
desc: ''
updated: 1791138601060
created: 1791138601060
---

## 2026.10.04 - One row per SPELL study, DOI resolved from PubMed

Parses each study README (PMID, GEO field, PCL dataset table) into pydantic records, resolves DOI, PMC id and journal citation from the PMID through NCBI E-utilities `esummary` (POST, 200 PMIDs per request), and writes `experiments/015-spell/results/spell_publications.csv`. The raw responses and the retrieval time are kept in `spell_pubmed_esummary.json`.

Two README quirks the parser handles:

- The `GEO ID:` field is a bare series id in 199 READMEs, a PCL filename that starts with one (`GSE9136.final.pcl`, `GSE34330GPL8154.sfp.pcl`) in 316, and `N/A` in 88. The series id is read from the start of the field.
- A README with several PCL files repeats `File last modified:` after each dataset row, so rows are read to the end of the file.

Used by [[experiments.015-spell.publications]].
