---
id: gv3z7tmhbjw7y6br57s2vbz
title: Gerdes2003_release_inventory
desc: ''
updated: 1791695668861
created: 1791695668861
---

## 2026.10.11 - Gerdes 2003 release inventory and raw-mirror deposit

Candidate row "Gerdes 2003" (*E. coli*, "Transposon fitness"): Gerdes SY, et al., Osterman AL. Experimental determination and system level analysis of essential genes in Escherichia coli MG1655. J Bacteriol 2003;185:5673-5684, doi:10.1128/JB.185.19.5673-5684.2003, PMC193955, PMID 13129938.

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/gerdes2003_release_inventory.py` (`--deposit` re-runs every recorded retrieval and renders the two PDFs). Results: `experiments/036-dataset-fixes-before-kg-build/results/gerdes2003_release_inventory.json`. Loader: [[torchcell.datasets.ecoli.gerdes2003]].

### Mirror route

Not in either Zotero library and not in the literature mirror; nothing was added to Zotero. The article body is walled for scripts (measured 2026-10-10: no PMC open-access copy, `journals.asm.org` 403, Europe PMC 403 and 500), filed as #925. The authors' Supplementary Online Material is served openly and is deposited in `$DATA_ROOT/torchcell-raw/gerdesExperimentalDeterminationSystem2003/` by `direct_url`:

| file | content | sha256 (prefix) |
|---|---|---|
| `data/table_s1.xls` | Table S1, 4,291 ORF rows | `2c4c2457` |
| `si/supplementary_table.html` | the Table S1 page: E, N, X, ? and the assertion error defined | `f2998227` |
| `si/genetic.pdf` + `.pdftotext.txt` | procedure 1: strain, transposon, fermentor medium and conditions | `2e8cb6f0` / `932b56c3` |
| `si/assertion_of_essentiality.pdf` + `.pdftotext.txt` | procedure 2: the assertion criteria | `cf73ff78` / `f19345e2` |
| `paper/pubmed_13129938.xml` | PubMed's record (abstract) | `294205fd` |

Every byte-stability check passed: each file fetched twice with the same sha256.

### Measured off the bytes

| quantity | count |
|---|---|
| Table S1 rows | 4,291 |
| calls | E 619, N 3,114, N* 6, X 331, ? 221 |
| abstract / overview | 620 essential, 3,126 dispensable; 3,746 unambiguous |
| rows without a Blattner ID | 64 (2 E, 41 N, 9 X, 12 ?) |
| Blattner ids on two rows | 9 (18 rows) |
| Baba 2006 Table 3's re-served call, same b-number | 4,113 identical of 4,133 shared |
| distinct single Blattner ids resolved on MG1655 | 4,167 of 4,209 |

11 quotes checked verbatim (PubMed abstract, both renderings, the Table S1 page).
