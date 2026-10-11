---
id: spcev8uw9y7vgv55rpjrsyb
title: Baba2006_release_inventory
desc: ''
updated: 1791695660991
created: 1791695660991
---

## 2026.10.11 - Baba 2006 release inventory and raw-mirror deposit

Candidate row "Baba 2006" (*E. coli*, "Modality / backbone"): Baba T, et al., Mori H. Construction of Escherichia coli K-12 in-frame, single-gene knockout mutants: the Keio collection. Mol Syst Biol 2006;2:2006.0008, doi:10.1038/msb4100050, PMC1681482.

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/baba2006_release_inventory.py` (`--deposit` re-runs every recorded retrieval; idempotent by sha256). Results: `experiments/036-dataset-fixes-before-kg-build/results/baba2006_release_inventory.json`. Loader: [[torchcell.datasets.ecoli.baba2006]].

### Mirror route

The paper is in the literature mirror (`babaConstructionEscherichiaColi2006`, Zotero item present), but `si/si1.pdf` there is a second copy of the article: no Supplementary Table is mirrored. The PMC Article Datasets bucket holds the whole supplement under `PMC1681482.1/` (`msb4100050-s1.pdf` to `-s10.xls`, md5s matching the bucket's own JSON); which object is which table is read from the article XML's `<supplementary-material>` captions. Deposited in `$DATA_ROOT/torchcell-raw/babaConstructionEscherichiaColi2006/` through `retrieve.pmc_cloud_object`, nothing to Zotero:

| file | table | sha256 (prefix) |
|---|---|---|
| `data/msb4100050-s4.xls` | Supplementary Table 2, deletion primers of the targeted ORFs | `897599a5` |
| `data/msb4100050-s5.xls` | Supplementary Table 3, verification, growth and re-served calls | `a1a9c30b` |
| `data/msb4100050-s8.xls` | Supplementary Table 6, essential gene candidates | `ded72d1f` |
| `paper/PMC1681482.1.txt` | PMC plain text, the quote anchor | `6e39bc2f` |

### Measured off the bytes

| quantity | count |
|---|---|
| Table 2 rows (distinct JW ids) | 4,288 |
| Table 3 rows | 4,378 = 4,288 targeted + 90 `N.A.` |
| Table 6 rows | 303 |
| targeted JW ids equal Table 2's | yes |
| `7. percent` = 100 x verified / tested | 4,288 of 4,288 |
| colonies tested | 4 on 382 ORFs, 8 on 3,906 |
| 2 or more verified (obtained) | 3,985 |
| fewer (not obtained) | 303 = Table 6; 302 with 0, secM with 1 of 8 |
| obtained with all three OD600 readings | 3,984 |
| JW ids on one BW25113 locus | 4,116 (125 not carried, 47 colliding) |

Every quote the inventory relies on (10) is located verbatim in the PMC text and recorded with its sha256 in the results JSON.
