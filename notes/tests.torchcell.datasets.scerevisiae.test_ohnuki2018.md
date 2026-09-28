---
id: md779zftey2ciiy1av7jm13
title: Test_ohnuki2018
desc: ''
updated: 1790549985561
created: 1790549985561
---

## 2026.09.27 - The Ohnuki 2018 essential-gene heterozygote CalMorph loader

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` (YAL001C and YDR001C current, YBR002C renamed to YBR001C, YCR001W retired). Four tests: the heterozygous records, the renamed and retired handling, the side files, the download refusal. Loader coverage from this file 94% (the mirror-copy branch at line 166 and `main()` remain). Finding: a missing CalMorph cell is stored as `0.0` (ohnuki2018.py line 295), where Ohya 2005 (line 265) and Ohnuki 2022 (line 259) drop the row. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
