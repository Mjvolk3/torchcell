---
id: 4a56wjvhtoopyf50y02hdka
title: Test_ohnuki2022
desc: ''
updated: 1790549977946
created: 1790549977946
---

## 2026.09.27 - The Ohnuki 2022 quadruple-deletion CalMorph loader

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` (YAL001C, YBR001C and the background gene YGL013C current; YDR012W renamed to the background gene YDR011W; YCR001W retired). Three tests: the records with the 3Delta background genotype, the side files, the download refusal. Loader coverage from this file 86%; the mirror-copy branch (lines 207 to 214) and `main()` remain. Finding: `gene_set.json` includes the three 3Delta background genes (ohnuki2022.py line 336); a missing CalMorph cell drops the row (line 259). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
