---
id: oxqn4lof2hxtyjxpm6s8d24
title: Test_ohya2005
desc: ''
updated: 1790549962179
created: 1790549962179
---

## 2026.09.27 - The Ohya 2005 CalMorph loader on two synthetic SCMD matrices

Both TSVs are written into `<root>/raw/`; the genome stub implements only `resolve_gene_name` and returns real `GeneNameResolution` objects (YAL001C, YDR001C, YER001W, YFR001W current; YBR002C renamed to YBR001C; YER002W renamed to YER001W, which is also a strain, so both keep their names; YCR001W retired). Four tests: the records with their CalMorph parameter dicts, the renamed and retired handling, the side files, the download refusal. A missing CalMorph cell drops the row (ohya2005.py line 265), which [[tests.torchcell.datasets.scerevisiae.test_ohnuki2018]] contrasts with the 2018 loader storing 0.0. Loader coverage from this file 94% (the mirror-copy branch at line 179 and `main()` remain). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
