---
id: 47bc0lvqesnqoqei6vvxdt2
title: Test_dasilveira2014
desc: ''
updated: 1790549954207
created: 1790549954207
---

## 2026.09.27 - The da Silveira dos Santos 2014 lipidome loader on synthetic tables

Both workbooks are written with openpyxl into `<root>/raw/`; the genome stub carries the two attributes the loader reads, `gene_set` (YAL001C, YBR001C) and `alias_to_systematic` (YBR002C to YBR001C, YAL003W to YAL001C). Five tests: records by `model_dump()` equality with the lipid values, the reference, the side files, the alias resolution, the download refusal on the real digest. Loader coverage from this file 92%; the successful mirror-copy branches (lines 183, 189, 195) need the real sha256-pinned files and `main()` a real `DATA_ROOT`. Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
