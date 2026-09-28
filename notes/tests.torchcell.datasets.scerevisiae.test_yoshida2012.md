---
id: f06ltzcq6456b284adck1ud
title: Test_yoshida2012
desc: ''
updated: 1790550009510
created: 1790550009510
---

## 2026.09.27 - The Yoshida 2012 organic-acid loader from its embedded Table 3

The only raw file is `paper.pdf`, whose presence is all PyG checks, so a placeholder is written and the values come from the module-level `TABLE_3` literal; `build_metabolite_s_id_map` (which loads Yeast9) is replaced by a stub that records its argument and returns `s_0001` to `s_0005` for the five acids. Five tests: the records with their organic-acid values and metabolite ids, the reference, the side files, the download paths. Loader coverage from this file 92% (the mirror-copy branch at lines 352 to 353 and `main()` remain). Finding: `download()` returns early without hashing an already-present raw file although its docstring says the sha256 is verified (yoshida2012.py line 334). Phase 7 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
