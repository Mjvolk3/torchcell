---
id: 2hrt1g0nlv07ajfe366xquo
title: Test_datasets_nucleotide_transformer
desc: ''
updated: 1790780684625
created: 1790780684625
---

## 2026.09.30 - Phase 18: windows, the broken fresh build, main

Zero to six tests (9 cases), 17 to 94 percent alone. The 5979 symmetric and max windows plus the 3' 300 and 5' 1003 windows on the fixture genome with the exact sequences, the stored values, caching, the `main()` call order (wandb, `load_dotenv`, the genome, the six dataset builds with exact paths).

Findings: a fresh build raises `AttributeError: 'NucleotideTransformerDataset' object has no attribute 'transformer'`, because `process` runs inside `super().__init__` (line 53) and reads `self.transformer` (112), which is only set at 60 (the window tests put the stand-in on the class attribute to get past it); the `has_special_codon` and `is_max_size` flags are unpacked but never passed on, so the 3' window starts at 112 rather than 109; the 5' window on the `+` strand ends at the 1-based start and includes the gene's first base (`s288c.py` line 285).

## 2026.09.30 - Findings retired (issue #543)

Retired: the fresh-build `AttributeError` (the class-level stand-in workaround is gone), the dropped codon flag, the missing space. Now asserted: a fresh build creates exactly one backbone and writes the store; a cached store is read with a backbone that raises if built and `transformer` stays `None`; the prime windows carry the codon (YAL001W 3' `[109, 409)`, 5' `[0, 103)`, YAL002C 3' `[0, 23)`, 5' `[29, 1032)`, YAL003W 3' `[5097, 5397)`, 5' `[1000, 2003)`); a new test asserts YAL001W's exact 3' string `CDS[9:12] + chrI[112:409]` and 5' string `chrI[0:100] + CDS[0:3]`; index and gene-id lookup agree.
