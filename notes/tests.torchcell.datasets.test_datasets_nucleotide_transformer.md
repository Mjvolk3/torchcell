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
