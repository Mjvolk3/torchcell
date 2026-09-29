---
id: i0253eig7vykqmoppn4oon5
title: Test_embedding
desc: ''
updated: 1790649314379
created: 1790649314379
---

## 2026.09.28 - Embedding datasets and their addition on a toy model (Phase 9)

8 tests. An invalid model name is refused before any directory is made (message `Invalid model_name 'bad'.Valid options are: toy, other`, no space after the period, lines 30 to 31); a dataset processes once and is reused on a second construction, indexed by position or gene id; no model name means an empty dataset whose processed file is `None.pt` (line 60); `__add__` merges disjoint keys per gene, `sum` with the zero identity reaches the same combination, duplicate keys and foreign operands are refused (the duplicate-key message repeats the key once per record, lines 148 to 155). Findings: `__add__` mutates the LEFT operand (lines 131 and 140) because PyG's `InMemoryDataset.get` returns a shallow copy sharing the `dna_windows` dict, so after `toy + other` the item `toy[0].dna_windows` carries the other's keys and repeating the addition raises the duplicate-key error; adding datasets over different gene sets raises `KeyError('other')` from PyG's collate (line 162). Coverage 21.9% to 98%. Phase 9 of [[plan.test-suite-buildout.2026.09.25]], recorded in [[test-campaign.2026.09.25]].
