---
id: zc006a9tgsdjzqvfrxxv6y8
title: test_protT5
desc: ''
updated: 1790780551045
created: 1790780551045
---

## 2026.09.30 - Phase 18: the ProtT5 dataset on a faked backbone

New file, six tests (8 cases), 25 to 90 percent alone, the same fixture and stand-in as the ESM2 file. Findings: the `_no_dubious_uncharacterized` variant lists classes in lowercase (lines 229-232) and excludes nothing; embedded genes are stored as numpy arrays (309) while excluded genes are torch `zeros(1, 1024)` with a hard-coded width (304), so the collate keeps a Python list and `ds["gene"]` returns a one-element list; with `model_name=None` the backbone is built on every construction because `initialize_model` runs before the early return (284-286) and nothing is stored.

## 2026.09.30 - Findings retired (issue #543)

Retired: mixed numpy/torch storage with a 1024-wide zero row, lowercase exclusion list, list-valued lookup, missing space, backbone built for `model_name=None`. Now asserted: one `[3, 4]` float32 tensor with the excluded row `zeros(1, 4)` at the stand-in's `hidden_size`; `_no_dubious_uncharacterized` embeds only YAL001W; gene-id and index lookup return the same `[1, 4]` tensor; the spaced message; `model_name=None` builds no backbone across two constructions.
