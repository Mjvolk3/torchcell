---
id: pfnjsyhmkb1w3zts0s24nsx
title: Test_one_hot_gene
desc: ''
updated: 1791269272189
created: 1791269272189
---

## 2026.10.06 - Phase 21 tests

- Fixture: `embedding_genome` (YAL001W, YAL002C, YAL003W); store `processed/one_hot_gene.pt` in `tmp_path`.
- Gene i stores `eye(3)[i]` as a `[1, 3]` float32 row; a pre-transform doubling the row is applied once per gene in order; a cache hit never calls `process`.
- Finding (one_hot_gene.py:52-54): the post-init rebuild branch calls `initialize_transformer`, which no class defines; unreachable because PyG has already processed.
- Reach (audit 1): dead code. Audit 1 revision: a sentinel `initialize_transformer` patched onto the class records no call during a fresh build.
