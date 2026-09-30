---
id: fjxtreg2jflsscmduki9sc3
title: kg-release-versioning
desc: ''
updated: 1790716468848
created: 1790716468848
---

## 2026.09.29

- [x] T4 of [[plan.data-release-program.2026.09.29]] (issue #467): `torchcell_version` and `torchcell_tag` in `KgRelease`, the manifest and `stamp`/`write-node` ([[torchcell.knowledge_graphs.releases]], [[torchcell.knowledge_graphs.kg_manifest]]); committed release snapshots ([[torchcell.knowledge_graphs.release_snapshot]]) bootstrapped for `2026.09.21-ab6d8c5d`; the generated compatibility page ([[scripts.kg_compat_page]]) linked from the README; semantic-release moved to a parser that reads `TAG(scope):` subjects ([[scripts.release_parser]], [[versioning]]).
- [x] Bumps are deliberate: `REL` minor, `DB` patch, `API` major, everything else no bump ([[versioning]], second 2026.09.29 section; [[scripts.release_parser]]; `tests/scripts/test_release_parser.py`; the contributing guide table). The `FEAT`/`FIX` map had moved `main` 1.2.1 to 1.5.0 in one evening.
