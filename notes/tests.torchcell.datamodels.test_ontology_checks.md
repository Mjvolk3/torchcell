---
id: 647qgtb2vr0vlem7c78k4e8
title: Test_ontology_checks
desc: ''
updated: 1790773225511
created: 1790773225511
---

## 2026.09.30 - Phase 16: every check on a fake schema module

New file, sixteen tests, 99 percent alone. A fake schema module of plain `BaseModel` classes is swapped in for `oc.s` (no real schema class is subclassed): orphans, lanes, back edges, enum collisions, the phenotype label map, the three adapter AST readers, media checks on a patched library, collisions by name and by InChIKey, the join-key census.

Findings: a non-literal dict key reads as "no properties" instead of "unreadable", producing a false mismatch (lines 503-504); a record with no compounds counts as "every compound identified" (889-898).
